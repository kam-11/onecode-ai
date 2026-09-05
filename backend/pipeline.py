"""
OneCode AI - Material Resolution Pipeline (core logic).
Importable module used by main.py (FastAPI). Same architecture validated in the
Kaggle notebook: normalize -> category ontology -> blocking (retrieval) ->
hard-block compatibility -> hybrid scoring -> 5-way decision -> CNMC + savings.

Semantic similarity uses sentence-transformers/MiniLM when available (internet
required to download the model on first run); falls back to TF-IDF otherwise so
the service still starts and works without it.
"""
import re
import itertools
import hashlib
import json
from datetime import datetime, timezone
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz

# ---------------- Semantic backend (auto-detect) ----------------
USE_REAL_EMBEDDINGS = True
_model = None
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    USE_REAL_EMBEDDINGS = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    print(f"[pipeline] sentence-transformers unavailable ({e}); using TF-IDF fallback.")

# ---------------- Normalization ----------------
ABBREVIATIONS = {
    r"\bBRG\b": "BEARING", r"\bDGBB\b": "DEEP GROOVE BALL BEARING",
    r"\bBLT\b": "BOLT", r"\bGR\b": "GRADE", r"\bGR\.": "GRADE",
    r"\bSS\b": "STAINLESS STEEL", r"\bCS\b": "CARBON STEEL",
    r"\bHEX\b": "HEXAGONAL", r"\bSQMM\b": "SQ MM",
    r"\bCLASS(\d)": r"CLASS \1",
}

def normalize_text(s):
    s = str(s).upper()
    s = re.sub(r"[^\w\s.]", " ", s)
    for pattern, repl in ABBREVIATIONS.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

UNIT_TO_BASE = {"mm": 1.0, "inch": 25.4, "model": None, "sqmm": 1.0}

def normalize_dimension(dim, unit):
    if unit in ("model",):
        return str(dim)
    try:
        parts = [float(p) for p in re.split(r"[xX]", str(dim))]
        factor = UNIT_TO_BASE.get(unit, 1.0) or 1.0
        return "x".join(f"{p*factor:.1f}" for p in parts)
    except (ValueError, TypeError):
        return str(dim)

# ---------------- Category ontology ----------------
CRITICAL_ATTRS = {
    "Bearing": [], "Fastener": ["grade"], "Pipe": ["grade", "pressure_rating"],
    "Valve": ["pressure_rating"], "Cable": ["voltage_rating"],
}
STRUCTURAL_ATTRS = {
    "Bearing": ["material", "norm_dimension", "standard"],
    "Fastener": ["material", "norm_dimension", "standard", "grade"],
    "Pipe": ["material", "norm_dimension", "standard", "grade", "pressure_rating"],
    "Valve": ["material", "norm_dimension", "standard", "pressure_rating"],
    "Cable": ["material", "norm_dimension", "standard", "voltage_rating"],
}
FAMILY_HINTS = {
    "Bearing": ["OPEN", "SEALED", "SHIELDED", "2RS", "ZZ"],
    "Fastener": [], "Pipe": ["THREADED", "WELDED", "FLANGED"],
    "Valve": [], "Cable": [],
}
BRAND_KEYWORDS = ["SKF", "FAG", "NSK", "TIMKEN", "NTN"]
CNMC_PREFIX = {"Bearing": "BRG", "Fastener": "FAS", "Pipe": "PIP", "Valve": "VLV", "Cable": "CBL"}

def extract_brand(desc):
    return next((b for b in BRAND_KEYWORDS if b in desc), None)

def has_missing_data(row, category):
    if row["material"] == "" or row["standard"] == "":
        return True
    return any(row.get(a, "") == "" for a in CRITICAL_ATTRS.get(category, []))

def critical_conflict(a, b, category):
    for attr in CRITICAL_ATTRS.get(category, []):
        va, vb = str(a.get(attr, "")).strip(), str(b.get(attr, "")).strip()
        if va and vb and va != vb:
            return attr, va, vb
    return None, None, None

def attribute_agreement(a, b, category):
    attrs = STRUCTURAL_ATTRS.get(category, ["material", "norm_dimension"])
    compared = matched = 0
    for attr in attrs:
        va, vb = str(a.get(attr, "")).strip(), str(b.get(attr, "")).strip()
        if va and vb:
            compared += 1
            matched += (va == vb)
    return (matched / compared) if compared else 0.0

def family_difference(a, b, category):
    hints = FAMILY_HINTS.get(category, [])
    tags_a = {h for h in hints if h in a["norm_description"]}
    tags_b = {h for h in hints if h in b["norm_description"]}
    return tags_a != tags_b

def generate_cnmc(category, material, dimension, standard):
    key = f"{category}|{material}|{dimension}|{standard}".upper()
    h = hashlib.md5(key.encode()).hexdigest()[:6].upper()
    return f"CNMC-{CNMC_PREFIX.get(category,'GEN')}-{h}"

def decide(a, b, category, score, dim_match, attr_score):
    conflict_attr, va, vb = critical_conflict(a, b, category)
    if conflict_attr:
        return "Not Equivalent", f"Critical attribute '{conflict_attr}' conflicts: '{va}' vs '{vb}'"
    if has_missing_data(a, category) or has_missing_data(b, category):
        return "Insufficient Data", "One or both records missing a required attribute"
    if not dim_match:
        return "Not Equivalent", "Dimension/model does not match after unit normalization"
    if family_difference(a, b, category):
        return "Near Duplicate", "Same core spec but a real variant difference (seal/end type) — treat as family, not identical"

    perfect_attrs = attr_score >= 0.999
    if perfect_attrs or score >= 0.85:
        brand_a, brand_b = extract_brand(a["norm_description"]), extract_brand(b["norm_description"])
        if brand_a and brand_b and brand_a != brand_b:
            return "Functional Equivalent", f"Matching spec (attribute agreement {attr_score:.0%}) but different manufacturer ('{brand_a}' vs '{brand_b}')"
        basis = "structured attributes match exactly" if perfect_attrs else f"high combined similarity ({score:.2f})"
        return "Exact Duplicate", f"{basis.capitalize()}; attribute agreement {attr_score:.0%}, similarity {score:.2f}"
    elif score >= 0.55:
        return "Near Duplicate", f"Moderate similarity ({score:.2f}); needs human confirmation"
    else:
        return "Not Equivalent", f"Low similarity ({score:.2f})"

def blocking_candidates(df):
    blocks = {}
    for idx, row in df.iterrows():
        blocks.setdefault(row["category"], []).append(idx)
    return [pair for idxs in blocks.values() for pair in itertools.combinations(idxs, 2)]

def run_pipeline(csv_path):
    df = pd.read_csv(csv_path).fillna("")
    df["norm_description"] = df["raw_description"].apply(normalize_text)
    df["norm_dimension"] = df.apply(lambda r: normalize_dimension(r["dimension_1"], r["unit"]), axis=1)

    if USE_REAL_EMBEDDINGS:
        embeddings = _model.encode(df["norm_description"].tolist(), normalize_embeddings=True)
        def semantic(i, j):
            return float(np.dot(embeddings[i], embeddings[j]))
    else:
        vectorizer = TfidfVectorizer().fit(df["norm_description"].tolist())
        def semantic(i, j):
            tf = vectorizer.transform([df.loc[i, "norm_description"], df.loc[j, "norm_description"]])
            return float(cosine_similarity(tf[0], tf[1])[0][0])

    pairs = blocking_candidates(df)
    results, audit_log, clusters = [], [], {}

    for i, j in pairs:
        a, b = df.loc[i], df.loc[j]
        category = a["category"]
        dim_match = a["norm_dimension"] == b["norm_dimension"]
        fuzzy_s = fuzz.token_sort_ratio(a["norm_description"], b["norm_description"]) / 100.0
        semantic_s = semantic(i, j)
        attr_score = attribute_agreement(a, b, category)
        combined = round(0.3 * fuzzy_s + 0.2 * semantic_s + 0.5 * attr_score, 4)
        decision, evidence = decide(a, b, category, combined, dim_match, attr_score)

        result = {
            "pair": [a["material_code"], b["material_code"]],
            "cpse_pair": [a["cpse_id"], b["cpse_id"]],
            "descriptions": [a["raw_description"], b["raw_description"]],
            "category": category, "combined_score": combined,
            "fuzzy_score": round(fuzzy_s, 4), "semantic_score": round(semantic_s, 4),
            "attribute_agreement": round(attr_score, 4),
            "decision": decision, "evidence": evidence,
            "review_status": "pending_human_review" if decision in ("Exact Duplicate", "Functional Equivalent") else "no_action_needed",
        }
        results.append(result)

        if decision in ("Exact Duplicate", "Functional Equivalent"):
            cnmc = generate_cnmc(category, a["material"], a["norm_dimension"], a["standard"])
            clusters.setdefault(cnmc, set()).update(result["pair"])
            audit_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "candidate_cluster_proposed",
                "materials": result["pair"], "cnmc": cnmc,
                "decision": decision, "status": "pending_human_review",
            })

    n = len(df)
    full_pairs = n * (n - 1) // 2

    # savings estimate (illustrative, not a guarantee)
    savings_rows = []
    for cnmc, codes in clusters.items():
        sub = df[df.material_code.isin(codes)]
        if len(sub) < 2:
            continue
        min_p, max_p = float(sub.unit_price.min()), float(sub.unit_price.max())
        qty = int(sub.quantity_procured.sum())
        savings_rows.append({
            "cnmc": cnmc, "cpses_involved": sub.cpse_id.tolist(),
            "price_range": [min_p, max_p], "combined_quantity": qty,
            "estimated_savings_inr": round((max_p - min_p) * qty * 0.5, 2),
            "note": "Illustrative estimate: 50% of price gap x combined quantity. Not a guaranteed figure.",
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "semantic_method": "sentence-transformers/MiniLM" if USE_REAL_EMBEDDINGS else "TF-IDF (fallback)",
        "total_records": n, "candidate_pairs_evaluated": len(pairs),
        "full_pairwise_would_be": full_pairs,
        "compute_saved_by_blocking_pct": round((1 - len(pairs) / full_pairs) * 100, 1) if full_pairs else 0,
        "decision_breakdown": dict(Counter(r["decision"] for r in results)),
        "results": results,
        "audit_log": audit_log,
        "clusters": {k: sorted(v) for k, v in clusters.items()},
        "savings": {
            "clusters": savings_rows,
            "total_estimated_savings_inr": round(sum(r["estimated_savings_inr"] for r in savings_rows), 2),
        },
    }
    return output
