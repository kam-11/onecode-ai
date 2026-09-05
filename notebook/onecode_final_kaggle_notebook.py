# =====================================================================
# CELL 1 — SETUP
# Paste this as the first cell in Kaggle/Colab and run it.
# On Kaggle: Settings -> Internet -> On (needed to download the MiniLM model).
# If no internet is available, the script auto-falls-back to TF-IDF so it
# still runs end-to-end -- just with a weaker semantic signal.
# =====================================================================
!pip install -q rapidfuzz scikit-learn sentence-transformers pandas

import os
os.makedirs("/kaggle/working/data", exist_ok=True)   # Colab: change to "/content/data"
DATA_DIR = "/kaggle/working/data"


# =====================================================================
# CELL 2 — SYNTHETIC MULTI-CPSE DATASET
# Best of both drafts: 5 categories (ontology proof), deliberate abbreviation
# variety, unit traps, exact/functional/near/conflict/insufficient cases.
# =====================================================================
import csv, random
random.seed(42)

rows = []
counters = {}
def next_code(cpse):
    counters[cpse] = counters.get(cpse, 1000) + 1
    return f"{cpse}-{counters[cpse]}"

def add(cpse, desc, category, material, dim, unit, standard, grade="",
        pressure="", voltage="", price=0, qty=0, cluster=""):
    rows.append(dict(cpse_id=cpse, material_code=next_code(cpse), raw_description=desc,
                      category=category, material=material, dimension_1=dim, unit=unit,
                      standard=standard, grade=grade, pressure_rating=pressure,
                      voltage_rating=voltage, unit_price=price, quantity_procured=qty,
                      true_cluster=cluster))

# BEARINGS
add("ONGC","SKF BRG 6205 2RS","Bearing","Steel","6205","model","ISO 15","N/A",price=420,qty=200,cluster="B1")
add("SAIL","Bearing Deep Groove 6205-2RS SKF","Bearing","Steel","6205","model","ISO 15","N/A",price=445,qty=150,cluster="B1")
add("NTPC","DGBB 6205 2RS","Bearing","Steel","6205","model","ISO 15","N/A",price=410,qty=300,cluster="B1")
add("CPCL","FAG Bearing 6205 2RS","Bearing","Steel","6205","model","ISO 15","N/A",price=460,qty=100,cluster="B2_equiv")
add("GAIL","NSK BRG DGBB 6205-2RS","Bearing","Steel","6205","model","ISO 15","N/A",price=455,qty=120,cluster="B2_equiv")
add("ONGC","SKF BRG 6205 open type","Bearing","Steel","6205","model","ISO 15","N/A",price=390,qty=80,cluster="B3_family")
add("NTPC","Bearing 6205 - spec not confirmed","Bearing","","6205","model","","",price=400,qty=50,cluster="B_insufficient")

# FASTENERS
add("ONGC","HEX BLT M12X50 SS304 DIN933 GR8.8","Fastener","SS304","12x50","mm","DIN933","8.8",price=12,qty=5000,cluster="F1")
add("SAIL","Bolt Hexagonal M12 x 50mm Stainless 304 DIN 933 Grade 8.8","Fastener","SS304","12x50","mm","DIN933","8.8",price=14,qty=3000,cluster="F1")
add("CPCL","BOLT HEX M12*50 SS304 DIN933 8.8","Fastener","SS304","12x50","mm","DIN933","8.8",price=13,qty=4200,cluster="F1")
add("NTPC","HEX BOLT M12x50 SS304 DIN933 GR10.9","Fastener","SS304","12x50","mm","DIN933","10.9",price=16,qty=1000,cluster="F2_conflict")
add("GAIL","Hex Bolt 0.5in x 2in SS304 DIN933 Grade 8.8","Fastener","SS304","12.7x50.8","inch","DIN933","8.8",price=15,qty=1800,cluster="F3_unit")

# PIPES
add("ONGC","SS PIPE 50MM ASTM A312 TP304L","Pipe","Stainless Steel","50","mm","ASTM A312","TP304L",pressure="Sch40",price=850,qty=600,cluster="P1")
add("CPCL","Stainless Steel Pipe 50mm ASTM A312 Grade TP304L Sch40","Pipe","Stainless Steel","50","mm","ASTM A312","TP304L",pressure="Sch40",price=880,qty=400,cluster="P1")
add("SAIL","SS Pipe 50mm ASTM A312 TP304L Sch80","Pipe","Stainless Steel","50","mm","ASTM A312","TP304L",pressure="Sch80",price=1100,qty=250,cluster="P2_conflict")
add("NTPC","SS Pipe 50mm A312 TP304L Sch40 threaded end","Pipe","Stainless Steel","50","mm","ASTM A312","TP304L",pressure="Sch40",price=900,qty=150,cluster="P3_family")
add("GAIL","Pipe 50mm - material grade TBD","Pipe","","50","mm","","",pressure="",price=800,qty=100,cluster="P_insufficient")

# VALVES (new category -- proves the ontology generalizes beyond the original 3)
add("ONGC","GATE VALVE 100MM CLASS150 CS BODY","Valve","Carbon Steel","100","mm","API600","N/A",pressure="150#",price=4200,qty=40,cluster="V1")
add("CPCL","Gate Valve 100mm Class 150 Carbon Steel Body API 600","Valve","Carbon Steel","100","mm","API600","N/A",pressure="150#",price=4350,qty=30,cluster="V1")
add("NTPC","Gate Valve 100mm Class300 CS Body API600","Valve","Carbon Steel","100","mm","API600","N/A",pressure="300#",price=5800,qty=20,cluster="V2_conflict")

# CABLES (new category)
add("SAIL","XLPE CABLE 3C 95SQMM 11KV","Cable","Copper","95","sqmm","IS7098","N/A",voltage="11kV",price=680,qty=2000,cluster="C1")
add("GAIL","XLPE Cable 3 Core 95 sq mm 11 kV IS 7098","Cable","Copper","95","sqmm","IS7098","N/A",voltage="11kV",price=705,qty=1500,cluster="C1")
add("ONGC","XLPE Cable 3C 95sqmm 33kV IS7098","Cable","Copper","95","sqmm","IS7098","N/A",voltage="33kV",price=980,qty=800,cluster="C2_conflict")

with open(f"{DATA_DIR}/sample_materials.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Generated {len(rows)} rows across {len(set(r['category'] for r in rows))} categories")


# =====================================================================
# CELL 3 — NORMALIZATION: abbreviation dictionary + unit handling
# This is the piece the earlier prototype skipped -- expanding engineering
# shorthand BEFORE scoring is what makes fuzzy/semantic similarity reliable
# across CPSEs that abbreviate differently.
# =====================================================================
import re

ABBREVIATIONS = {
    r"\bBRG\b": "BEARING", r"\bDGBB\b": "DEEP GROOVE BALL BEARING",
    r"\bBLT\b": "BOLT", r"\bGR\b": "GRADE", r"\bGR\.": "GRADE",
    r"\bSS\b": "STAINLESS STEEL", r"\bCS\b": "CARBON STEEL",
    r"\bHEX\b": "HEXAGONAL", r"\bC\b": "CORE", r"\bSQMM\b": "SQ MM",
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


# =====================================================================
# CELL 4 — CATEGORY-SPECIFIC ENGINEERING ONTOLOGY
# Each category defines its own critical (hard-block), structural, and
# family-hint attributes -- no single universal rule list.
# =====================================================================
CRITICAL_ATTRS = {
    "Bearing": [],
    "Fastener": ["grade"],
    "Pipe": ["grade", "pressure_rating"],
    "Valve": ["pressure_rating"],
    "Cable": ["voltage_rating"],
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
    "Fastener": [],
    "Pipe": ["THREADED", "WELDED", "FLANGED"],
    "Valve": [],
    "Cable": [],
}
BRAND_KEYWORDS = ["SKF", "FAG", "NSK", "TIMKEN", "NTN"]


# =====================================================================
# CELL 5 — MATCHING PIPELINE: retrieval -> compatibility -> hybrid score -> decision
# Semantic similarity uses sentence-transformers (MiniLM) when available;
# falls back to TF-IDF automatically if there's no internet access.
# =====================================================================
import itertools, json
from datetime import datetime, timezone
import pandas as pd
from rapidfuzz import fuzz

USE_REAL_EMBEDDINGS = True
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_all(texts):
        return _model.encode(texts, normalize_embeddings=True)
    def semantic_sim(vec_a, vec_b):
        return float(np.dot(vec_a, vec_b))
except Exception as e:
    print(f"[info] sentence-transformers unavailable ({e}); falling back to TF-IDF.")
    USE_REAL_EMBEDDINGS = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(f"{DATA_DIR}/sample_materials.csv").fillna("")
df["norm_description"] = df["raw_description"].apply(normalize_text)
df["norm_dimension"] = df.apply(lambda r: normalize_dimension(r["dimension_1"], r["unit"]), axis=1)

if USE_REAL_EMBEDDINGS:
    embeddings = embed_all(df["norm_description"].tolist())
else:
    _vectorizer = TfidfVectorizer().fit(df["norm_description"].tolist())

def get_semantic_score(i, j):
    if USE_REAL_EMBEDDINGS:
        return semantic_sim(embeddings[i], embeddings[j])
    else:
        tfidf = _vectorizer.transform([df.loc[i, "norm_description"], df.loc[j, "norm_description"]])
        return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])

def blocking_candidates(df):
    blocks = {}
    for idx, row in df.iterrows():
        blocks.setdefault(row["category"], []).append(idx)
    return [pair for idxs in blocks.values() for pair in itertools.combinations(idxs, 2)]

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

def extract_brand(desc):
    return next((b for b in BRAND_KEYWORDS if b in desc), None)

def decide(a, b, category, score, fuzzy_s, dim_match, attr_score):
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

pairs = blocking_candidates(df)
results, audit_log = [], []

for i, j in pairs:
    a, b = df.loc[i], df.loc[j]
    category = a["category"]
    dim_match = a["norm_dimension"] == b["norm_dimension"]
    fuzzy_s = fuzz.token_sort_ratio(a["norm_description"], b["norm_description"]) / 100.0
    semantic_s = get_semantic_score(i, j)
    attr_score = attribute_agreement(a, b, category)
    combined = round(0.3 * fuzzy_s + 0.2 * semantic_s + 0.5 * attr_score, 4)

    decision, evidence = decide(a, b, category, combined, fuzzy_s, dim_match, attr_score)

    results.append({
        "pair": [a["material_code"], b["material_code"]],
        "cpse_pair": [a["cpse_id"], b["cpse_id"]],
        "descriptions": [a["raw_description"], b["raw_description"]],
        "category": category,
        "combined_score": combined, "fuzzy_score": round(fuzzy_s, 4),
        "semantic_score": round(semantic_s, 4), "attribute_agreement": round(attr_score, 4),
        "decision": decision, "evidence": evidence,
    })

    if decision in ("Exact Duplicate", "Functional Equivalent"):
        audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "materials": [a["material_code"], b["material_code"]],
            "decision": decision, "status": "pending_human_review",
        })

n = len(df)
full_pairs = n * (n - 1) // 2
output = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "semantic_method": "sentence-transformers/MiniLM" if USE_REAL_EMBEDDINGS else "TF-IDF (fallback, no internet)",
    "total_records": n, "candidate_pairs_evaluated": len(pairs),
    "full_pairwise_would_be": full_pairs,
    "compute_saved_by_blocking_pct": round((1 - len(pairs) / full_pairs) * 100, 1),
    "results": results, "audit_log": audit_log,
}
with open(f"{DATA_DIR}/output.json", "w") as f:
    json.dump(output, f, indent=2)

from collections import Counter
print("Semantic method:", output["semantic_method"])
print("Pairs evaluated:", len(pairs), "/ full pairwise:", full_pairs,
      f"({output['compute_saved_by_blocking_pct']}% saved)")
print(Counter(r["decision"] for r in results))


# =====================================================================
# CELL 6 — DETERMINISTIC CNMC GENERATION (governance layer)
# Only for pairs approved as Exact Duplicate / Functional Equivalent.
# =====================================================================
import hashlib

def generate_cnmc(category, material, dimension, standard):
    key = f"{category}|{material}|{dimension}|{standard}".upper()
    h = hashlib.md5(key.encode()).hexdigest()[:6].upper()
    prefix = {"Bearing":"BRG","Fastener":"FAS","Pipe":"PIP","Valve":"VLV","Cable":"CBL"}.get(category,"GEN")
    return f"CNMC-{prefix}-{h}"

clusters = {}
for r in results:
    if r["decision"] in ("Exact Duplicate", "Functional Equivalent"):
        a_row = df[df.material_code == r["pair"][0]].iloc[0]
        cnmc = generate_cnmc(a_row["category"], a_row["material"], a_row["norm_dimension"], a_row["standard"])
        clusters.setdefault(cnmc, set()).update(r["pair"])

print(f"\n{len(clusters)} canonical CNMC clusters proposed:")
for cnmc, codes in clusters.items():
    print(f"  {cnmc}  <-  {sorted(codes)}")


# =====================================================================
# CELL 7 — EVALUATION: precision/recall + false-merge rate on critical conflicts
# =====================================================================
from sklearn.metrics import precision_recall_fscore_support

code_to_cluster = dict(zip(df["material_code"], df["true_cluster"]))
MERGE_DECISIONS = {"Exact Duplicate", "Functional Equivalent"}

def expected_relation(ca, cb):
    if ca == cb and "conflict" not in ca and "insufficient" not in ca:
        return "should_merge"
    if "conflict" in ca or "conflict" in cb:
        return "must_block"
    if "insufficient" in ca or "insufficient" in cb:
        return "insufficient"
    return "should_not_merge"

y_true, y_pred, conflict_cases = [], [], []
for r in results:
    ca, cb = code_to_cluster.get(r["pair"][0], ""), code_to_cluster.get(r["pair"][1], "")
    if not ca or not cb:
        continue
    expected = expected_relation(ca, cb)
    predicted_merge = r["decision"] in MERGE_DECISIONS
    if expected == "must_block":
        conflict_cases.append(not predicted_merge)
    y_true.append(1 if expected == "should_merge" else 0)
    y_pred.append(1 if predicted_merge else 0)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
false_merge_rate = 1 - (sum(conflict_cases) / len(conflict_cases)) if conflict_cases else 0.0

metrics = {
    "merge_detection_precision": round(precision, 3),
    "merge_detection_recall": round(recall, 3),
    "merge_detection_f1": round(f1, 3),
    "critical_conflict_cases": len(conflict_cases),
    "critical_conflicts_correctly_blocked": sum(conflict_cases),
    "false_merge_rate_on_critical_conflicts": round(false_merge_rate, 3),
}
print(json.dumps(metrics, indent=2))
with open(f"{DATA_DIR}/eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


# =====================================================================
# CELL 8 — PROCUREMENT SAVINGS ESTIMATE (framed as an estimate, not a guarantee)
# =====================================================================
savings_rows = []
for cnmc, codes in clusters.items():
    sub = df[df.material_code.isin(codes)]
    if len(sub) < 2:
        continue
    min_price, max_price = sub.unit_price.min(), sub.unit_price.max()
    total_qty = sub.quantity_procured.sum()
    potential_saving = (max_price - min_price) * total_qty * 0.5  # conservative: half the price gap, as an estimate
    savings_rows.append({
        "cnmc": cnmc, "cpses_involved": sub.cpse_id.tolist(),
        "price_range": [float(min_price), float(max_price)],
        "combined_quantity": int(total_qty),
        "estimated_savings_inr": round(float(potential_saving), 2),
        "note": "Estimate based on synthetic prices/quantities and a conservative 50% price-gap assumption -- not a guaranteed figure.",
    })

total_estimated_savings = sum(r["estimated_savings_inr"] for r in savings_rows)
print(f"\nEstimated collaborative-procurement savings across {len(savings_rows)} clusters: "
      f"Rs {total_estimated_savings:,.0f} (illustrative, synthetic data)")

with open(f"{DATA_DIR}/savings_estimate.json", "w") as f:
    json.dump({"clusters": savings_rows, "total_estimated_savings_inr": total_estimated_savings}, f, indent=2)
