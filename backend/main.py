"""
OneCode AI - FastAPI backend.

Endpoints:
  GET  /health              - liveness check
  POST /match                - run the pipeline (bundled sample data, or an uploaded CSV) and store the result
  GET  /results              - last run's pairwise results, optional ?decision= filter
  GET  /clusters             - proposed CNMC clusters from the last run
  GET  /audit                - audit log (proposals + review actions)
  POST /review                - approve/reject/modify a proposed pair; writes to the audit log
  GET  /savings               - procurement savings estimate from the last run
  GET  /dashboard-stats        - summary numbers for the frontend stat strip

Run locally:  uvicorn main:app --reload --port 8000
Deploy (Render): start command ->  uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Optional, Literal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pipeline

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(APP_DIR, "data", "sample_materials.csv")

app = FastAPI(title="OneCode AI - Material Resolution API", version="1.0")

# Wide-open CORS for the hackathon demo -- tighten origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for the last pipeline run + review actions.
# Fine for a demo; swap for a real DB (Postgres) before production use.
STATE = {"last_output": None, "reviewed": {}}


class ReviewRequest(BaseModel):
    pair: list[str]                       # e.g. ["ONGC-1001", "SAIL-1001"]
    action: Literal["approve", "reject", "modify"]
    reviewer: str = "demo_user"
    note: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "semantic_method": pipeline.USE_REAL_EMBEDDINGS and "MiniLM" or "TF-IDF"}


@app.post("/match")
def run_match(file: Optional[UploadFile] = File(None)):
    """Run the full pipeline. If a CSV is uploaded, use it; otherwise use the bundled sample dataset."""
    csv_path = DEFAULT_CSV
    tmp_path = None
    try:
        if file is not None:
            if not file.filename.endswith(".csv"):
                raise HTTPException(400, "Only .csv files are supported right now.")
            tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{datetime.now().timestamp()}.csv")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            csv_path = tmp_path

        output = pipeline.run_pipeline(csv_path)
        STATE["last_output"] = output
        STATE["reviewed"] = {}
        return {
            "message": "Pipeline run complete.",
            "total_records": output["total_records"],
            "candidate_pairs_evaluated": output["candidate_pairs_evaluated"],
            "compute_saved_by_blocking_pct": output["compute_saved_by_blocking_pct"],
            "decision_breakdown": output["decision_breakdown"],
            "semantic_method": output["semantic_method"],
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _require_run():
    if STATE["last_output"] is None:
        raise HTTPException(400, "No pipeline run yet. Call POST /match first.")
    return STATE["last_output"]


@app.get("/results")
def get_results(decision: Optional[str] = None):
    output = _require_run()
    results = output["results"]
    if decision:
        results = [r for r in results if r["decision"].lower() == decision.lower()]
    # overlay any review actions taken since the run
    for r in results:
        key = tuple(sorted(r["pair"]))
        if key in STATE["reviewed"]:
            r = {**r, "review_status": STATE["reviewed"][key]["action"]}
    return {"count": len(results), "results": results}


@app.get("/clusters")
def get_clusters():
    output = _require_run()
    return {"clusters": output["clusters"]}


@app.get("/audit")
def get_audit():
    output = _require_run()
    log = list(output["audit_log"])
    log.extend(STATE["reviewed"].values())
    log.sort(key=lambda x: x["timestamp"])
    return {"count": len(log), "audit_log": log}


@app.post("/review")
def review_pair(req: ReviewRequest):
    output = _require_run()
    key = tuple(sorted(req.pair))
    valid_pairs = {tuple(sorted(r["pair"])) for r in output["results"]}
    if key not in valid_pairs:
        raise HTTPException(404, f"Pair {req.pair} not found in the last pipeline run.")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": f"human_{req.action}",
        "materials": list(req.pair),
        "reviewer": req.reviewer,
        "note": req.note,
        "status": "reversible",  # governance: every human action is logged and can be rolled back
    }
    STATE["reviewed"][key] = entry
    return {"message": f"Recorded '{req.action}' for {req.pair}.", "entry": entry}


@app.get("/savings")
def get_savings():
    output = _require_run()
    return output["savings"]


@app.get("/dashboard-stats")
def dashboard_stats():
    output = _require_run()
    return {
        "total_records": output["total_records"],
        "candidate_pairs_evaluated": output["candidate_pairs_evaluated"],
        "full_pairwise_would_be": output["full_pairwise_would_be"],
        "compute_saved_by_blocking_pct": output["compute_saved_by_blocking_pct"],
        "decision_breakdown": output["decision_breakdown"],
        "total_estimated_savings_inr": output["savings"]["total_estimated_savings_inr"],
        "clusters_proposed": len(output["clusters"]),
        "semantic_method": output["semantic_method"],
    }
