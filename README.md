# OneCode AI — One Nation, One Material Code

SIH 2026 · Problem Statement 26099 · Ministry of Petroleum & Natural Gas / CPCL

AI-driven reconciliation layer that resolves duplicate, near-duplicate, and
functionally-equivalent material records across CPSEs into a Common National
Material Code (CNMC), without replacing any CPSE's existing SAP/ERP system.

## Repository structure

```
onecode-ai/
├── backend/              FastAPI service — the working pipeline as an API
│   ├── main.py             API endpoints (/match, /results, /review, /savings, ...)
│   ├── pipeline.py          core matching engine (normalize, ontology, blocking, scoring, decision)
│   ├── generate_dataset.py  regenerates the bundled synthetic 5-category dataset
│   ├── data/sample_materials.csv
│   ├── requirements.txt      LIGHT deps — TF-IDF semantic matching (Render free tier)
│   ├── requirements-full.txt FULL deps — adds sentence-transformers/MiniLM (needs more RAM)
│   ├── Dockerfile            for deploying the FULL version (Hugging Face Spaces / Railway)
│   └── README.md             detailed API + deployment instructions
├── dashboard/            Single-file interactive HTML console (no server needed to view)
│   └── onecode_prototype_dashboard.html
└── notebook/             Kaggle/Colab-ready pipeline (paste as 8 cells, run top to bottom)
    └── onecode_final_kaggle_notebook.py
```

## Where to deploy the backend — decision guide

You need real semantic embeddings (sentence-transformers/MiniLM) for the
strongest possible matching quality, but that needs more RAM than most free
tiers give you by default. Pick based on what you have:

| Host | Tier | Which requirements file | Real embeddings? | Notes |
|---|---|---|---|---|
| **Render** | Free (512MB RAM) | `requirements.txt` | No — auto-falls-back to TF-IDF | Easiest setup, native Python buildpack, no Dockerfile needed. Sleeps after inactivity. |
| **Hugging Face Spaces** | Free (Docker SDK, ~16GB RAM) | `requirements-full.txt` + `Dockerfile` | **Yes** | Best free option for real embeddings. Set Space SDK to "Docker" when creating it. |
| **Railway** | Free trial credits | `requirements-full.txt` + `Dockerfile` | Yes | Good RAM, but free credits run out — fine for a demo window, not long-term free hosting. |
| **Render** | Paid ($7/mo+) | `requirements-full.txt` | Yes | If you want everything on one host and don't mind a small cost. |

**Recommended for SIH:** deploy on **Hugging Face Spaces** with the Dockerfile
for the real-embeddings version — it's free, has enough RAM, and the resulting
URL (`https://huggingface.co/spaces/<you>/onecode-ai`) is just as presentable
as any other live link for judges.

If you want the absolute simplest setup and are fine with TF-IDF (still a
legitimate lexical-similarity method — just say so if asked), Render's free
tier with `requirements.txt` takes about 10 minutes end to end.

### Deploying to Hugging Face Spaces (recommended)

1. huggingface.co → New Space → SDK: **Docker** → name it `onecode-ai`
2. Upload everything inside `backend/` (including `Dockerfile`) to the Space's file browser, or connect it to this GitHub repo's `backend/` folder
3. Space builds automatically from the Dockerfile → you get a live URL
4. Test: `https://<your-space-url>/health` should return `{"status":"ok","semantic_method":"MiniLM"}`

### Deploying to Render (simplest, TF-IDF)

1. render.com → New → Web Service → connect this GitHub repo
2. Root directory: `backend`
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy → test `/health`, should return `{"status":"ok","semantic_method":"TF-IDF"}`

## After the backend is live

Update the `fetch()` calls in `dashboard/onecode_prototype_dashboard.html` to
point at your live backend URL instead of the embedded JSON, then deploy that
HTML file as a static site on Vercel or Netlify (drag-and-drop works, no build
step needed) for the frontend link.

## Running the notebook (Kaggle/Colab)

Paste each `# CELL N` block from `notebook/onecode_final_kaggle_notebook.py`
into its own notebook cell, in order. On Kaggle, turn on Settings → Internet
before running Cell 5 so it can download the MiniLM model — otherwise it
auto-falls-back to TF-IDF and still runs end-to-end.
