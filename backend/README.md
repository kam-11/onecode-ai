# OneCode AI — Backend API

FastAPI service wrapping the material resolution pipeline (normalize → category
ontology → blocking → hard-block checks → hybrid scoring → 5-way decision →
CNMC generation → savings estimate).

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API docs (Swagger UI) —
you can call every endpoint from the browser without writing any client code.

## Endpoints

| Method | Path              | Purpose                                                   |
|--------|-------------------|------------------------------------------------------------|
| GET    | /health           | Liveness check + which semantic engine is active          |
| POST   | /match            | Run the pipeline (bundled sample CSV, or upload your own) |
| GET    | /results          | Pairwise results; optional `?decision=Exact Duplicate`    |
| GET    | /clusters         | Proposed CNMC clusters from the last run                  |
| GET    | /audit            | Full audit log (proposals + human review actions)         |
| POST   | /review           | Approve / reject / modify a proposed pair                 |
| GET    | /savings          | Procurement savings estimate (illustrative)                |
| GET    | /dashboard-stats  | Summary numbers for a frontend stat strip                  |

Call `POST /match` once after starting the service (or after uploading a new
CSV) before calling the GET endpoints — they read from the last run's result.

## Semantic engine

By default (`requirements.txt`) this uses TF-IDF cosine similarity for the
semantic-similarity signal — lightweight, no model download, safe for Render's
free tier.

To use real sentence-transformers/MiniLM embeddings instead, install
`requirements-full.txt`. Only do this on a paid hosting tier or locally —
on Render's free 512MB tier, torch + the model download will likely exceed
memory and fail to deploy.

## Deploy on Render (free tier)

1. Push this `backend/` folder to a GitHub repo.
2. On render.com: New → Web Service → connect the repo.
3. Root directory: `backend` (if the repo has other folders alongside it).
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Deploy. You'll get a permanent URL like `https://onecode-ai.onrender.com`.
7. Free tier sleeps after inactivity — hit `/health` a few minutes before a
   live demo to warm it up.

## Uploading a real dataset

`POST /match` accepts an optional CSV upload with the same columns as
`data/sample_materials.csv` (cpse_id, material_code, raw_description,
category, material, dimension_1, unit, standard, grade, pressure_rating,
voltage_rating, unit_price, quantity_procured). If no file is sent, it runs
on the bundled synthetic sample.
