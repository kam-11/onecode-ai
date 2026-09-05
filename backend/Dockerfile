# Use this Dockerfile when deploying the FULL version (real sentence-transformers
# embeddings) to a host with more memory than Render's free tier -- e.g.
# Hugging Face Spaces (Docker SDK, free CPU tier has ~16GB RAM) or Railway.
#
# For Render's free tier, skip this file entirely and use requirements.txt
# (TF-IDF only) with the native Python buildpack instead -- see README.md.

FROM python:3.11-slim

WORKDIR /app

COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

COPY . .

# Hugging Face Spaces expects the app to listen on port 7860 by default.
# Render/Railway inject $PORT instead -- this line works for both since
# $PORT falls back to 7860 if it isn't set.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
