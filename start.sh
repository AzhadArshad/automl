#!/bin/bash
# Launch the FastAPI backend (internal, port 8000) and the Streamlit UI
# (public, port 7860 — the port Hugging Face Spaces exposes).
set -e

uvicorn serving.api:app --host 0.0.0.0 --port 8000 &

exec streamlit run ui/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
