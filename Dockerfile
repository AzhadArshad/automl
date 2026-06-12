# Hugging Face Spaces (Docker SDK) — runs FastAPI + Streamlit in one container.
FROM python:3.11-slim

# libgomp1 is the OpenMP runtime required by LightGBM and XGBoost wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

# Spaces runs the container as a non-root user (uid 1000): give it a writable
# HOME and make the app dir writable for outputs/, mlruns/, and temp CSVs.
RUN mkdir -p outputs mlruns && chmod -R 777 /app
ENV HOME=/app \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 7860

CMD ["bash", "start.sh"]
