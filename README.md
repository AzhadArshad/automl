---
title: AutoML
emoji: 🌌
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AutoML — End-to-End AutoML for Tabular Data

Upload any tabular CSV, pick a target column, and get a tuned, explained, ready-to-download ML model — no code required.

**Live demo:** [huggingface.co/spaces/AzhadArshad/automl](https://huggingface.co/spaces/AzhadArshad/automl)

## What it does

From a single CSV upload, the pipeline automatically:

1. **Profiles the data** — classifies columns (numerical / categorical / datetime / text), drops ID columns, flags potential target-leakage columns
2. **Detects the task** — binary, multiclass, or regression, with the right metric (ROC AUC / F1 Macro / RMSE)
3. **Preprocesses** — median imputation, robust scaling, one-hot & ordinal encoding, datetime expansion
4. **Engineers features** _(optional)_ — polynomial interactions on the most target-correlated columns, replayed identically at inference time
5. **Trains a model zoo** — Logistic/Ridge, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost, MLP, KNN, SVM — each with 5-fold cross-validation, in parallel
6. **Tunes the top models** with Optuna hyperparameter search (live per-trial progress)
7. **Ensembles** the tuned models — weighted average, simple average, or stacking with a meta-learner on out-of-fold predictions
8. **Explains predictions** with SHAP — global importance, beeswarm, and single-prediction waterfall plots
9. **Tracks every run** in MLflow — params, metrics, plots, and the final model artifact

## Architecture

```
┌─────────────┐   REST    ┌──────────────┐   background job   ┌─────────────────────┐
│  Streamlit  │ ────────► │   FastAPI    │ ─────────────────► │   AutoML pipeline    │
│   ui/app.py │ ◄──────── │ serving/api  │    live progress   │ ingest → detect →    │
└─────────────┘  polling  └──────────────┘     callbacks      │ preprocess → train → │
                                                              │ tune → ensemble →    │
                                                              │ explain → track      │
                                                              └─────────────────────┘
```

- **FastAPI backend** — training runs as an async background job; the UI polls a status endpoint. One job at a time, with upload size/row/column guards for public hosting.
- **Streamlit frontend** — a thin REST client with no ML logic; caches immutable results per job.
- **Single Docker container** — both services in one image, deployable to Hugging Face Spaces free tier.

## Run locally

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/AzhadArshad/automl.git
cd automl
uv venv && uv pip install -r requirements.txt

# Terminal 1 — API
uv run uvicorn serving.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — UI
uv run streamlit run ui/app.py
```

Open http://localhost:8501.

### Or with Docker

```bash
docker build -t automl .
docker run --rm -p 7860:7860 automl
```

Open http://localhost:7860.

## Use as a Python library

```python
from automl import AutoML

aml = AutoML(time_limit=120, top_n_models=3, n_optuna_trials=20)
aml.fit("titanic.csv", target="Survived")

print(aml.leaderboard())        # sorted CV results for every model
print(aml.explain())            # {feature: mean |SHAP|}, best first
preds = aml.predict(new_df)     # raw, unprocessed DataFrame in — labels out
proba = aml.predict_proba(new_df)

aml.export("model.pkl")         # pickle the fitted ensemble
aml.report("report.html")       # self-contained HTML report
```

## REST API

| Method | Endpoint                | Description                                         |
| ------ | ----------------------- | --------------------------------------------------- |
| POST   | `/train`                | Upload CSV + config, returns `job_id` immediately   |
| GET    | `/status/{job_id}`      | Progress %, current step, error if any              |
| GET    | `/leaderboard/{job_id}` | Sorted model results + task type + metric           |
| POST   | `/predict`              | Single-row JSON prediction (labels + probabilities) |
| POST   | `/predict/batch`        | Upload CSV, get CSV back with a `prediction` column |
| GET    | `/explain/{job_id}`     | SHAP feature importance + plot paths                |
| GET    | `/export/{job_id}`      | Download the full fitted pipeline as `.pkl`         |

## Using a downloaded model

The export is the **full pipeline** (preprocessing + feature engineering + ensemble), so `.predict()` accepts raw data with the original CSV columns. The pickle references this project's classes, so load it from a clone of this repo:

```python
import pickle
import pandas as pd

with open("automl_pipeline.pkl", "rb") as f:
    aml = pickle.load(f)

print(aml.predict(pd.read_csv("new_data.csv")))
```

## Project structure

```
automl/
├── core/            # ingestion & profiling, task detection, preprocessing, feature engineering
├── models/          # model zoo, parallel CV trainer
├── tuning/          # Optuna search spaces & tuner
├── ensemble/        # simple / weighted / stacking ensembles
├── explainability/  # SHAP explainer + plots
├── tracking/        # MLflow logging
├── serving/         # FastAPI backend
├── ui/              # Streamlit frontend
├── automl.py        # orchestrator class (also: python automl.py runs a smoke test)
└── report.py        # self-contained HTML report generator
```

## Hosted demo limits (free tier)

- Max **20 MB / 50k rows / 1k columns** per upload
- One training job at a time
- Models live in memory — train and predict in the same session; restarts clear finished jobs

## Tech stack

scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · SHAP · MLflow · FastAPI · Streamlit · Plotly · Docker
