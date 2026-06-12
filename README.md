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

**Live demo:** [huggingface.co/spaces/Aju360/automl](https://huggingface.co/spaces/Aju360/automl)

Built as a production-style portfolio project: a complete AutoML engine (profiling → preprocessing → training → tuning → ensembling → explainability → tracking) wrapped in a FastAPI backend and a Streamlit frontend, shipped as a single Docker container.

---

## Table of contents

- [How the pipeline works](#how-the-pipeline-works)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Run locally](#run-locally)
- [Use as a Python library](#use-as-a-python-library)
- [REST API](#rest-api)
- [Using a downloaded model](#using-a-downloaded-model)
- [Module-by-module guide](#module-by-module-guide)
- [Configuration reference](#configuration-reference)
- [Hosted demo limits](#hosted-demo-limits-free-tier)

---

## How the pipeline works

A single call — `AutoML().fit("data.csv", target="...")` — runs nine stages:

### 1. Ingestion & profiling (`core/ingestion.py`)
The CSV is loaded and every column is classified by inspecting its dtype, cardinality, and uniqueness ratio:

- **Datetime** — object columns where >80% of values parse as dates
- **ID** — integer or string columns with a >95% unique ratio (dropped, with a warning)
- **Text** — string columns with >50 unique values (excluded from modeling)
- **Categorical** — remaining low-cardinality string columns
- **Numerical** — everything numeric that isn't an ID

It also computes null percentages, skewness, and cardinality per column, and flags columns whose *names* suggest target leakage (`target`, `label`, `output`, `predict`).

### 2. Task detection (`core/task_detector.py`)
The target column determines the task and metric:

| Target looks like | Task | Metric |
|---|---|---|
| 2 unique values | Binary classification | ROC AUC |
| 3–20 unique values, int/string | Multiclass classification | F1 Macro |
| Continuous float / many unique ints | Regression | RMSE |

### 3. Preprocessing (`core/preprocessor.py`)
A single sklearn `ColumnTransformer` handles everything, so the exact same transform is replayed at inference:

- Numerical → median imputation + `RobustScaler` (outlier-resistant)
- Categorical (≤15 unique) → constant imputation + one-hot encoding
- Categorical (>15 unique) → ordinal encoding with unknown-value handling
- Datetime → expanded into year / month / day / dayofweek / hour / is_weekend

### 4. Feature engineering — optional (`core/feature_engineer.py`)
Degree-2 polynomial and interaction terms are generated for the **5 numerical columns most correlated with the target**. The chosen columns are persisted on the fitted model and replayed identically at inference (ranking by correlation is impossible at predict time — there's no target — so the fit-time selection is stored, not recomputed). K-fold target encoding for high-cardinality categoricals is also implemented, using out-of-fold means to avoid leakage.

### 5. Model zoo training (`models/model_zoo.py`, `models/trainer.py`)
Up to 9 model families are trained with 5-fold cross-validation (stratified for classification), in parallel via joblib:

> Logistic Regression / Ridge · RandomForest · ExtraTrees · XGBoost · LightGBM · CatBoost · MLP · KNN (≤50k rows) · SVM (≤20k rows)

Slow models are automatically gated by dataset size, and any model that errors is skipped with a warning rather than failing the run. Results are returned as a leaderboard sorted best-first (ascending RMSE for regression, descending score otherwise).

### 6. Hyperparameter tuning (`tuning/optuna_tuner.py`)
The top-N leaderboard models go through Optuna search with per-model search spaces (tree depth, learning rates, regularization, etc.). A trial-level callback streams progress to the UI so the progress bar advances after **every trial**, not every model.

### 7. Ensembling (`ensemble/ensembler.py`)
Three strategies over the tuned top-N models:

- **Weighted average** (default) — predictions weighted by CV score (inverse RMSE for regression, so better models weigh more)
- **Simple average**
- **Stacking** — a meta-learner (LogisticRegression / Ridge) trained on out-of-fold base-model predictions, which prevents the meta-model from seeing leaked in-fold predictions

The fitted ensemble exposes `predict()` / `predict_proba()` with proper class labels and `(n, n_classes)` probability shapes.

### 8. Explainability (`explainability/shap_explainer.py`)
`TreeExplainer` for tree models (fast, exact), `KernelExplainer` fallback for everything else (sampled background for tractability). Produces global mean-|SHAP| importance, a beeswarm summary plot, and a single-prediction waterfall — saved as images and surfaced in both the UI and the HTML report.

### 9. Experiment tracking (`tracking/mlflow_tracker.py`)
Every run logs dataset stats, the full leaderboard, the best model's tuned params, SHAP plots, and the pickled model to MLflow. Tracking failures are non-fatal — the pipeline never dies because logging did.

---

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

- **FastAPI backend** (`serving/api.py`) — `/train` returns a `job_id` immediately and runs training as a background task. A progress callback wired into the AutoML instance updates an in-memory job store in real time; the UI polls `/status` every 3 seconds.
- **Streamlit frontend** (`ui/app.py`) — a thin REST client with zero ML logic. Immutable results (leaderboard, SHAP, model bytes) are cached per job so reruns don't refetch.
- **Single Docker container** — `start.sh` launches uvicorn (internal, port 8000) and Streamlit (public, port 7860). On Hugging Face Spaces only 7860 is exposed, so the API is reachable only through the UI's server-side calls.

### Hardening for public hosting

Since the demo is open to the internet, the API enforces:

- Upload caps: **20 MB / 50,000 rows / 1,000 columns**
- **One training job at a time** (HTTP 429 otherwise) — checked atomically with job creation to avoid a request-interleaving race
- Server-side clamps on tuning effort (`n_optuna_trials ≤ 50`, `top_n_models ≤ 8`), regardless of what the client sends
- **Bounded memory**: only the 3 most recent finished jobs are kept; older fitted models are evicted
- Request timeouts on every UI→API call

---

## Design decisions

A few choices worth calling out:

- **Same-transform guarantee.** Everything that touches features at fit time (datetime expansion → ColumnTransformer → polynomial features) is either a fitted sklearn object or persisted state replayed at inference. This is what makes the downloaded pipeline work on raw CSVs.
- **Leaderboard direction is metric-aware.** RMSE sorts ascending; ROC AUC / F1 sort descending. Ensemble weights are inverted for regression so lower-error models dominate the blend.
- **Failures degrade, never crash.** A model failing CV, SHAP failing on an exotic estimator, or MLflow being unavailable all log warnings and continue — a partial result beats a dead job.
- **The export is the whole pipeline, not just the model.** `/export` pickles the fitted AutoML instance (with the training data stripped out) so `predict()` accepts raw data. A bare ensemble pickle would require the user to reproduce preprocessing themselves — a classic deployment trap.
- **Optuna's objective stays in "higher is better" space** (`neg_root_mean_squared_error` for regression) so a single `direction="maximize"` study works for every task.

---

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

### Quick smoke test (no UI)

```bash
uv run python automl.py
```

Trains on a synthetic dataset end-to-end and writes a model pickle + HTML report.

---

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

---

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

Example — train and predict from the command line:

```bash
# Start a job
curl -F "file=@titanic.csv" -F "target=Survived" http://localhost:8000/train
# → {"job_id": "..."}

# Poll until done
curl http://localhost:8000/status/<job_id>

# Predict a single row
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"job_id": "<job_id>", "features": {"Pclass": 1, "Sex": "female", "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "C"}}'
# → {"prediction": "Survived", "raw_prediction": 1, "probabilities": {...}}
```

---

## Using a downloaded model

The export is the **full pipeline** (preprocessing + feature engineering + ensemble), so `.predict()` accepts raw data with the original CSV columns. The pickle references this project's classes, so load it from a clone of this repo:

```python
import pickle
import pandas as pd

with open("automl_pipeline.pkl", "rb") as f:
    aml = pickle.load(f)

print(aml.predict(pd.read_csv("new_data.csv")))
```

---

## Module-by-module guide

```
automl/
├── core/
│   ├── ingestion.py        # CSV loading, column classification, data profiling, leakage flags
│   ├── task_detector.py    # task type + default metric inference from the target column
│   ├── preprocessor.py     # ColumnTransformer builder: impute, scale, encode, datetime expansion
│   └── feature_engineer.py # polynomial features (persisted column choice) + K-fold target encoding
├── models/
│   ├── model_zoo.py        # {name: estimator} factory, gated by dataset size and task
│   ├── trainer.py          # parallel 5-fold CV over the zoo → sorted leaderboard
│   └── base_model.py       # abstract contract for custom model wrappers
├── tuning/
│   └── optuna_tuner.py     # per-model search spaces, study runner, trial-level progress callbacks
├── ensemble/
│   └── ensembler.py        # simple / weighted / stacking ensembles with OOF meta-features
├── explainability/
│   └── shap_explainer.py   # Tree/Kernel explainer selection, importance + 3 plot types
├── tracking/
│   └── mlflow_tracker.py   # params, metrics, artifacts, model logging (non-fatal on failure)
├── serving/
│   └── api.py              # FastAPI app: async jobs, guards, prediction, export
├── ui/
│   └── app.py              # Streamlit frontend: upload, progress, leaderboard, SHAP, predict, export
├── automl.py               # AutoML orchestrator class + smoke test (python automl.py)
├── report.py               # self-contained HTML report (plotly + base64-embedded SHAP images)
├── Dockerfile + start.sh   # single-container deployment (HF Spaces compatible)
└── requirements-deploy.txt # slimmed deps for hosting (no torch/TabNet)
```

---

## Configuration reference

All knobs on the `AutoML` constructor (mirrored as form fields on `/train` and sidebar controls in the UI):

| Parameter | Default | Description |
|---|---|---|
| `time_limit` | 300 | Soft wall-clock budget in seconds (informational) |
| `metric` | `"auto"` | Evaluation metric; `auto` picks per task (ROC AUC / F1 Macro / RMSE) |
| `top_n_models` | 3 | How many leaderboard leaders get Optuna tuning + ensembling |
| `enable_feature_engineering` | `True` | Polynomial features on top target-correlated columns |
| `ensemble_strategy` | `"weighted"` | `weighted` · `simple` · `stacking` |
| `n_optuna_trials` | 50 | Optuna trials per tuned model |
| `mlflow_experiment` | `"AutoML"` | MLflow experiment name |
| `output_dir` | `"outputs"` | Where SHAP plots and artifacts are written |

---

## Hosted demo limits (free tier)

- Max **20 MB / 50k rows / 1k columns** per upload
- One training job at a time
- Models live in memory — train and predict in the same session; restarts clear finished jobs
- CPU-only (2 vCPUs) — a default run on a ~1k-row dataset takes ~1–3 minutes

## Tech stack

scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · SHAP · MLflow · FastAPI · Streamlit · Plotly · Docker
