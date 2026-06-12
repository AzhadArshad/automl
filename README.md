---
title: AutoML
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AutoML — README to be written in Phase 4

# Plan to be edited out later

# AutoML System — Claude Code Prompt

## Project Overview

Build a production-grade tabular AutoML system from scratch. This is a portfolio project that should demonstrate deep ML engineering knowledge. The system should accept a CSV file and target column, automatically train and tune multiple models, ensemble the best ones, explain predictions, track experiments, and serve results via a FastAPI backend with a Streamlit UI.

---

## Tech Stack

- **Core ML:** scikit-learn, XGBoost, LightGBM, CatBoost, pytorch-tabnet
- **Hyperparameter tuning:** Optuna
- **Experiment tracking:** MLflow
- **Explainability:** SHAP
- **API:** FastAPI with async background tasks
- **UI:** Streamlit
- **Data profiling:** ydata-profiling

---

## Project Structure to Create

```
automl/
├── core/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   └── task_detector.py
│
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── model_zoo.py
│   └── trainer.py
│
├── tuning/
│   ├── __init__.py
│   └── optuna_tuner.py
│
├── ensemble/
│   ├── __init__.py
│   └── ensembler.py
│
├── explainability/
│   ├── __init__.py
│   └── shap_explainer.py
│
├── tracking/
│   ├── __init__.py
│   └── mlflow_tracker.py
│
├── serving/
│   ├── __init__.py
│   └── api.py
│
├── ui/
│   └── app.py
│
├── automl.py
├── report.py
├── requirements.txt
└── README.md
```

---

## Phase 1 — Core Engine

### `core/ingestion.py`

Responsibilities:

- Load CSV into a pandas DataFrame
- Detect column types: numerical, categorical, datetime, text (high cardinality string)
- Compute basic stats per column: null percentage, cardinality, skewness, dtype
- Flag columns that are likely ID columns (unique ratio > 0.95) and warn user
- Flag potential target leakage columns
- Return a structured `DataProfile` dataclass with all metadata

```python
@dataclass
class DataProfile:
    df: pd.DataFrame
    numerical_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    text_cols: list[str]
    id_cols: list[str]          # likely IDs, should be dropped
    null_summary: dict          # col -> null %
    n_rows: int
    n_cols: int
```

### `core/task_detector.py`

Responsibilities:

- Inspect the target column and infer the ML task type
- Logic:
  - 2 unique values → `binary_classification`
  - 3–20 unique values AND dtype is object/int → `multiclass_classification`
  - continuous float OR integer with many unique values → `regression`
- Return a `TaskType` enum: `BINARY`, `MULTICLASS`, `REGRESSION`
- Also infer the best default metric per task:
  - BINARY → `roc_auc`
  - MULTICLASS → `f1_macro`
  - REGRESSION → `rmse`

### `core/preprocessor.py`

Build a single sklearn Pipeline that handles:

- **Missing values:**
  - Numerical → `SimpleImputer(strategy="median")`
  - Categorical → `SimpleImputer(strategy="constant", fill_value="MISSING")`
- **Encoding:**
  - Categorical columns with cardinality ≤ 15 → `OneHotEncoder(handle_unknown="ignore")`
  - Categorical columns with cardinality > 15 → `OrdinalEncoder(handle_unknown="use_encoded_value")`
- **Scaling:**
  - Numerical → `RobustScaler()` (more robust to outliers than StandardScaler)
- **Datetime:**
  - Extract: year, month, day, dayofweek, hour, is_weekend
  - Drop original datetime column after extraction
- Return a fitted `ColumnTransformer` pipeline
- Also return the feature names after transformation

### `core/feature_engineer.py`

Responsibilities:

- Polynomial features for top numerical columns by correlation with target (degree=2, top 5 cols only)
- Interaction terms between top correlated numerical pairs
- Target encoding for high-cardinality categoricals (using cross-val to avoid leakage)
- Keep this optional and togglable via a flag `enable_feature_engineering=True`

### `models/model_zoo.py`

Implement a `get_models(n_rows, task_type)` function that returns a dict of model name → model instance.

Rules:

- Always include: LogisticRegression (or Ridge for regression), RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost, MLP (sklearn), TabNet
- Include KNN only if `n_rows <= 50_000`
- Include SVM only if `n_rows <= 20_000`
- Use classification or regression variants based on `task_type`
- CatBoost and TabNet must have verbose=0
- SVC must have `probability=True` so it works with ensembling

### `models/trainer.py`

Responsibilities:

- Accept preprocessed data + model zoo dict
- Run 5-fold stratified cross-validation for each model (StratifiedKFold for classification, KFold for regression)
- Use `joblib.Parallel` to train models in parallel (`n_jobs=-1`)
- Track per model: mean CV score, std, fit time
- Return a sorted leaderboard as a pandas DataFrame:

```
Model           | CV Score | Std   | Fit Time
----------------|----------|-------|----------
LightGBM        | 0.934    | 0.008 | 2.3s
XGBoost         | 0.931    | 0.011 | 3.1s
...
```

- Auto-skip models that raise errors (log warning, continue)

---

## Phase 2 — Intelligence Layer

### `tuning/optuna_tuner.py`

Responsibilities:

- Accept a model name + training data
- Define search spaces per model:

```python
search_spaces = {
    "XGBoost": {
        "n_estimators":     ("int", 100, 1000),
        "max_depth":        ("int", 3, 10),
        "learning_rate":    ("float_log", 0.01, 0.3),
        "subsample":        ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "LightGBM": {
        "n_estimators":     ("int", 100, 1000),
        "max_depth":        ("int", 3, 15),
        "learning_rate":    ("float_log", 0.01, 0.3),
        "num_leaves":       ("int", 20, 300),
        "subsample":        ("float", 0.6, 1.0),
    },
    "CatBoost": {
        "iterations":       ("int", 100, 1000),
        "depth":            ("int", 4, 10),
        "learning_rate":    ("float_log", 0.01, 0.3),
    },
    "RandomForest": {
        "n_estimators":     ("int", 100, 500),
        "max_depth":        ("int", 3, 20),
        "min_samples_split":("int", 2, 20),
    },
    # Add for SVM, KNN, MLP, TabNet as well
}
```

- Run N Optuna trials (default 50) with CV scoring
- Use `optuna.integration.OptunaSearchCV` or manual study
- Return best params + best score
- Tune only top-N models from leaderboard (default top 3) to save time

### `ensemble/ensembler.py`

Implement three ensembling strategies:

1. **Simple average** — average predictions of top 3 models
2. **Weighted average** — weight each model by its CV score
3. **Stacking** — train a meta-learner (LogisticRegression or Ridge) on out-of-fold predictions of base models

Default to weighted average. Make strategy configurable.

Return the ensemble as a class that implements `.predict()` and `.predict_proba()`.

---

## Phase 3 — Polish

### `explainability/shap_explainer.py`

Responsibilities:

- Use `shap.TreeExplainer` for tree-based models (fast)
- Use `shap.KernelExplainer` for others (sample 200 rows for speed)
- Generate and save:
  - Global feature importance bar chart
  - Beeswarm summary plot
  - Single prediction waterfall plot
- Return feature importance as a sorted dict `{feature: mean_abs_shap}`

### `tracking/mlflow_tracker.py`

Log per AutoML run:

- Dataset stats: n_rows, n_cols, null %, task type, target column
- Per model: name, params, CV score, std, fit time
- Best model name + params
- Ensemble strategy used
- SHAP plots as artifacts
- Final model as artifact (pickle)
- Use `mlflow.start_run()` context manager

### `serving/api.py`

FastAPI app with these endpoints:

```
POST   /train                → upload CSV + config, returns job_id (async)
GET    /status/{job_id}      → training progress and current step
GET    /leaderboard/{job_id} → sorted model results
POST   /predict              → single row JSON prediction
POST   /predict/batch        → upload CSV, returns CSV with predictions
GET    /explain/{job_id}     → SHAP feature importance + plot URLs
GET    /export/{job_id}      → download best model as .pkl
```

Important implementation details:

- `/train` must be fully async using `BackgroundTasks` — returns `job_id` immediately
- Store job state in an in-memory dict (keyed by job_id) with fields: status, progress %, current_step, leaderboard, model, explainer
- Use `uuid4()` for job IDs
- `/predict` must apply the same preprocessing pipeline before predicting
- Return probabilities for classification tasks
- `/predict/batch` returns a downloadable CSV using `StreamingResponse`
- Add CORS middleware for Streamlit frontend

### `ui/app.py`

Streamlit UI with these sections:

1. **Upload section** — drag and drop CSV, select target column from dropdown, configure options (time limit, metric, top N models)
2. **Training section** — live progress bar polling `/status`, show current step text
3. **Leaderboard section** — styled DataFrame with color-coded scores, bar chart of model comparison
4. **Explainability section** — SHAP bar chart, feature importance table
5. **Predict section** — form with input fields auto-generated from feature names, show prediction + probability gauge
6. **Export section** — download best model button

Use `st.session_state` for job_id persistence across reruns.
Use `time.sleep(3)` + `st.rerun()` loop for live polling.

---

## Phase 4 — Main Orchestrator

### `automl.py`

A clean `AutoML` class that ties everything together:

```python
class AutoML:
    def __init__(
        self,
        time_limit: int = 300,
        metric: str = "auto",
        top_n_models: int = 3,
        enable_feature_engineering: bool = True,
        ensemble_strategy: str = "weighted",
        n_optuna_trials: int = 50,
    ):
        ...

    def fit(self, path: str, target: str) -> "AutoML":
        # Full pipeline: ingest → detect → preprocess → engineer → train → tune → ensemble → explain → track
        ...

    def leaderboard(self) -> pd.DataFrame:
        ...

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        ...

    def explain(self) -> dict:
        ...

    def export(self, path: str = "best_model.pkl") -> None:
        ...

    def report(self, path: str = "report.html") -> None:
        ...
```

### `report.py`

Generate a self-contained HTML report including:

- Dataset summary table
- Task type and metric used
- Leaderboard table with color-coded scores
- Model comparison bar chart (use plotly, embed as HTML)
- SHAP feature importance chart (embed as base64 image)
- Best model params
- Training time summary

---

## requirements.txt

```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
pytorch-tabnet
optuna
mlflow
shap
fastapi
uvicorn
streamlit
ydata-profiling
plotly
joblib
python-multipart
```

---

## Build Order

Build in this strict order — each phase depends on the previous:

1. `core/ingestion.py` + `core/task_detector.py`
2. `core/preprocessor.py`
3. `core/feature_engineer.py`
4. `models/model_zoo.py` + `models/base_model.py`
5. `models/trainer.py`
6. `tuning/optuna_tuner.py`
7. `ensemble/ensembler.py`
8. `automl.py` (wire phases 1-7 together, test end-to-end)
9. `explainability/shap_explainer.py`
10. `tracking/mlflow_tracker.py`
11. `serving/api.py`
12. `ui/app.py`
13. `report.py`
14. `README.md`

---

## Code Quality Requirements

- Every class and function must have docstrings
- Use type hints throughout
- Use dataclasses for structured return types
- All errors must be caught and logged with meaningful messages — never crash silently
- Every module must be independently importable and testable
- Add a `if __name__ == "__main__"` block in `automl.py` with a sample run on a toy dataset (use sklearn's `make_classification`)

---

## Testing the Full Pipeline

After building, verify with this end-to-end test:

```python
from automl import AutoML

aml = AutoML(time_limit=120, top_n_models=3)
aml.fit("titanic.csv", target="Survived")

print(aml.leaderboard())
print(aml.explain())
aml.export("titanic_model.pkl")
aml.report("titanic_report.html")
```

The leaderboard should show at least 6 models with CV scores, the explain output should show top 5 features with SHAP values, and export + report should produce valid files.
