"""FastAPI backend — async training jobs, prediction, and explainability endpoints."""

import io
import logging
import pickle
import uuid
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="AutoML API", version="1.0.0")

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job store  {job_id: JobState}
# ---------------------------------------------------------------------------

class JobState:
    """Mutable container for a single training job's lifecycle."""

    def __init__(self) -> None:
        self.status: str = "queued"          # queued | running | done | failed
        self.progress: int = 0               # 0–100
        self.current_step: str = ""
        self.error: str = ""
        self.leaderboard: list[dict] = []
        self.feature_names: list[str] = []
        self.task_type: str = ""
        self.model: Any = None               # fitted EnsembleResult
        self.preprocessor: Any = None        # fitted ColumnTransformer
        self.explainer_result: Any = None    # ExplainerResult


_jobs: dict[str, JobState] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TrainConfig(BaseModel):
    target: str
    time_limit: int = 300
    metric: str = "auto"
    top_n_models: int = 3
    enable_feature_engineering: bool = True
    ensemble_strategy: str = "weighted"
    n_optuna_trials: int = 50


class PredictRequest(BaseModel):
    job_id: str
    features: dict[str, Any]   # {column_name: value}


class BatchPredictResponse(BaseModel):
    job_id: str
    rows_predicted: int


# ---------------------------------------------------------------------------
# Background training task
# ---------------------------------------------------------------------------

def _run_training(job_id: str, csv_bytes: bytes, config: TrainConfig) -> None:
    """Full AutoML pipeline executed as a background task.

    Updates JobState.progress and JobState.current_step at each stage so
    the /status endpoint can report live progress to the UI.

    Args:
        job_id: Unique identifier for this job.
        csv_bytes: Raw CSV file content.
        config: Training configuration from the client.
    """
    state = _jobs[job_id]
    state.status = "running"

    def _step(msg: str, pct: int) -> None:
        state.current_step = msg
        state.progress = pct
        logger.info("[%s] %d%% — %s", job_id, pct, msg)

    try:
        import tempfile, os
        from core.ingestion import load_data
        from core.task_detector import detect_task
        from core.preprocessor import fit_preprocessor
        from core.feature_engineer import run_feature_engineering
        from models.model_zoo import get_models
        from models.trainer import train_all
        from tuning.optuna_tuner import tune_top_models, _build_model
        from ensemble.ensembler import build_ensemble
        from explainability.shap_explainer import explain

        # 1 — Ingest
        _step("Ingesting data", 5)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name
        profile = load_data(tmp_path)
        os.unlink(tmp_path)

        # 2 — Detect task
        _step("Detecting task type", 10)
        task_type, metric = detect_task(profile.df, config.target)
        state.task_type = task_type.value

        # 3 — Prepare features / target
        _step("Preprocessing", 20)
        feature_cols = [
            c for c in profile.df.columns
            if c != config.target and c not in profile.id_cols
        ]
        X_raw = profile.df[feature_cols].copy()
        y = profile.df[config.target].values

        num_cols   = [c for c in profile.numerical_cols   if c in feature_cols]
        cat_cols   = [c for c in profile.categorical_cols if c in feature_cols]
        dt_cols    = [c for c in profile.datetime_cols    if c in feature_cols]

        prep_result = fit_preprocessor(
            X_raw, num_cols, cat_cols, dt_cols, profile.cardinality
        )
        state.preprocessor = prep_result.pipeline
        state.feature_names = prep_result.feature_names

        X = prep_result.pipeline.transform(
            _apply_datetime_expansion(X_raw, dt_cols)
        )

        # 4 — Feature engineering (optional)
        _step("Feature engineering", 30)
        if config.enable_feature_engineering:
            X_df = pd.DataFrame(X, columns=prep_result.feature_names)
            X_df = run_feature_engineering(
                X_df, prep_result.feature_names, [], pd.Series(y),
                enable=True
            )
            X = X_df.values
            state.feature_names = list(X_df.columns)

        # 5 — Train baseline models
        _step("Training models (cross-validation)", 40)
        zoo = get_models(profile.n_rows, task_type)
        leaderboard = train_all(zoo, X, y, task_type)
        state.leaderboard = leaderboard.to_dict(orient="records")

        # 6 — Tune top models
        _step("Tuning hyperparameters (Optuna)", 60)
        tuning_results = tune_top_models(
            leaderboard, zoo, X, y, task_type,
            top_n=config.top_n_models,
            n_trials=config.n_optuna_trials,
        )

        # Build tuned estimators
        top_names = leaderboard["Model"].head(config.top_n_models).tolist()
        top_estimators = []
        top_scores = []
        for name in top_names:
            params = tuning_results[name].best_params if name in tuning_results else {}
            est = _build_model(name, params, task_type)
            top_estimators.append(est)
            score = (
                tuning_results[name].best_score
                if name in tuning_results
                else leaderboard.loc[leaderboard["Model"] == name, "CV Score"].values[0]
            )
            top_scores.append(float(score))

        # 7 — Ensemble
        _step("Building ensemble", 75)
        ensemble = build_ensemble(
            top_estimators, top_scores, X, y, task_type,
            strategy=config.ensemble_strategy,
        )
        state.model = ensemble

        # 8 — Explain
        _step("Computing SHAP explanations", 88)
        exp_result = explain(
            ensemble.base_models[0],   # explain the best single model
            X,
            state.feature_names,
        )
        state.explainer_result = exp_result

        _step("Done", 100)
        state.status = "done"

    except Exception as exc:
        logger.exception("Training job %s failed: %s", job_id, exc)
        state.status = "failed"
        state.error = str(exc)


def _apply_datetime_expansion(df: pd.DataFrame, datetime_cols: list[str]) -> pd.DataFrame:
    """Expand datetime columns to calendar features before passing to the preprocessor."""
    from core.preprocessor import _extract_datetime_features
    return _extract_datetime_features(df, datetime_cols)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/train")
async def train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target: str = Form(...),
    time_limit: int = Form(300),
    metric: str = Form("auto"),
    top_n_models: int = Form(3),
    enable_feature_engineering: bool = Form(True),
    ensemble_strategy: str = Form("weighted"),
    n_optuna_trials: int = Form(50),
):
    """Upload a CSV and start an async training job.

    Returns a job_id immediately. Poll /status/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobState()

    csv_bytes = await file.read()
    config = TrainConfig(
        target=target,
        time_limit=time_limit,
        metric=metric,
        top_n_models=top_n_models,
        enable_feature_engineering=enable_feature_engineering,
        ensemble_strategy=ensemble_strategy,
        n_optuna_trials=n_optuna_trials,
    )

    background_tasks.add_task(_run_training, job_id, csv_bytes, config)
    logger.info("Training job queued: %s", job_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    """Return the current status, progress %, and current step for a job."""
    state = _get_job(job_id)
    return {
        "job_id": job_id,
        "status": state.status,
        "progress": state.progress,
        "current_step": state.current_step,
        "error": state.error,
    }


@app.get("/leaderboard/{job_id}")
def leaderboard(job_id: str):
    """Return the sorted model leaderboard for a completed (or in-progress) job."""
    state = _get_job(job_id)
    return {"job_id": job_id, "leaderboard": state.leaderboard}


@app.post("/predict")
def predict(request: PredictRequest):
    """Run a single-row prediction.

    The feature dict is converted to a one-row DataFrame, preprocessed
    with the same pipeline used during training, then passed to the ensemble.

    Returns class label (or value) and, for classifiers, probabilities.
    """
    state = _get_job(request.job_id)
    _require_done(state)

    row_df = pd.DataFrame([request.features])
    X = _preprocess_input(row_df, state)

    prediction = state.model.predict(X)[0]
    result: dict[str, Any] = {"prediction": _json_safe(prediction)}

    if state.task_type != "regression":
        try:
            proba = state.model.predict_proba(X)[0]
            result["probabilities"] = [round(float(p), 4) for p in proba]
        except Exception:
            pass

    return result


@app.post("/predict/batch")
async def predict_batch(job_id: str = Form(...), file: UploadFile = File(...)):
    """Accept a CSV, run predictions on every row, return CSV with a 'prediction' column."""
    state = _get_job(job_id)
    _require_done(state)

    csv_bytes = await file.read()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    X = _preprocess_input(df, state)

    predictions = state.model.predict(X)
    df["prediction"] = predictions

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get("/explain/{job_id}")
def explain_endpoint(job_id: str):
    """Return SHAP feature importance and plot file paths."""
    state = _get_job(job_id)
    _require_done(state)

    if state.explainer_result is None:
        raise HTTPException(status_code=404, detail="Explainability results not available.")

    return {
        "job_id": job_id,
        "feature_importance": state.explainer_result.feature_importance,
        "plot_paths": state.explainer_result.plot_paths,
    }


@app.get("/export/{job_id}")
def export_model(job_id: str):
    """Download the fitted ensemble model as a .pkl file."""
    state = _get_job(job_id)
    _require_done(state)

    buf = io.BytesIO()
    pickle.dump(state.model, buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={job_id}_model.pkl"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_job(job_id: str) -> JobState:
    """Fetch job state or raise 404."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _jobs[job_id]


def _require_done(state: JobState) -> None:
    """Raise 400 if the job hasn't finished successfully."""
    if state.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet (status={state.status}).",
        )


def _preprocess_input(df: pd.DataFrame, state: JobState) -> np.ndarray:
    """Apply the stored preprocessing pipeline to a new DataFrame.

    Args:
        df: Raw input DataFrame (from /predict or /predict/batch).
        state: JobState containing the fitted preprocessor.

    Returns:
        Transformed numpy array ready for model inference.
    """
    try:
        return state.preprocessor.transform(df)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Preprocessing failed: {exc}",
        )


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
