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
        self.metric: str = ""                # resolved evaluation metric (e.g. roc_auc, rmse)
        self.model: Any = None               # fitted EnsembleResult
        self.preprocessor: Any = None        # fitted ColumnTransformer
        self.explainer_result: Any = None    # ExplainerResult
        self.aml: Any = None                 # fitted AutoML instance (for full transform chain)
        self.classes: list = []              # original target class labels (classification only)
        self.class_labels: list = []         # human-readable label for each class


_jobs: dict[str, JobState] = {}

# Guards for public deployment — keep one CPU-heavy job at a time and
# reject uploads big enough to peg the box for an hour.
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB
_MAX_UPLOAD_ROWS = 50_000
_MAX_UPLOAD_COLS = 1_000
# How many finished jobs to keep in memory — each one holds a fitted ensemble
# and SHAP values, so an unbounded store is a slow memory leak on a long-lived
# deployment.
_MAX_FINISHED_JOBS = 3


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
    """Thin background task — delegates entirely to AutoML.fit().

    Writes the CSV to a temp file, wires a progress callback into the
    AutoML instance so JobState updates in real time, then pulls results
    out of the fitted AutoML object.

    Args:
        job_id: Unique identifier for this job.
        csv_bytes: Raw CSV file content.
        config: Training configuration from the client.
    """
    import tempfile
    import sys
    import os

    # Add project root to path so AutoML can be imported inside the worker
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from automl import AutoML

    state = _jobs[job_id]
    state.status = "running"

    def _on_progress(msg: str, pct: int) -> None:
        state.current_step = msg
        state.progress = pct
        logger.info("[%s] %d%% — %s", job_id, pct, msg)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name

        aml = AutoML(
            time_limit=config.time_limit,
            metric=config.metric,
            top_n_models=config.top_n_models,
            enable_feature_engineering=config.enable_feature_engineering,
            ensemble_strategy=config.ensemble_strategy,
            n_optuna_trials=config.n_optuna_trials,
            progress_callback=_on_progress,
        )
        aml.fit(tmp_path, config.target)

        # Pull results out of the fitted AutoML object
        state.aml              = aml
        state.model            = aml._ensemble
        state.preprocessor     = aml._preprocessor
        state.feature_names    = aml._feature_names
        state.task_type        = aml._task_type.value
        state.metric           = aml.metric
        state.classes          = aml._classes_
        state.class_labels     = aml._class_labels_
        state.leaderboard      = aml.leaderboard().to_dict(orient="records")
        state.explainer_result = aml._explainer_result
        state.status           = "done"

    except Exception as exc:
        logger.exception("Training job %s failed: %s", job_id, exc)
        state.status = "failed"
        state.error = str(exc)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
    csv_bytes = await file.read()
    if len(csv_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    n_rows = csv_bytes.count(b"\n")
    if n_rows > _MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Too many rows (~{n_rows:,}; max {_MAX_UPLOAD_ROWS:,}).",
        )
    n_cols = csv_bytes.split(b"\n", 1)[0].count(b",") + 1
    if n_cols > _MAX_UPLOAD_COLS:
        raise HTTPException(
            status_code=413,
            detail=f"Too many columns (~{n_cols:,}; max {_MAX_UPLOAD_COLS:,}).",
        )

    # Clamp tuning effort server-side — the UI slider caps at 50, but direct
    # API callers could otherwise request unbounded CPU time.
    n_optuna_trials = max(1, min(n_optuna_trials, 50))
    top_n_models = max(1, min(top_n_models, 8))

    # Busy check + job insert must stay in one synchronous block (no awaits
    # between them): the event loop can interleave another /train request at
    # any await point, letting two jobs slip past the one-at-a-time guard.
    if any(s.status in ("queued", "running") for s in _jobs.values()):
        raise HTTPException(
            status_code=429,
            detail="A training job is already running — please try again in a few minutes.",
        )

    # Evict the oldest finished jobs (dicts preserve insertion order) so a
    # long-lived deployment doesn't accumulate fitted models in memory.
    finished = [jid for jid, s in _jobs.items() if s.status in ("done", "failed")]
    while len(finished) >= _MAX_FINISHED_JOBS:
        del _jobs[finished.pop(0)]

    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobState()
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
    return {
        "job_id": job_id,
        "leaderboard": state.leaderboard,
        "task_type": state.task_type,
        "metric": state.metric,
    }


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
    pred_value = _json_safe(prediction)

    classes      = state.classes or []
    class_labels = state.class_labels or [str(c) for c in classes]

    # Decode numeric prediction to human-readable label where possible
    pred_label = pred_value
    decoded = False
    if classes and pred_value in classes:
        idx = classes.index(pred_value)
        pred_label = class_labels[idx]
        decoded = pred_label != str(pred_value)  # only flag as decoded if label actually changed

    result: dict[str, Any] = {"prediction": pred_label}
    if decoded:
        result["raw_prediction"] = pred_value   # only include when label was changed

    if state.task_type != "regression":
        try:
            proba = state.model.predict_proba(X)[0]
            result["probabilities"] = {
                label: round(float(p), 4)
                for label, p in zip(class_labels, proba)
            }
        except Exception:
            pass

    return result


@app.post("/predict/batch")
async def predict_batch(job_id: str = Form(...), file: UploadFile = File(...)):
    """Accept a CSV, run predictions on every row, return CSV with a 'prediction' column."""
    state = _get_job(job_id)
    _require_done(state)

    csv_bytes = await file.read()
    if len(csv_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
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
    """Download the full fitted pipeline as a .pkl file.

    Exports the fitted AutoML instance — preprocessing, feature engineering,
    and ensemble together — so `.predict(raw_df)` works on unprocessed data.
    The training dataset is stripped out before pickling. Unpickling requires
    this project's source code on the Python path.
    """
    import copy

    state = _get_job(job_id)
    _require_done(state)

    aml = copy.copy(state.aml)
    aml._progress_callback = None      # request-local closure — not picklable
    profile = copy.copy(aml._profile)
    profile.df = pd.DataFrame()        # don't ship the user's training data
    aml._profile = profile

    buf = io.BytesIO()
    pickle.dump(aml, buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=automl_pipeline.pkl"},
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
    """Apply the full preprocessing + feature engineering chain to a new DataFrame.

    Uses aml._transform() so that feature engineering (polynomial features,
    target encoding) is applied in the same way as during training — preventing
    the feature count mismatch that occurs when only the ColumnTransformer is used.

    Args:
        df: Raw input DataFrame (same columns as training, minus target).
        state: JobState containing the fitted AutoML instance.

    Returns:
        Transformed numpy array ready for model inference.
    """
    try:
        return state.aml._transform(df)
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
