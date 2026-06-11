"""Cross-validation trainer — evaluates all models and returns a sorted leaderboard."""

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from core.task_detector import TaskType

logger = logging.getLogger(__name__)

# sklearn scoring strings per task
_SCORER: dict[TaskType, str] = {
    TaskType.BINARY: "roc_auc",
    TaskType.MULTICLASS: "f1_macro",
    TaskType.REGRESSION: "neg_root_mean_squared_error",
}

# For regression the scorer is negated — flip sign so higher = better in leaderboard
_NEGATE: set[TaskType] = {TaskType.REGRESSION}


@dataclass
class ModelResult:
    """CV result for a single model."""

    name: str
    cv_score: float
    cv_std: float
    fit_time: float


def _evaluate_one(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    n_splits: int,
) -> ModelResult | None:
    """Run cross-validation for a single model and return its result.

    Returns None if the model raises any error (logged as a warning).

    Args:
        name: Model name for display.
        model: Unfitted sklearn-compatible estimator.
        X: Feature matrix.
        y: Target vector.
        task_type: Determines CV strategy and scorer.
        n_splits: Number of CV folds.

    Returns:
        ModelResult or None on failure.
    """
    scorer = _SCORER[task_type]

    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Force single-threaded inner parallelism to avoid deadlock with joblib's thread pool
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)

    try:
        t0 = time.perf_counter()

        # TabNet's PyTorch training loop hangs inside joblib threads on macOS — skip CV
        if type(model).__name__ in ("TabNetClassifier", "TabNetRegressor"):
            logger.info("%-20s  skipped CV (TabNet — PyTorch incompatible with joblib threads)", name)
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)

        elapsed = time.perf_counter() - t0

        # Flip sign for negated scorers so the leaderboard always shows higher = better
        if task_type in _NEGATE:
            scores = -scores

        result = ModelResult(
            name=name,
            cv_score=round(float(scores.mean()), 5),
            cv_std=round(float(scores.std()), 5),
            fit_time=round(elapsed, 2),
        )
        logger.info(
            "%-20s  score=%.4f  std=%.4f  time=%.1fs",
            name, result.cv_score, result.cv_std, result.fit_time,
        )
        return result

    except Exception as exc:
        logger.warning("Model '%s' failed during CV and was skipped: %s", name, exc)
        return None




def train_all(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    n_splits: int = 5,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Evaluate all models in parallel via cross-validation.

    Args:
        models: Dict of {name: estimator} from model_zoo.get_models().
        X: Preprocessed feature matrix.
        y: Target vector.
        task_type: Used to select CV strategy and scorer.
        n_splits: Number of CV folds (default 5).
        n_jobs: Parallel jobs for joblib (-1 = all cores).

    Returns:
        DataFrame leaderboard sorted by CV score descending, with columns:
        Model, CV Score, Std, Fit Time.
    """
    logger.info(
        "Starting CV training: %d models, %d folds, task=%s",
        len(models), n_splits, task_type.value,
    )

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_evaluate_one)(name, model, X, y, task_type, n_splits)
        for name, model in models.items()
    )

    valid = [r for r in results if r is not None]

    if not valid:
        raise RuntimeError("All models failed during cross-validation. Check your data.")

    leaderboard = pd.DataFrame(
        {
            "Model": [r.name for r in valid],
            "CV Score": [r.cv_score for r in valid],
            "Std": [r.cv_std for r in valid],
            "Fit Time (s)": [r.fit_time for r in valid],
        }
    ).sort_values("CV Score", ascending=False).reset_index(drop=True)

    logger.info("Leaderboard:\n%s", leaderboard.to_string(index=False))
    return leaderboard
