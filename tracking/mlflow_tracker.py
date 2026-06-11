"""MLflow experiment tracking — logs dataset stats, model results, and artefacts."""

import logging
import os
import pickle
from contextlib import contextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

logger = logging.getLogger(__name__)


@contextmanager
def start_run(experiment_name: str = "AutoML", run_name: str | None = None):
    """Context manager that creates (or reuses) an MLflow experiment and starts a run.

    Usage::

        with start_run("MyExperiment", run_name="titanic_run") as run:
            log_dataset(...)
            log_leaderboard(...)

    Args:
        experiment_name: MLflow experiment name (created if it doesn't exist).
        run_name: Optional human-readable name for this specific run.

    Yields:
        The active mlflow.ActiveRun object.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("MLflow run started: %s", run.info.run_id)
        yield run
    logger.info("MLflow run ended: %s", run.info.run_id)


def log_dataset(
    n_rows: int,
    n_cols: int,
    null_summary: dict[str, float],
    task_type: str,
    target_col: str,
) -> None:
    """Log dataset-level statistics as MLflow params.

    Args:
        n_rows: Number of rows in the training set.
        n_cols: Number of feature columns.
        null_summary: Dict of col → null percentage.
        task_type: String representation of the detected task type.
        target_col: Name of the target column.
    """
    avg_null = round(sum(null_summary.values()) / max(len(null_summary), 1), 4)
    cols_with_nulls = sum(1 for v in null_summary.values() if v > 0)

    mlflow.log_params({
        "dataset.n_rows": n_rows,
        "dataset.n_cols": n_cols,
        "dataset.avg_null_pct": avg_null,
        "dataset.cols_with_nulls": cols_with_nulls,
        "dataset.task_type": task_type,
        "dataset.target_col": target_col,
    })
    logger.info("Logged dataset stats to MLflow.")


def log_leaderboard(leaderboard: pd.DataFrame) -> None:
    """Log each model's CV score and fit time as MLflow metrics.

    Args:
        leaderboard: DataFrame with columns Model, CV Score, Std, Fit Time (s).
    """
    for _, row in leaderboard.iterrows():
        name = row["Model"].replace(" ", "_").lower()
        mlflow.log_metrics({
            f"cv_score.{name}": row["CV Score"],
            f"cv_std.{name}": row["Std"],
            f"fit_time.{name}": row["Fit Time (s)"],
        })
    logger.info("Logged leaderboard (%d models) to MLflow.", len(leaderboard))


def log_best_model(
    model_name: str,
    params: dict[str, Any],
    cv_score: float,
    ensemble_strategy: str,
) -> None:
    """Log the best model name, its tuned params, and ensemble strategy.

    Args:
        model_name: Name of the best-performing model.
        params: Tuned hyperparameters (from Optuna or defaults).
        cv_score: Best CV score after tuning.
        ensemble_strategy: One of 'simple', 'weighted', 'stacking'.
    """
    mlflow.log_param("best_model.name", model_name)
    mlflow.log_param("best_model.ensemble_strategy", ensemble_strategy)
    mlflow.log_metric("best_model.cv_score", cv_score)

    # Log each hyperparameter with a namespaced key
    for k, v in params.items():
        try:
            mlflow.log_param(f"best_model.param.{k}", v)
        except Exception as exc:
            logger.warning("Could not log param '%s': %s", k, exc)

    logger.info("Logged best model '%s' (score=%.4f) to MLflow.", model_name, cv_score)


def log_shap_plots(plot_paths: dict[str, str]) -> None:
    """Log SHAP plot images as MLflow artefacts.

    Args:
        plot_paths: Dict of {plot_type: file_path} from ExplainerResult.plot_paths.
    """
    for plot_type, path in plot_paths.items():
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path="shap_plots")
            logger.info("Logged SHAP artefact '%s' → %s", plot_type, path)
        else:
            logger.warning("SHAP plot file not found, skipping: %s", path)


def log_model_artifact(model: Any, filename: str = "best_model.pkl") -> None:
    """Pickle the final model and log it as an MLflow artefact.

    Also attempts to log via mlflow.sklearn for richer model metadata.

    Args:
        model: Fitted model or EnsembleResult to persist.
        filename: Output pickle filename (logged inside the 'model' artefact dir).
    """
    try:
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Logged model via mlflow.sklearn.")
    except Exception:
        # Fallback: plain pickle for non-sklearn objects (e.g. EnsembleResult)
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(filename, artifact_path="model")
        logger.info("Logged model as pickle artefact: %s", filename)


def log_feature_importance(feature_importance: dict[str, float]) -> None:
    """Log top-20 SHAP feature importances as MLflow metrics.

    Args:
        feature_importance: Dict of {feature_name: mean_abs_shap}, sorted desc.
    """
    for rank, (feature, score) in enumerate(list(feature_importance.items())[:20]):
        safe_name = feature.replace(" ", "_").replace("(", "").replace(")", "")
        try:
            mlflow.log_metric(f"shap.{safe_name}", round(score, 6))
        except Exception as exc:
            logger.warning("Could not log SHAP metric for '%s': %s", feature, exc)

    logger.info("Logged top SHAP importances to MLflow.")


def track_run(
    experiment_name: str,
    run_name: str,
    n_rows: int,
    n_cols: int,
    null_summary: dict[str, float],
    task_type: str,
    target_col: str,
    leaderboard: pd.DataFrame,
    best_model_name: str,
    best_params: dict[str, Any],
    best_cv_score: float,
    ensemble_strategy: str,
    model: Any,
    plot_paths: dict[str, str],
    feature_importance: dict[str, float],
) -> str:
    """Convenience wrapper — runs all logging steps inside a single MLflow run.

    Args:
        experiment_name: MLflow experiment name.
        run_name: Name for this specific run.
        n_rows: Dataset row count.
        n_cols: Dataset column count.
        null_summary: Col → null pct mapping.
        task_type: Detected task type string.
        target_col: Target column name.
        leaderboard: Model leaderboard DataFrame.
        best_model_name: Name of the best model.
        best_params: Tuned hyperparameters for the best model.
        best_cv_score: Best CV score after tuning.
        ensemble_strategy: Ensemble strategy used.
        model: Fitted final model or EnsembleResult.
        plot_paths: SHAP plot file paths.
        feature_importance: SHAP feature importance dict.

    Returns:
        The MLflow run_id string.
    """
    with start_run(experiment_name, run_name=run_name) as run:
        log_dataset(n_rows, n_cols, null_summary, task_type, target_col)
        log_leaderboard(leaderboard)
        log_best_model(best_model_name, best_params, best_cv_score, ensemble_strategy)
        log_shap_plots(plot_paths)
        log_model_artifact(model)
        log_feature_importance(feature_importance)

    return run.info.run_id
