"""Automatic ML task and metric detection from the target column."""

import logging
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"


DEFAULT_METRICS: dict[TaskType, str] = {
    TaskType.BINARY: "roc_auc",
    TaskType.MULTICLASS: "f1_macro",
    TaskType.REGRESSION: "rmse",
}


def detect_task(df: pd.DataFrame, target: str) -> tuple[TaskType, str]:
    """Infer the ML task type and best default metric from the target column.

    Args:
        df: The full DataFrame (after loading).
        target: Name of the target column.

    Returns:
        A tuple of (TaskType, metric_name).

    Raises:
        KeyError: If the target column is not present in the DataFrame.
        ValueError: If the target column is entirely null.
    """
    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found. Available columns: {list(df.columns)}"
        )

    series = df[target].dropna()
    if series.empty:
        raise ValueError(f"Target column '{target}' has no non-null values.")

    n_unique = series.nunique()
    dtype = series.dtype

    if n_unique == 2:
        task = TaskType.BINARY
    elif n_unique <= 20 and (
        pd.api.types.is_object_dtype(dtype) or pd.api.types.is_integer_dtype(dtype)
    ):
        task = TaskType.MULTICLASS
    elif pd.api.types.is_float_dtype(dtype) or n_unique > 20:
        task = TaskType.REGRESSION
    else:
        task = TaskType.MULTICLASS

    metric = DEFAULT_METRICS[task]
    logger.info(
        "Detected task=%s  |  unique target values=%d  |  default metric=%s",
        task.value,
        n_unique,
        metric,
    )
    return task, metric
