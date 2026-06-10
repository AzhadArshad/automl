"""Data ingestion and profiling module."""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataProfile:
    """Structured profile of the loaded dataset."""

    df: pd.DataFrame
    numerical_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    text_cols: list[str]
    id_cols: list[str]
    null_summary: dict[str, float]
    skewness: dict[str, float]
    cardinality: dict[str, int]
    n_rows: int
    n_cols: int
    warnings: list[str] = field(default_factory=list)


def load_data(path: str) -> DataProfile:
    """Load a CSV file and return a fully profiled DataProfile.

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        DataProfile with column classifications, stats, and warnings.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ValueError: If the file is empty or unparseable.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {path}")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    logger.info("Loaded %d rows x %d cols from %s", len(df), len(df.columns), path)

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    text_cols: list[str] = []
    id_cols: list[str] = []
    null_summary: dict[str, float] = {}
    skewness: dict[str, float] = {}
    cardinality: dict[str, int] = {}
    warnings: list[str] = []

    for col in df.columns:
        null_pct = df[col].isna().mean()
        null_summary[col] = round(float(null_pct), 4)
        n_unique = df[col].nunique()
        cardinality[col] = int(n_unique)
        unique_ratio = n_unique / len(df)

        # Try parsing object columns as datetime before anything else
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
                    datetime_cols.append(col)
                    continue
            except Exception:
                pass

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if unique_ratio > 0.95 and n_unique > 10:
                id_cols.append(col)
                warnings.append(
                    f"Column '{col}' looks like an ID (unique ratio={unique_ratio:.2f}). Consider dropping it."
                )
            else:
                numerical_cols.append(col)
                try:
                    skewness[col] = round(float(df[col].skew()), 4)
                except Exception:
                    skewness[col] = 0.0
            continue

        # Object / string columns
        if unique_ratio > 0.95 and n_unique > 10:
            id_cols.append(col)
            warnings.append(
                f"Column '{col}' looks like an ID (unique ratio={unique_ratio:.2f}). Consider dropping it."
            )
        elif n_unique > 50:
            text_cols.append(col)
            warnings.append(
                f"Column '{col}' has high cardinality ({n_unique} unique values) — treated as free text."
            )
        else:
            categorical_cols.append(col)

    # Leakage heuristic: flag columns whose name hints at being a target
    suspicious = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("target", "label", "output", "predict"))
    ]
    for col in suspicious:
        warnings.append(
            f"Column '{col}' may be a target leakage column — verify before training."
        )

    for w in warnings:
        logger.warning(w)

    return DataProfile(
        df=df,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        text_cols=text_cols,
        id_cols=id_cols,
        null_summary=null_summary,
        skewness=skewness,
        cardinality=cardinality,
        n_rows=len(df),
        n_cols=len(df.columns),
        warnings=warnings,
    )
