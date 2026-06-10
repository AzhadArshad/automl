"""Preprocessing pipeline builder."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, OneHotEncoder

logger = logging.getLogger(__name__)

# Cardinality threshold: columns at or below this use OHE, above use ordinal
_OHE_MAX_CARDINALITY = 15


@dataclass
class PreprocessorResult:
    """Holds the fitted ColumnTransformer and the resulting feature names."""

    pipeline: ColumnTransformer
    feature_names: list[str]


def _extract_datetime_features(df: pd.DataFrame, datetime_cols: list[str]) -> pd.DataFrame:
    """Replace each datetime column with numeric calendar features.

    Extracts: year, month, day, dayofweek, hour, is_weekend.
    The original datetime column is dropped.

    Args:
        df: Input DataFrame (will be modified in-place via copy).
        datetime_cols: List of datetime column names.

    Returns:
        DataFrame with datetime columns replaced by extracted features.
    """
    df = df.copy()
    for col in datetime_cols:
        try:
            dt = pd.to_datetime(df[col])
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_dayofweek"] = dt.dt.dayofweek
            df[f"{col}_hour"] = dt.dt.hour
            df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            df.drop(columns=[col], inplace=True)
            logger.info("Extracted datetime features from column '%s'.", col)
        except Exception as exc:
            logger.warning("Could not extract datetime features from '%s': %s", col, exc)
    return df


def build_preprocessor(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
    cardinality: dict[str, int],
) -> tuple[ColumnTransformer, pd.DataFrame]:
    """Build and return an unfitted ColumnTransformer plus datetime-expanded DataFrame.

    Args:
        df: The raw DataFrame (target column should already be removed by the caller).
        numerical_cols: List of numerical feature column names.
        categorical_cols: List of categorical feature column names.
        datetime_cols: List of datetime column names.
        cardinality: Mapping of column name → number of unique values.

    Returns:
        A tuple of:
            - Unfitted ColumnTransformer (call .fit_transform(df_expanded) externally).
            - DataFrame with datetime columns already replaced by numeric features.
    """
    # Step 1 — expand datetime columns into numeric calendar features
    df_expanded = _extract_datetime_features(df, datetime_cols)

    # Datetime-derived columns are numeric; add them to numerical list
    datetime_derived: list[str] = []
    for col in datetime_cols:
        for suffix in ("year", "month", "day", "dayofweek", "hour", "is_weekend"):
            derived = f"{col}_{suffix}"
            if derived in df_expanded.columns:
                datetime_derived.append(derived)

    all_numerical = numerical_cols + datetime_derived

    # Step 2 — split categoricals by cardinality
    low_card_cats = [c for c in categorical_cols if cardinality.get(c, 0) <= _OHE_MAX_CARDINALITY]
    high_card_cats = [c for c in categorical_cols if cardinality.get(c, 0) > _OHE_MAX_CARDINALITY]

    transformers = []

    if all_numerical:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        transformers.append(("numerical", num_pipeline, all_numerical))

    if low_card_cats:
        low_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("low_card_cat", low_cat_pipeline, low_card_cats))

    if high_card_cats:
        high_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ])
        transformers.append(("high_card_cat", high_cat_pipeline, high_card_cats))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        "Preprocessor built: %d numerical, %d low-card cat (OHE), %d high-card cat (ordinal), "
        "%d datetime cols expanded.",
        len(all_numerical),
        len(low_card_cats),
        len(high_card_cats),
        len(datetime_cols),
    )

    return preprocessor, df_expanded


def fit_preprocessor(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
    cardinality: dict[str, int],
) -> PreprocessorResult:
    """Build, fit, and return the preprocessor plus final feature names.

    Args:
        df: Feature DataFrame (target column removed).
        numerical_cols: Numerical column names.
        categorical_cols: Categorical column names.
        datetime_cols: Datetime column names.
        cardinality: Column → unique count mapping.

    Returns:
        PreprocessorResult with fitted pipeline and feature name list.
    """
    preprocessor, df_expanded = build_preprocessor(
        df, numerical_cols, categorical_cols, datetime_cols, cardinality
    )
    preprocessor.fit(df_expanded)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []

    logger.info("Preprocessor fitted. Output features: %d", len(feature_names))
    return PreprocessorResult(pipeline=preprocessor, feature_names=feature_names)
