"""Optional feature engineering: polynomial features and target encoding."""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

# How many top-correlated numerical columns to use for polynomial features
_TOP_N_POLY = 5


def _top_correlated_numericals(
    df: pd.DataFrame, numerical_cols: list[str], target: pd.Series, top_n: int
) -> list[str]:
    """Return the top-N numerical columns most correlated with the target.

    Args:
        df: Feature DataFrame.
        numerical_cols: Candidate column names.
        target: Target series.
        top_n: Number of columns to select.

    Returns:
        List of column names sorted by absolute correlation, descending.
    """
    correlations: dict[str, float] = {}
    for col in numerical_cols:
        try:
            correlations[col] = abs(df[col].corr(target))
        except Exception:
            correlations[col] = 0.0
    sorted_cols = sorted(correlations, key=lambda c: correlations[c], reverse=True)
    return sorted_cols[:top_n]


def add_polynomial_features(
    df: pd.DataFrame,
    numerical_cols: list[str],
    target: pd.Series,
    degree: int = 2,
    poly_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add degree-2 polynomial interaction terms for the top correlated numerical columns.

    Args:
        df: Feature DataFrame (no target column).
        numerical_cols: All numerical column names in df.
        target: Target series (used only to rank correlations).
        degree: Polynomial degree (default 2).
        poly_cols: If given, expand exactly these columns instead of ranking by
            correlation. Used at inference so the engineered features match
            training exactly (the target is not available then).

    Returns:
        Tuple of (DataFrame with new polynomial columns appended, list of
        columns that were expanded — pass back in as poly_cols at inference).
    """
    if poly_cols is not None:
        top_cols = [c for c in poly_cols if c in df.columns]
    else:
        if not numerical_cols:
            return df, []
        top_cols = _top_correlated_numericals(df, numerical_cols, target, _TOP_N_POLY)

    if not top_cols:
        return df, []

    try:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_array = poly.fit_transform(df[top_cols].fillna(0))
        poly_names = [
            f"poly_{name}" for name in poly.get_feature_names_out(top_cols)
        ]
        poly_df = pd.DataFrame(poly_array, columns=poly_names, index=df.index)

        # Drop columns that already exist in the original df
        new_cols = [c for c in poly_df.columns if c not in df.columns]
        df = pd.concat([df, poly_df[new_cols]], axis=1)
        logger.info("Added %d polynomial features from top-%d numerical cols.", len(new_cols), _TOP_N_POLY)
    except Exception as exc:
        logger.warning("Polynomial feature generation failed: %s", exc)
        return df, []

    return df, top_cols


def add_target_encoding(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Add target-encoded versions of high-cardinality categorical columns.

    Uses K-fold cross-encoding to avoid target leakage: each fold's encoding
    is computed from the other folds.

    Args:
        df: Feature DataFrame (no target column).
        categorical_cols: Categorical column names to encode.
        target: Target series aligned with df.
        n_splits: Number of folds for cross-encoding (default 5).

    Returns:
        DataFrame with new `te_<col>` columns appended.
    """
    if not categorical_cols:
        return df

    from sklearn.model_selection import KFold

    df = df.copy()
    global_mean = target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in categorical_cols:
        encoded = pd.Series(index=df.index, dtype=float)
        for train_idx, val_idx in kf.split(df):
            mapping = target.iloc[train_idx].groupby(df[col].iloc[train_idx]).mean()
            encoded.iloc[val_idx] = df[col].iloc[val_idx].map(mapping).fillna(global_mean)
        df[f"te_{col}"] = encoded
        logger.info("Target-encoded column '%s'.", col)

    return df


def run_feature_engineering(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
    target: pd.Series,
    enable: bool = True,
    poly_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run all optional feature engineering steps.

    Args:
        df: Feature DataFrame (no target column).
        numerical_cols: Numerical column names.
        categorical_cols: Categorical column names.
        target: Target series.
        enable: If False, returns df unchanged (no-op).
        poly_cols: Columns to expand polynomially (inference). None means
            rank by target correlation (training).

    Returns:
        Tuple of (augmented DataFrame, columns used for polynomial features).
    """
    if not enable:
        return df, []

    df, used_poly_cols = add_polynomial_features(df, numerical_cols, target, poly_cols=poly_cols)
    df = add_target_encoding(df, categorical_cols, target)
    return df, used_poly_cols
