"""SHAP-based model explainability: global importance, summary plots, and single-row waterfalls."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shap

logger = logging.getLogger(__name__)

# Tree-based model class names that support TreeExplainer (fast)
_TREE_MODEL_CLASSES = (
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
)

# Max background samples for KernelExplainer (slow fallback)
_KERNEL_BACKGROUND_SAMPLES = 100
# Max rows to compute SHAP values on (keeps it fast for large datasets)
_MAX_EXPLAIN_ROWS = 500


@dataclass
class ExplainerResult:
    """Output of a SHAP explanation run."""

    feature_importance: dict[str, float]   # feature name → mean |SHAP|, sorted desc
    plot_paths: dict[str, str]             # plot type → file path
    shap_values: np.ndarray = field(repr=False, default=None)
    feature_names: list[str] = field(default_factory=list)


def _is_tree_model(model) -> bool:
    """Check if a model supports the fast TreeExplainer."""
    return type(model).__name__ in _TREE_MODEL_CLASSES


def _get_explainer(model, X_background: np.ndarray) -> shap.Explainer:
    """Return the appropriate SHAP explainer for the given model.

    Uses TreeExplainer for tree-based models (fast, exact).
    Falls back to KernelExplainer for all others (slow, model-agnostic).

    Args:
        model: Fitted sklearn-compatible estimator.
        X_background: Background dataset used by KernelExplainer.

    Returns:
        A SHAP explainer instance.
    """
    if _is_tree_model(model):
        logger.info("Using TreeExplainer for %s.", type(model).__name__)
        return shap.TreeExplainer(model)

    logger.info(
        "Using KernelExplainer for %s (background=%d rows).",
        type(model).__name__, len(X_background),
    )
    # KernelExplainer needs a callable — use predict_proba for classifiers, predict otherwise
    if hasattr(model, "predict_proba"):
        fn = model.predict_proba
    else:
        fn = model.predict

    background = shap.sample(X_background, min(_KERNEL_BACKGROUND_SAMPLES, len(X_background)))
    return shap.KernelExplainer(fn, background)


def _compute_shap_values(explainer, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values, handling both old and new SHAP API shapes.

    For binary classification TreeExplainer returns shape (n, features) for
    class 1. For multiclass it returns (n_classes, n, features). We normalise
    to always return (n, features).

    Args:
        explainer: Fitted SHAP explainer.
        X: Feature matrix to explain.

    Returns:
        SHAP values array, shape (n_samples, n_features).
    """
    values = explainer.shap_values(X)

    # Multiclass: list of arrays, one per class → take class 1 or mean abs
    if isinstance(values, list):
        if len(values) == 2:
            # Binary: index 1 is positive class
            return values[1]
        # Multiclass: average absolute values across classes
        return np.mean(np.abs(np.stack(values, axis=0)), axis=0)

    return values


def explain(
    model,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str = "shap_plots",
    row_index: int = 0,
) -> ExplainerResult:
    """Run a full SHAP explanation pass and save plots to disk.

    Generates three artefacts:
    - Bar chart of global mean |SHAP| feature importance
    - Beeswarm summary plot (distribution of SHAP values per feature)
    - Waterfall plot for a single prediction (row_index)

    Args:
        model: Fitted estimator (tree-based or any sklearn model).
        X: Preprocessed feature matrix (numpy array).
        feature_names: Feature names aligned with X columns.
        output_dir: Directory to save plot images.
        row_index: Row in X to use for the single-prediction waterfall plot.

    Returns:
        ExplainerResult with feature_importance dict and plot_paths dict.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless — no display required
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Cap rows for speed
    n = min(_MAX_EXPLAIN_ROWS, X.shape[0])
    X_sample = X[:n]

    try:
        explainer = _get_explainer(model, X_sample)
        shap_values = _compute_shap_values(explainer, X_sample)
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return ExplainerResult(feature_importance={}, plot_paths={})

    # Global importance: mean absolute SHAP per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(
        sorted(
            zip(feature_names, mean_abs.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    plot_paths: dict[str, str] = {}

    # --- Bar chart ---
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        top_n = 20
        names = list(importance.keys())[:top_n]
        vals = list(importance.values())[:top_n]
        ax.barh(names[::-1], vals[::-1], color="#1f77b4")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (SHAP)")
        plt.tight_layout()
        path = os.path.join(output_dir, "shap_bar.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        plot_paths["bar"] = path
        logger.info("Saved SHAP bar chart → %s", path)
    except Exception as exc:
        logger.warning("Failed to save SHAP bar chart: %s", exc)

    # --- Beeswarm summary plot ---
    try:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        path = os.path.join(output_dir, "shap_beeswarm.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        plot_paths["beeswarm"] = path
        logger.info("Saved SHAP beeswarm → %s", path)
    except Exception as exc:
        logger.warning("Failed to save SHAP beeswarm: %s", exc)

    # --- Waterfall plot for a single row ---
    try:
        row_idx = min(row_index, n - 1)
        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
        explanation = shap.Explanation(
            values=shap_values[row_idx],
            base_values=expected_value,
            data=X_sample[row_idx],
            feature_names=feature_names,
        )
        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        path = os.path.join(output_dir, "shap_waterfall.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        plot_paths["waterfall"] = path
        logger.info("Saved SHAP waterfall → %s", path)
    except Exception as exc:
        logger.warning("Failed to save SHAP waterfall: %s", exc)

    return ExplainerResult(
        feature_importance=importance,
        plot_paths=plot_paths,
        shap_values=shap_values,
        feature_names=feature_names,
    )
