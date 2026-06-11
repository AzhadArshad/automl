"""Ensemble strategies: simple average, weighted average, and stacking."""

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from core.task_detector import TaskType

logger = logging.getLogger(__name__)

EnsembleStrategy = Literal["simple", "weighted", "stacking"]


@dataclass
class EnsembleResult:
    """Fitted ensemble ready for inference."""

    strategy: EnsembleStrategy
    task_type: TaskType
    # Fitted base estimators (in the same order as weights / oof columns)
    base_models: list = field(default_factory=list)
    # CV scores used as weights (weighted strategy only)
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    # Fitted meta-model (stacking only)
    meta_model: object = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return final predictions for the input feature matrix.

        For classification this returns class labels.
        For regression this returns continuous values.

        Args:
            X: Preprocessed feature matrix, shape (n_samples, n_features).

        Returns:
            Predictions array, shape (n_samples,).
        """
        if self.strategy == "stacking":
            return self._stacking_predict(X)

        probas = self._collect_probas_or_preds(X)

        if self.strategy == "simple":
            blended = np.mean(probas, axis=0)
        else:  # weighted
            blended = np.average(probas, axis=0, weights=self.weights)

        if self.task_type == TaskType.REGRESSION:
            return blended
        return (blended >= 0.5).astype(int) if blended.ndim == 1 else np.argmax(blended, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for classification tasks.

        Args:
            X: Preprocessed feature matrix, shape (n_samples, n_features).

        Returns:
            Probability matrix, shape (n_samples, n_classes).

        Raises:
            NotImplementedError: If called on a regression ensemble.
        """
        if self.task_type == TaskType.REGRESSION:
            raise NotImplementedError("predict_proba is not available for regression ensembles.")

        if self.strategy == "stacking":
            return self.meta_model.predict_proba(self._oof_for_predict(X))

        probas = self._collect_probas_or_preds(X)

        if self.strategy == "simple":
            return np.mean(probas, axis=0)
        return np.average(probas, axis=0, weights=self.weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_probas_or_preds(self, X: np.ndarray) -> list[np.ndarray]:
        """Gather predictions/probas from every base model.

        Returns a list where each element is shape (n_samples,) for
        regression/binary or (n_samples, n_classes) for multiclass.
        """
        out = []
        for model in self.base_models:
            if self.task_type == TaskType.REGRESSION:
                out.append(model.predict(X))
            elif self.task_type == TaskType.BINARY:
                # Keep only the positive-class column → shape (n_samples,)
                out.append(model.predict_proba(X)[:, 1])
            else:
                out.append(model.predict_proba(X))
        return out

    def _oof_for_predict(self, X: np.ndarray) -> np.ndarray:
        """Build the meta-feature matrix from base model predictions at inference time."""
        cols = []
        for model in self.base_models:
            if self.task_type == TaskType.BINARY:
                cols.append(model.predict_proba(X)[:, 1].reshape(-1, 1))
            elif self.task_type == TaskType.MULTICLASS:
                cols.append(model.predict_proba(X))
            else:
                cols.append(model.predict(X).reshape(-1, 1))
        return np.hstack(cols)

    def _stacking_predict(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._oof_for_predict(X)
        return self.meta_model.predict(meta_X)


def _build_oof_matrix(
    base_estimators: list,
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    n_splits: int = 5,
) -> tuple[np.ndarray, list]:
    """Produce out-of-fold predictions for the stacking meta-model.

    Each base model is trained K times, predicting only on rows it never
    saw during training. This prevents leakage into the meta-model.

    Args:
        base_estimators: List of unfitted sklearn-compatible estimators.
        X: Full feature matrix.
        y: Target vector.
        task_type: Used to choose CV strategy and prediction type.
        n_splits: Number of CV folds (default 5).

    Returns:
        Tuple of:
            - oof_matrix: shape (n_samples, n_meta_features)
            - fitted_models: list of models each fitted on the full X, y
              (used for inference, not for the OOF matrix).
    """
    is_clf = task_type in (TaskType.BINARY, TaskType.MULTICLASS)
    cv = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if is_clf
        else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    )

    n_samples = X.shape[0]
    oof_cols = []

    for estimator in base_estimators:
        if task_type == TaskType.BINARY:
            col = np.zeros(n_samples)
            for train_idx, val_idx in cv.split(X, y):
                clone = _clone_estimator(estimator)
                clone.fit(X[train_idx], y[train_idx])
                col[val_idx] = clone.predict_proba(X[val_idx])[:, 1]
            oof_cols.append(col.reshape(-1, 1))

        elif task_type == TaskType.MULTICLASS:
            n_classes = len(np.unique(y))
            block = np.zeros((n_samples, n_classes))
            for train_idx, val_idx in cv.split(X, y):
                clone = _clone_estimator(estimator)
                clone.fit(X[train_idx], y[train_idx])
                block[val_idx] = clone.predict_proba(X[val_idx])
            oof_cols.append(block)

        else:  # regression
            col = np.zeros(n_samples)
            for train_idx, val_idx in cv.split(X, y):
                clone = _clone_estimator(estimator)
                clone.fit(X[train_idx], y[train_idx])
                col[val_idx] = clone.predict(X[val_idx])
            oof_cols.append(col.reshape(-1, 1))

    oof_matrix = np.hstack(oof_cols)

    # Fit each base model on the full dataset for inference
    fitted_models = []
    for estimator in base_estimators:
        clone = _clone_estimator(estimator)
        clone.fit(X, y)
        fitted_models.append(clone)

    return oof_matrix, fitted_models


def _clone_estimator(estimator):
    """Return a fresh copy of an estimator with the same parameters."""
    from sklearn.base import clone
    try:
        return clone(estimator)
    except Exception:
        # Fallback for non-sklearn estimators (e.g. TabNet)
        return estimator.__class__(**estimator.get_params())


def build_ensemble(
    base_estimators: list,
    cv_scores: list[float],
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    strategy: EnsembleStrategy = "weighted",
    n_splits: int = 5,
) -> EnsembleResult:
    """Fit and return an ensemble of the given base models.

    Args:
        base_estimators: List of unfitted sklearn-compatible estimators.
        cv_scores: CV score per model (same order as base_estimators).
            Used as weights in the 'weighted' strategy.
        X: Full training feature matrix (preprocessed).
        y: Target vector.
        task_type: Task type.
        strategy: One of 'simple', 'weighted', 'stacking'.
        n_splits: CV folds (used only for stacking OOF generation).

    Returns:
        Fitted EnsembleResult ready to call .predict() / .predict_proba().
    """
    logger.info(
        "Building ensemble: strategy=%s  models=%d  task=%s",
        strategy, len(base_estimators), task_type.value,
    )

    weights = np.array(cv_scores, dtype=float)

    if strategy in ("simple", "weighted"):
        # Fit every base model on the full training set
        fitted = []
        for est in base_estimators:
            clone = _clone_estimator(est)
            clone.fit(X, y)
            fitted.append(clone)

        return EnsembleResult(
            strategy=strategy,
            task_type=task_type,
            base_models=fitted,
            weights=weights,
        )

    # --- Stacking ---
    oof_matrix, fitted_models = _build_oof_matrix(
        base_estimators, X, y, task_type, n_splits=n_splits
    )

    # Meta-model: logistic regression for classification, ridge for regression
    if task_type == TaskType.REGRESSION:
        meta = Ridge()
    else:
        meta = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)

    meta.fit(oof_matrix, y)
    logger.info("Stacking meta-model fitted on OOF matrix shape %s.", oof_matrix.shape)

    return EnsembleResult(
        strategy=strategy,
        task_type=task_type,
        base_models=fitted_models,
        weights=weights,
        meta_model=meta,
    )
