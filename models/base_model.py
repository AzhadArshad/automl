"""Abstract base class that every model wrapper must implement."""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Minimal contract every model in the zoo must satisfy.

    Concrete subclasses wrap a scikit-learn–compatible estimator and must
    implement `fit`, `predict`, and (for classifiers) `predict_proba`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used in the leaderboard."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Train the model on the given data.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Target vector, shape (n_samples,).

        Returns:
            self, to allow method chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels (classification) or values (regression).

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predictions, shape (n_samples,).
        """

    def predict_proba(self, *_) -> np.ndarray:
        """Return class probabilities. Only required for classifiers.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Probability matrix, shape (n_samples, n_classes).

        Raises:
            NotImplementedError: If the model does not support probabilities.
        """
        raise NotImplementedError(f"{self.name} does not support predict_proba.")

    def get_estimator(self):
        """Return the underlying sklearn-compatible estimator object.

        Used by SHAP and ensembling code that need direct access.
        """
        raise NotImplementedError(f"{self.name} does not expose a raw estimator.")
