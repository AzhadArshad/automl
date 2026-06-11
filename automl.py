"""AutoML — top-level orchestrator that wires every phase together."""

import logging
import os
import pickle
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class AutoML:
    """End-to-end AutoML pipeline for tabular data.

    Orchestrates: ingestion → task detection → preprocessing →
    feature engineering → model training → hyperparameter tuning →
    ensembling → explainability → experiment tracking.

    Example::

        aml = AutoML(time_limit=120, top_n_models=3)
        aml.fit("titanic.csv", target="Survived")
        print(aml.leaderboard())
        aml.export("model.pkl")
        aml.report("report.html")
    """

    def __init__(
        self,
        time_limit: int = 300,
        metric: str = "auto",
        top_n_models: int = 3,
        enable_feature_engineering: bool = True,
        ensemble_strategy: str = "weighted",
        n_optuna_trials: int = 50,
        mlflow_experiment: str = "AutoML",
        output_dir: str = "outputs",
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        """Initialise the AutoML system.

        Args:
            time_limit: Soft wall-clock limit in seconds (informational for now;
                enforced limit comes in a future version).
            metric: Evaluation metric. "auto" picks the best default per task.
            top_n_models: Number of top models to tune with Optuna.
            enable_feature_engineering: Whether to add polynomial and target-
                encoded features before training.
            ensemble_strategy: One of "weighted", "simple", "stacking".
            n_optuna_trials: Optuna trials per model during tuning.
            mlflow_experiment: MLflow experiment name for tracking.
            output_dir: Directory for SHAP plots and exported artefacts.
            progress_callback: Optional callable(step_name, pct) invoked at
                each pipeline stage. Used by the FastAPI layer to report live
                progress without polling internals.
        """
        self.time_limit = time_limit
        self.metric = metric
        self.top_n_models = top_n_models
        self.enable_feature_engineering = enable_feature_engineering
        self.ensemble_strategy = ensemble_strategy
        self.n_optuna_trials = n_optuna_trials
        self.mlflow_experiment = mlflow_experiment
        self.output_dir = output_dir
        self._progress_callback = progress_callback

        # Set by .fit()
        self._profile = None
        self._task_type = None
        self._default_metric = None
        self._preprocessor = None
        self._feature_names: list[str] = []
        self._leaderboard: Optional[pd.DataFrame] = None
        self._tuning_results: dict = {}
        self._ensemble = None
        self._explainer_result = None
        self._mlflow_run_id: Optional[str] = None
        self._fit_duration: float = 0.0
        self._target: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, path: str, target: str) -> "AutoML":
        """Run the full AutoML pipeline on a CSV file.

        Steps:
            1. Ingest & profile data
            2. Detect task type and metric
            3. Preprocess (impute, encode, scale, datetime expansion)
            4. Feature engineering (optional)
            5. Train all models via 5-fold CV
            6. Tune top-N models with Optuna
            7. Build ensemble from tuned models
            8. Compute SHAP explanations
            9. Track everything in MLflow

        Args:
            path: Path to the input CSV file.
            target: Name of the target column.

        Returns:
            self, for method chaining.
        """
        t0 = time.perf_counter()
        self._target = target
        os.makedirs(self.output_dir, exist_ok=True)

        # 1 — Ingest
        logger.info("═" * 60)
        self._step("Ingesting data", 5)
        from core.ingestion import load_data
        self._profile = load_data(path)
        logger.info(
            "Dataset: %d rows × %d cols | ID cols dropped: %s",
            self._profile.n_rows, self._profile.n_cols,
            self._profile.id_cols or "none",
        )

        # 2 — Detect task
        self._step("Detecting task type", 10)
        from core.task_detector import detect_task
        self._task_type, self._default_metric = detect_task(self._profile.df, target)
        if self.metric == "auto":
            self.metric = self._default_metric
        logger.info("Task: %s | Metric: %s", self._task_type.value, self.metric)

        # 3 — Preprocess
        self._step("Preprocessing", 20)
        from core.preprocessor import fit_preprocessor, _extract_datetime_features

        feature_cols = [
            c for c in self._profile.df.columns
            if c != target and c not in self._profile.id_cols
        ]
        X_raw = self._profile.df[feature_cols].copy()
        y = self._profile.df[target].values

        num_cols = [c for c in self._profile.numerical_cols   if c in feature_cols]
        cat_cols = [c for c in self._profile.categorical_cols if c in feature_cols]
        dt_cols  = [c for c in self._profile.datetime_cols    if c in feature_cols]

        prep_result = fit_preprocessor(
            X_raw, num_cols, cat_cols, dt_cols, self._profile.cardinality
        )
        self._preprocessor = prep_result.pipeline
        self._feature_names = prep_result.feature_names

        X_expanded = _extract_datetime_features(X_raw, dt_cols)
        X = self._preprocessor.transform(X_expanded)

        # 4 — Feature engineering
        self._step(f"Feature engineering (enabled={self.enable_feature_engineering})", 30)
        if self.enable_feature_engineering:
            from core.feature_engineer import run_feature_engineering
            X_df = pd.DataFrame(X, columns=self._feature_names)
            X_df = run_feature_engineering(
                X_df,
                numerical_cols=self._feature_names,
                categorical_cols=[],
                target=pd.Series(y),
                enable=True,
            )
            X = X_df.values
            self._feature_names = list(X_df.columns)

        # 5 — Train baseline models
        self._step("Training models (cross-validation)", 40)
        from models.model_zoo import get_models
        from models.trainer import train_all
        zoo = get_models(self._profile.n_rows, self._task_type)
        self._leaderboard = train_all(zoo, X, y, self._task_type)
        logger.info("\n%s", self._leaderboard.to_string(index=False))

        # 6 — Tune top models
        self._step(f"Tuning hyperparameters (Optuna, top {self.top_n_models} models)", 60)
        from tuning.optuna_tuner import tune_top_models, _build_model
        self._tuning_results = tune_top_models(
            self._leaderboard, zoo, X, y, self._task_type,
            top_n=self.top_n_models,
            n_trials=self.n_optuna_trials,
        )

        # Reconstruct tuned estimators in leaderboard order
        top_names = self._leaderboard["Model"].head(self.top_n_models).tolist()
        top_estimators = []
        top_scores = []
        for name in top_names:
            params = (
                self._tuning_results[name].best_params
                if name in self._tuning_results else {}
            )
            top_estimators.append(_build_model(name, params, self._task_type))
            score = (
                self._tuning_results[name].best_score
                if name in self._tuning_results
                else float(
                    self._leaderboard.loc[
                        self._leaderboard["Model"] == name, "CV Score"
                    ].values[0]
                )
            )
            top_scores.append(score)

        # 7 — Ensemble
        self._step(f"Building ensemble (strategy={self.ensemble_strategy})", 75)
        from ensemble.ensembler import build_ensemble
        self._ensemble = build_ensemble(
            top_estimators, top_scores, X, y, self._task_type,
            strategy=self.ensemble_strategy,
        )

        # 8 — Explain
        self._step("Computing SHAP explanations", 88)
        from explainability.shap_explainer import explain as shap_explain
        self._explainer_result = shap_explain(
            self._ensemble.base_models[0],
            X,
            self._feature_names,
            output_dir=os.path.join(self.output_dir, "shap_plots"),
        )

        # 9 — Track
        self._step("Logging to MLflow", 95)
        best_name = top_names[0]
        best_params = self._tuning_results.get(best_name, None)
        best_params_dict = best_params.best_params if best_params else {}
        best_score = top_scores[0]

        try:
            from tracking.mlflow_tracker import track_run
            self._mlflow_run_id = track_run(
                experiment_name=self.mlflow_experiment,
                run_name=f"{best_name}_{self._task_type.value}",
                n_rows=self._profile.n_rows,
                n_cols=self._profile.n_cols,
                null_summary=self._profile.null_summary,
                task_type=self._task_type.value,
                target_col=target,
                leaderboard=self._leaderboard,
                best_model_name=best_name,
                best_params=best_params_dict,
                best_cv_score=best_score,
                ensemble_strategy=self.ensemble_strategy,
                model=self._ensemble,
                plot_paths=self._explainer_result.plot_paths,
                feature_importance=self._explainer_result.feature_importance,
            )
            logger.info("MLflow run_id: %s", self._mlflow_run_id)
        except Exception as exc:
            logger.warning("MLflow tracking failed (non-fatal): %s", exc)

        self._fit_duration = time.perf_counter() - t0
        self._step("Done", 100)
        logger.info("═" * 60)
        logger.info("AutoML complete in %.1fs", self._fit_duration)
        return self

    def leaderboard(self) -> pd.DataFrame:
        """Return the model leaderboard sorted by CV score.

        Returns:
            DataFrame with columns: Model, CV Score, Std, Fit Time (s).

        Raises:
            RuntimeError: If called before .fit().
        """
        self._require_fitted()
        return self._leaderboard.copy()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run inference on a new DataFrame.

        Applies the same preprocessing pipeline used during training.

        Args:
            df: Raw feature DataFrame (same columns as training, minus target).

        Returns:
            Predictions array, shape (n_samples,).
        """
        self._require_fitted()
        X = self._transform(df)
        return self._ensemble.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for classification tasks.

        Args:
            df: Raw feature DataFrame.

        Returns:
            Probability matrix, shape (n_samples, n_classes).

        Raises:
            NotImplementedError: If called on a regression model.
        """
        self._require_fitted()
        X = self._transform(df)
        return self._ensemble.predict_proba(X)

    def explain(self) -> dict[str, float]:
        """Return feature importance as {feature_name: mean_abs_shap}.

        Returns:
            Sorted dict of feature → importance (highest first).

        Raises:
            RuntimeError: If called before .fit() or if SHAP failed.
        """
        self._require_fitted()
        if not self._explainer_result or not self._explainer_result.feature_importance:
            raise RuntimeError("SHAP explanations are not available.")
        return self._explainer_result.feature_importance

    def export(self, path: str = "best_model.pkl") -> None:
        """Pickle the fitted ensemble model to disk.

        Args:
            path: Output file path (default "best_model.pkl").
        """
        self._require_fitted()
        with open(path, "wb") as f:
            pickle.dump(self._ensemble, f)
        logger.info("Model exported to: %s", path)

    def report(self, path: str = "report.html") -> None:
        """Generate a self-contained HTML report and save it to disk.

        Args:
            path: Output HTML file path (default "report.html").
        """
        self._require_fitted()
        from report import generate_report
        generate_report(
            profile=self._profile,
            task_type=self._task_type,
            metric=self.metric,
            leaderboard=self._leaderboard,
            tuning_results=self._tuning_results,
            explainer_result=self._explainer_result,
            fit_duration=self._fit_duration,
            output_path=path,
        )
        logger.info("Report saved to: %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        """Raise RuntimeError if .fit() has not been called."""
        if self._ensemble is None:
            raise RuntimeError("Call .fit() before using this method.")

    def _step(self, msg: str, pct: int) -> None:
        """Log a pipeline step and fire the progress callback if one is set."""
        logger.info(msg)
        if self._progress_callback is not None:
            try:
                self._progress_callback(msg, pct)
            except Exception:
                pass

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the stored preprocessing pipeline to a new DataFrame."""
        from core.preprocessor import _extract_datetime_features
        dt_cols = [c for c in self._profile.datetime_cols if c in df.columns]
        df_expanded = _extract_datetime_features(df, dt_cols)
        return self._preprocessor.transform(df_expanded)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from sklearn.datasets import make_classification

    logger.info("Running smoke test with synthetic classification dataset…")

    X_raw, y_raw = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    df = pd.DataFrame(X_raw, columns=[f"feature_{i}" for i in range(X_raw.shape[1])])
    df["target"] = y_raw

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    aml = AutoML(
        time_limit=120,
        top_n_models=2,
        n_optuna_trials=5,          # keep the smoke test fast
        enable_feature_engineering=False,
        ensemble_strategy="weighted",
    )
    aml.fit(tmp_path, target="target")

    print("\n── Leaderboard ──")
    print(aml.leaderboard())

    print("\n── Top 5 SHAP features ──")
    importance = aml.explain()
    for feat, score in list(importance.items())[:5]:
        print(f"  {feat:<30} {score:.4f}")

    aml.export("smoke_test_model.pkl")
    aml.report("smoke_test_report.html")

    os.unlink(tmp_path)
    logger.info("Smoke test complete.")
