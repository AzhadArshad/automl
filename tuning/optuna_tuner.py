"""Optuna-based hyperparameter tuner for the top models from the leaderboard."""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from core.task_detector import TaskType

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# sklearn scoring strings (same convention as trainer.py)
_SCORER: dict[TaskType, str] = {
    TaskType.BINARY: "roc_auc",
    TaskType.MULTICLASS: "f1_macro",
    TaskType.REGRESSION: "neg_root_mean_squared_error",
}
_NEGATE: set[TaskType] = {TaskType.REGRESSION}


@dataclass
class TuningResult:
    """Best params and score found by Optuna for one model."""

    model_name: str
    best_params: dict
    best_score: float
    n_trials: int


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Map a model name to an Optuna parameter suggestion block.

    Each parameter is described as a tuple so the same search space can be
    reused for display or documentation purposes.

    Args:
        trial: Current Optuna trial object.
        model_name: Name of the model (must match a key in the search space).

    Returns:
        Dict of hyperparameter name → suggested value.

    Raises:
        ValueError: If no search space is defined for the model name.
    """
    spaces: dict[str, dict] = {
        "XGBoost": {
            "n_estimators":     ("int",       100, 1000),
            "max_depth":        ("int",       3,   10),
            "learning_rate":    ("float_log", 1e-2, 0.3),
            "subsample":        ("float",     0.6, 1.0),
            "colsample_bytree": ("float",     0.6, 1.0),
            "min_child_weight": ("int",       1,   10),
        },
        "LightGBM": {
            "n_estimators":  ("int",       100, 1000),
            "max_depth":     ("int",       3,   15),
            "learning_rate": ("float_log", 1e-2, 0.3),
            "num_leaves":    ("int",       20,  300),
            "subsample":     ("float",     0.6, 1.0),
            "min_child_samples": ("int",   5,   100),
        },
        "CatBoost": {
            "iterations":    ("int",       100, 1000),
            "depth":         ("int",       4,   10),
            "learning_rate": ("float_log", 1e-2, 0.3),
            "l2_leaf_reg":   ("float_log", 1e-2, 10.0),
        },
        "RandomForest": {
            "n_estimators":     ("int",   100, 500),
            "max_depth":        ("int",   3,   20),
            "min_samples_split":("int",   2,   20),
            "min_samples_leaf": ("int",   1,   10),
            "max_features":     ("float", 0.3, 1.0),
        },
        "ExtraTrees": {
            "n_estimators":     ("int",   100, 500),
            "max_depth":        ("int",   3,   20),
            "min_samples_split":("int",   2,   20),
            "min_samples_leaf": ("int",   1,   10),
        },
        "SVM": {
            "C":     ("float_log", 1e-2, 100.0),
            "gamma": ("float_log", 1e-4, 1.0),
        },
        "KNN": {
            "n_neighbors": ("int",        3,   30),
            "leaf_size":   ("int",        10,  50),
        },
        "MLP": {
            "hidden_layer_sizes": ("categorical", [(64,), (128,), (128, 64), (256, 128)]),
            "alpha":              ("float_log",   1e-5, 1e-1),
            "learning_rate_init": ("float_log",   1e-4, 1e-1),
        },
        "TabNet": {
            "n_d":          ("int",       8,  64),
            "n_a":          ("int",       8,  64),
            "n_steps":      ("int",       3,  10),
            "gamma":        ("float",     1.0, 2.0),
            "momentum":     ("float",     0.01, 0.4),
        },
        "Ridge": {
            "alpha": ("float_log", 1e-3, 100.0),
        },
        "LogisticRegression": {
            "C":       ("float_log", 1e-3, 10.0),
            "max_iter":("int",       200,  2000),
        },
    }

    if model_name not in spaces:
        raise ValueError(f"No search space defined for model '{model_name}'.")

    params = {}
    for param_name, spec in spaces[model_name].items():
        kind = spec[0]
        if kind == "int":
            params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
        elif kind == "float":
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2])
        elif kind == "float_log":
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2], log=True)
        elif kind == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, spec[1])

    return params


def _build_model(model_name: str, params: dict, task_type: TaskType):
    """Instantiate a model with the given hyperparameters.

    Args:
        model_name: Model key matching the zoo and search space.
        params: Hyperparameter dict (from Optuna trial or best_params).
        task_type: Determines whether to use classifier or regressor variant.

    Returns:
        Unfitted sklearn-compatible estimator.
    """
    is_clf = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

    if model_name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if is_clf else XGBRegressor
        return cls(verbosity=0, use_label_encoder=False, random_state=42, n_jobs=-1, **params)

    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if is_clf else LGBMRegressor
        return cls(verbose=-1, random_state=42, n_jobs=-1, **params)

    if model_name == "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if is_clf else CatBoostRegressor
        return cls(verbose=0, allow_writing_files=False, random_state=42, **params)

    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        return cls(n_jobs=-1, random_state=42, **params)

    if model_name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        cls = ExtraTreesClassifier if is_clf else ExtraTreesRegressor
        return cls(n_jobs=-1, random_state=42, **params)

    if model_name == "SVM":
        from sklearn.svm import SVC, SVR
        return SVC(probability=True, kernel="rbf", random_state=42, **params) if is_clf else SVR(kernel="rbf", **params)

    if model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        cls = KNeighborsClassifier if is_clf else KNeighborsRegressor
        return cls(n_jobs=-1, **params)

    if model_name == "MLP":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        cls = MLPClassifier if is_clf else MLPRegressor
        return cls(max_iter=300, random_state=42, **params)

    if model_name == "TabNet":
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        cls = TabNetClassifier if is_clf else TabNetRegressor
        return cls(verbose=0, seed=42, **params)

    if model_name == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)

    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(solver="lbfgs", n_jobs=-1, **params)

    raise ValueError(f"Unknown model name: '{model_name}'")


def tune_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    n_trials: int = 50,
    n_splits: int = 5,
    trial_callback=None,
) -> TuningResult:
    """Run Optuna hyperparameter search for one model.

    Args:
        model_name: Name of the model to tune (must exist in search space).
        X: Preprocessed feature matrix.
        y: Target vector.
        task_type: Task type for scorer and CV strategy selection.
        n_trials: Number of Optuna trials (default 50).
        n_splits: CV folds per trial (default 5).
        trial_callback: Optional callable(trial_number, n_trials) called after
            each completed trial. Used for live progress reporting.

    Returns:
        TuningResult with best_params and best_score.
    """
    scorer = _SCORER[task_type]
    cv = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if task_type in (TaskType.BINARY, TaskType.MULTICLASS)
        else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        model = _build_model(model_name, params, task_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=1)
        # neg_root_mean_squared_error is already "higher = better" — return it
        # as-is so direction="maximize" minimises RMSE.
        return float(scores.mean())

    def _on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial_callback is not None:
            trial_callback(trial.number + 1, n_trials)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_on_trial_end])

    best_score = study.best_value
    if task_type in _NEGATE:
        best_score = -best_score

    logger.info(
        "Tuned %-20s  best_score=%.4f  trials=%d",
        model_name, best_score, n_trials,
    )

    return TuningResult(
        model_name=model_name,
        best_params=study.best_params,
        best_score=round(best_score, 5),
        n_trials=n_trials,
    )


def tune_top_models(
    leaderboard: "pd.DataFrame",
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    top_n: int = 3,
    n_trials: int = 50,
    progress_callback=None,
    start_pct: int = 60,
    end_pct: int = 75,
) -> dict[str, TuningResult]:
    """Tune the top-N models from the leaderboard.

    Models without a defined search space are skipped with a warning.

    Args:
        leaderboard: DataFrame returned by trainer.train_all().
        models: Original model dict from model_zoo (used for name validation).
        X: Preprocessed feature matrix.
        y: Target vector.
        task_type: Task type.
        top_n: How many top models to tune (default 3).
        n_trials: Optuna trials per model (default 50).
        progress_callback: Optional callable(msg, pct) for live UI progress.
            Fires after every Optuna trial so the progress bar moves smoothly.
        start_pct: Progress % at the start of tuning (default 60).
        end_pct: Progress % at the end of tuning (default 75).

    Returns:
        Dict of {model_name: TuningResult} for successfully tuned models.
    """
    top_names = leaderboard["Model"].head(top_n).tolist()
    logger.info("Tuning top-%d models: %s", top_n, top_names)

    total_trials = top_n * n_trials
    completed_trials = 0
    pct_range = end_pct - start_pct

    results: dict[str, TuningResult] = {}
    for model_idx, name in enumerate(top_names):
        if progress_callback:
            progress_callback(
                f"Tuning {name} (model {model_idx + 1}/{top_n})",
                start_pct + int((completed_trials / max(total_trials, 1)) * pct_range),
            )

        def _trial_cb(trial_num: int, _n_trials: int, _name=name, _model_idx=model_idx) -> None:
            nonlocal completed_trials
            completed_trials += 1
            if progress_callback:
                pct = start_pct + int((completed_trials / max(total_trials, 1)) * pct_range)
                progress_callback(
                    f"Tuning {_name} — trial {trial_num}/{_n_trials} "
                    f"(model {_model_idx + 1}/{top_n})",
                    pct,
                )

        try:
            result = tune_model(
                name, X, y, task_type,
                n_trials=n_trials,
                trial_callback=_trial_cb,
            )
            results[name] = result
        except ValueError as exc:
            logger.warning("Skipping tuning for '%s': %s", name, exc)
            completed_trials += n_trials  # advance counter so % stays correct
        except Exception as exc:
            logger.warning("Tuning failed for '%s': %s", name, exc)
            completed_trials += n_trials

    return results
