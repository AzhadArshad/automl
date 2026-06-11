"""Model zoo — returns a dict of model name → estimator based on dataset size and task."""

import logging

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from core.task_detector import TaskType

logger = logging.getLogger(__name__)

# Row-count thresholds for slow models
_KNN_MAX_ROWS = 50_000
_SVM_MAX_ROWS = 20_000


def get_models(n_rows: int, task_type: TaskType) -> dict:
    """Return a dict of {model_name: estimator} suited to the dataset size and task.

    Args:
        n_rows: Number of training rows (used to gate slow models).
        task_type: One of TaskType.BINARY, MULTICLASS, or REGRESSION.

    Returns:
        Dict mapping model name strings to unfitted sklearn-compatible estimators.
    """
    is_clf = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

    models: dict = {}

    # --- Always-included models ---
    if is_clf:
        models["LogisticRegression"] = LogisticRegression(
            max_iter=1000, solver="lbfgs", n_jobs=-1
        )
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        )
        models["ExtraTrees"] = ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        )
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            verbose=0,
            random_state=42,
        )
        models["MLP"] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=42,
        )
    else:
        models["Ridge"] = Ridge()
        models["RandomForest"] = RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=42
        )
        models["ExtraTrees"] = ExtraTreesRegressor(
            n_estimators=200, n_jobs=-1, random_state=42
        )
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )
        models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            verbose=0,
            random_state=42,
        )
        models["MLP"] = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=42,
        )

    # --- TabNet (classification or regression) ---
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

        if is_clf:
            models["TabNet"] = TabNetClassifier(verbose=0, seed=42)
        else:
            models["TabNet"] = TabNetRegressor(verbose=0, seed=42)
    except ImportError:
        logger.warning("pytorch-tabnet not installed — TabNet skipped.")

    # --- KNN: only for smaller datasets ---
    if n_rows <= _KNN_MAX_ROWS:
        if is_clf:
            models["KNN"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        else:
            models["KNN"] = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        logger.info("KNN included (n_rows=%d ≤ %d).", n_rows, _KNN_MAX_ROWS)
    else:
        logger.info("KNN skipped (n_rows=%d > %d).", n_rows, _KNN_MAX_ROWS)

    # --- SVM: only for small datasets ---
    if n_rows <= _SVM_MAX_ROWS:
        if is_clf:
            # probability=True required for ensembling (uses Platt scaling internally)
            models["SVM"] = SVC(probability=True, kernel="rbf", random_state=42)
        else:
            models["SVM"] = SVR(kernel="rbf")
        logger.info("SVM included (n_rows=%d ≤ %d).", n_rows, _SVM_MAX_ROWS)
    else:
        logger.info("SVM skipped (n_rows=%d > %d).", n_rows, _SVM_MAX_ROWS)

    logger.info("Model zoo assembled: %s", list(models.keys()))
    return models
