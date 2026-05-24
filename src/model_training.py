"""
模型训练模块 (创新点5)
- 多模型支持: 逻辑回归、随机森林、SVM、朴素贝叶斯
- GridSearchCV / RandomizedSearchCV 超参数调优
- 模型保存与加载
- K 折交叉验证
"""
import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib

import config

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "lr": {
        "name": "Logistic Regression",
        "cls": LogisticRegression,
        "default_params": {"max_iter": 5000, "solver": "saga", "random_state": config.RANDOM_STATE},
    },
    "rf": {
        "name": "Random Forest",
        "cls": RandomForestClassifier,
        "default_params": {"random_state": config.RANDOM_STATE},
    },
    "svm": {
        "name": "SVM",
        "cls": SVC,
        "default_params": {"random_state": config.RANDOM_STATE, "probability": True},
    },
    "nb": {
        "name": "Naive Bayes",
        "cls": GaussianNB,
        "default_params": {},
    },
}


def _get_model_class(model_key):
    """Get the model class."""
    return MODEL_REGISTRY[model_key]["cls"]


def _build_model(model_key, params=None):
    """Build a model instance with default + custom params."""
    info = MODEL_REGISTRY[model_key]
    merged = {**info["default_params"]}
    if params:
        merged.update(params)
    cls = _get_model_class(model_key)
    return cls(**merged)


def split_data(df, target_col=None, test_size=None):
    """Split data into train/test sets."""
    target_col = target_col or config.TARGET_COL
    test_size = test_size or config.TEST_SIZE

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def tune_model(model_key, X_train, y_train, use_grid_search=None):
    """Hyperparameter tuning via GridSearchCV.

    Returns
    -------
    model : best estimator
    cv_results : dict
    """
    use_grid_search = use_grid_search if use_grid_search is not None else config.USE_GRID_SEARCH

    if not use_grid_search:
        model = _build_model(model_key)
        model.fit(X_train, y_train)
        logger.info("Trained %s (no tuning)", MODEL_REGISTRY[model_key]["name"])
        return model, None

    param_grid = config.PARAM_GRIDS.get(model_key)
    if param_grid is None:
        logger.warning("No param grid for %s; training with defaults.", model_key)
        model = _build_model(model_key)
        model.fit(X_train, y_train)
        return model, None

    logger.info("Tuning %s with GridSearchCV (cv=%d)...", MODEL_REGISTRY[model_key]["name"], config.GRID_SEARCH_CV)

    base_model = _build_model(model_key)
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=config.GRID_SEARCH_CV,
        scoring=config.GRID_SEARCH_SCORING,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    logger.info("  Best params: %s", grid.best_params_)
    logger.info("  Best CV score (%.3s): %.4f", config.GRID_SEARCH_SCORING, grid.best_score_)

    return grid.best_estimator_, grid.cv_results_


def train_models(df, model_keys=None, target_col=None, tune=True):
    """Train specified models with optional hyperparameter tuning.

    Returns
    -------
    dict[str, object]
        Trained models keyed by model_key.
    tuple
        X_train, X_test, y_train, y_test
    """
    model_keys = model_keys or config.MODELS
    target_col = target_col or config.TARGET_COL

    logger.info("=" * 50)
    logger.info("Model Training Pipeline Start")
    logger.info("Models to train: %s", [MODEL_REGISTRY.get(k, {}).get("name", k) for k in model_keys])
    logger.info("=" * 50)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_col)

    trained_models = {}
    cv_results_all = {}

    for key in model_keys:
        if key not in MODEL_REGISTRY:
            logger.warning("Unknown model key '%s'; skipping.", key)
            continue
        try:
            model, cv_result = tune_model(key, X_train, y_train, use_grid_search=tune)
            trained_models[key] = model
            cv_results_all[key] = cv_result
            # Save model
            save_model(model, key)
            # Cross-validation summary
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=config.CV_FOLDS, scoring="roc_auc"
            )
            logger.info(
                "  CV %d-fold AUC: %.4f (±%.4f)",
                config.CV_FOLDS, cv_scores.mean(), cv_scores.std(),
            )
        except Exception as e:
            logger.error("Failed to train %s: %s", key, e, exc_info=True)

    logger.info("Model Training Pipeline Complete")
    return trained_models, (X_train, X_test, y_train, y_test)


def save_model(model, model_key):
    """Save a trained model to disk."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, f"{model_key}_model.pkl")
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)


def load_model(model_key):
    """Load a trained model from disk."""
    path = os.path.join(config.MODELS_DIR, f"{model_key}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
