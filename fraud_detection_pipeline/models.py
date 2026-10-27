from __future__ import annotations
import hashlib, json, re, tempfile, warnings
from pathlib import Path
from urllib.parse import urlparse
import joblib, mlflow, mlflow.sklearn, numpy as np, optuna, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore", category=UserWarning)

from .config import SEED

def suggest_model(trial: optuna.Trial, model_name: str,scale_pos_weight: float):
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )

    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
            max_depth=trial.suggest_categorical(
                "max_depth", [None, 8, 12, 20, 30]
            ),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.7]
            ),
            
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )

    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
            max_depth=trial.suggest_categorical(
                "max_depth", [None, 8, 12, 20, 30]
            ),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),

            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )

    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            max_iter=trial.suggest_int("max_iter", 100, 500, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            l2_regularization=trial.suggest_float(
                "l2_regularization", 1e-5, 10, log=True
            ),
            random_state=SEED,
        )

    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.65, 1.0
            ),
            min_child_weight=trial.suggest_float(
                "min_child_weight", 1, 20, log=True
            ),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            scale_pos_weight= scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
        )

    raise ValueError(f"Unknown model: {model_name}")


def suggest_model_from_params(model_name, params,scale_pos_weight):
    """Rebuild an estimator using a completed Optuna trial's parameters."""
    class FrozenTrialLike:
        def suggest_float(self, name, *args, **kwargs):
            return params[name]

        def suggest_int(self, name, *args, **kwargs):
            return params[name]

        def suggest_categorical(self, name, choices):
            return params[name]

    return suggest_model(FrozenTrialLike(), model_name,scale_pos_weight)
