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

from .config import CONFIG, TARGET, DROP_COLUMNS, OUTPUT_PATH, SEED


def evaluate_and_publish(best, X_test, y_test, data_hash):
    model = best["pipeline"]
    # Reuse the validation-selected threshold; never tune on the test set.
    threshold = float(best["validation_threshold"])

    # The test set is evaluated only after model selection and threshold tuning.
    test_scores = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_scores >= threshold).astype(int)

    metrics = {
        "test_precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "test_f1": float(
            f1_score(y_test, test_predictions, zero_division=0)
        ),
        "test_average_precision": float(
            average_precision_score(y_test, test_scores)
        ),
        "validation_recall_at_selected_threshold": float(
            best["validation_recall"]
        ),
        "validation_f1_at_selected_threshold": float(
            best["validation_f1"]
        ),
        "decision_threshold": float(threshold),
    }

    print("\nBest model:", best["model_name"])
    print("Validation precision:", best["validation_precision"])
    print("Validation recall:", best["validation_recall"])
    print("Validation F1:", best["validation_f1"])
    print("Selected threshold:", threshold)
    print("\nFinal test metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))
    print("\nTest classification report:")
    print(classification_report(y_test, test_predictions, zero_division=0))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": model,
        "threshold": threshold,
        "feature_columns": best.get("feature_columns", list(X_test.columns)),
        "model_name": best["model_name"],
        "validation_precision": best["validation_precision"],
        "validation_recall": best["validation_recall"],
        "validation_f1": best["validation_f1"],
        "test_metrics": metrics,
        "data_sha256": data_hash,
        "random_state": SEED,
    }
    joblib.dump(bundle, OUTPUT_PATH)

    # One final MLflow run owns the chosen model and its evaluation artifacts.
    with mlflow.start_run(run_name="best_model_final_evaluation") as run:
        mlflow.log_param("selected_model", best["model_name"])
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_param("random_state", SEED)
        mlflow.log_param("joblib_path", str(OUTPUT_PATH))
        mlflow.log_params({
            f"best_{k}": str(v) for k, v in best["params"].items()
        })
        mlflow.log_metrics(metrics)

        mlflow.log_dict(
            classification_report(
                y_test, test_predictions, output_dict=True,
                zero_division=0,
            ),
            "evaluation/classification_report.json",
        )
        mlflow.log_dict(
            {"confusion_matrix": confusion_matrix(
                y_test, test_predictions
            ).tolist()},
            "evaluation/confusion_matrix.json",
        )
        mlflow.log_artifact(str(OUTPUT_PATH), artifact_path="export")

        # Log the deployable preprocessing + estimator pipeline to MLflow.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=CONFIG["registered_model_name"],
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
                "xgboost.core.Booster".strip(),
                "xgboost.sklearn.XGBClassifier".strip(),
            ],
        )

        mlflow.set_tag("model_family", best["model_name"])
        mlflow.set_tag("data_sha256", data_hash)
        mlflow.set_tag("evaluation_split", "untouched_test")
        mlflow.log_metric("validation_precision", best["validation_precision"])
        mlflow.log_metric("validation_recall", best["validation_recall"])
        mlflow.log_metric("validation_f1", best["validation_f1"])

        print("Final MLflow run ID:", run.info.run_id)

    print("Joblib saved to:", OUTPUT_PATH.resolve())
