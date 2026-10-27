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

from .config import CONFIG, TARGET

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    numerical = X.select_dtypes(include=["number"]).columns.tolist()

    unsupported = sorted(set(X.columns) - set(categorical) - set(numerical))
    if unsupported:
        raise ValueError(
            f"Unsupported feature dtypes: {unsupported}. "
            "Convert them explicitly before training."
        )

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numerical,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])

    # Dense output is required by HistGradientBoostingClassifier. This is
    # memory-intensive for high-cardinality categoricals; this project drops
    # identifier columns, so review cardinality before adding such columns.
    return preprocessor


def save_preprocessed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> Path:
    """Save a CSV of imputed/scaled/encoded features and target/split labels.

    The preprocessing transformer is fitted only on the training split, then
    applied to validation and test data to avoid leakage.
    """
    output_path = Path(
        CONFIG.get(
            "preprocessed_data_output_path",
            "artifacts/preprocessed_data.csv",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()

    frames = []
    for split_name, X_split, y_split in (
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ):
        transformed = preprocessor.transform(X_split)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        split_df = pd.DataFrame(
            transformed,
            columns=feature_names,
            index=X_split.index,
        )
        split_df[normalize_column(TARGET)] = y_split.reindex(X_split.index).astype(int)
        split_df["_split"] = split_name
        frames.append(split_df.reset_index(drop=True))

    preprocessed_df = pd.concat(frames, ignore_index=True)
    preprocessed_df.to_csv(output_path, index=False)

    print(f"Preprocessed CSV saved to: {output_path.resolve()}")
    print(
        f"Preprocessed CSV shape: {preprocessed_df.shape[0]:,} rows x "
        f"{preprocessed_df.shape[1]:,} columns"
    )
    print("Preprocessed split counts:", preprocessed_df["_split"].value_counts().to_dict())
    return output_path
