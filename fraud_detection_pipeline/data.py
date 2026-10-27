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

from .config import CONFIG, TARGET, DROP_COLUMNS, REQUIRED_COLUMNS, SEED


def download_dataset(url: str) -> Path:
    """Download a CSV from Google Drive or a direct HTTP(S) URL."""
    if not url or "PASTE_YOUR" in url:
        raise ValueError("Set data_url in training_config.json.")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        path = Path(url)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    # Google Drive sharing links: use gdown to handle confirmation pages.
    if "drive.google.com" in parsed.netloc:
        import gdown

        output = Path(tempfile.gettempdir()) / "training_dataset.csv"
        result = gdown.download(url, output = str(output), quiet=False)
        if not result or not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(
                "Google Drive download failed. Check sharing permissions; "
                "the file must be accessible to your account or via the link."
            )
        return output

    # Direct CSV URL.
    import requests

    output = Path(tempfile.gettempdir()) / "training_dataset.csv"
    response = requests.get(url, timeout=(15, 180))
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise ValueError(
            "The URL returned an HTML page, not a CSV. "
            "Use a direct file URL or a Google Drive sharing link."
        )

    output.write_bytes(response.content)
    if output.stat().st_size == 0:
        raise ValueError("Downloaded dataset is empty.")
    return output


def ingest_data() -> tuple[pd.DataFrame, str]:
    path = download_dataset(CONFIG["data_url"])

    # Hash the raw file for reproducibility.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError("Dataset contains zero rows.")

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    print(f"Raw file SHA-256: {digest}")
    columns_to_check = ["isFlaggedFraud", "step", "nameOrig", "nameDest"]
    if set(columns_to_check).issubset(df.columns):
        df.drop(columns=["isFlaggedFraud","step","nameOrig", "nameDest"], inplace=True)
    return df, digest


def validate_data(df: pd.DataFrame) -> pd.DataFrame:

    target = TARGET
    required = REQUIRED_COLUMNS

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Normalize configured drop-column names too.
    # The target itself must never be dropped.
    if target not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    if df[target].isna().any():
        raise ValueError(
            f"Target '{TARGET}' contains missing labels. "
            "Investigate or explicitly remove these rows."
        )

    # This pipeline expects binary labels represented as 0 and 1.
    values = set(df[target].unique().tolist())
    if not values.issubset({0, 1, False, True}):
        raise ValueError(
            f"Expected binary target labels 0/1; found {list(values)[:20]}. "
            "Map labels explicitly before training."
        )

    df[target] = df[target].astype(int)

    if df[target].nunique() != 2:
        raise ValueError("Training requires both positive and negative labels.")

    # Duplicate records are reported, not silently removed.
    duplicate_count = int(df.duplicated().sum())
    print(f"Exact duplicate rows: {duplicate_count:,}")

    if duplicate_count:
        if CONFIG.get("drop_duplicate_rows", False):
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"Rows after duplicate removal: {len(df):,}")

    print("Validated columns:", df.columns.tolist())
    print("Target distribution:")
    print(df[target].value_counts(dropna=False).to_dict())

    return df


def prepare_splits(df: pd.DataFrame):

    global TARGET
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")



    # 80% training, 20% temporary data
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET],
    )

    # Split remaining 20% equally: 10% validation, 10% testing
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[TARGET],
    )
    TARGET = "isFraud"

    EXCLUDED_COLUMNS = [
    "isFraud",
    "step",
    "nameOrig",
    "nameDest",
    "isFlaggedFraud",
    "tx_id"
    ]

    feature_cols = [
        col for col in train_df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].astype(int)

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET].astype(int)

    categorical = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = [
        col for col in feature_cols
        if col not in categorical
    ]

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


    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


def scale_pos_weight_calculator(y_train):
    positives = int(y_train.sum())
    negatives = len(y_train) - positives

    if positives == 0:
        raise ValueError("Training set contains no fraud examples.")

    scale_pos_weight = negatives / positives
    return scale_pos_weight
