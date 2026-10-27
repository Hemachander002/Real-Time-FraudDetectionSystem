from __future__ import annotations
import mlflow, optuna
from .config import CONFIG
from .data import ingest_data, validate_data, prepare_splits,scale_pos_weight_calculator
from .preprocessing import save_preprocessed_data
from .training import run_experiments
from .evaluation import evaluate_and_publish

def main():
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data, data_hash = ingest_data()
    data = validate_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test,preprocessor = prepare_splits(data)
    scale_pos_weight = scale_pos_weight_calculator(y_train)
    print(f"Train={len(X_train):,}, Validation={len(X_val):,}, Test={len(X_test):,}")
    best = run_experiments(X_train, X_val, y_train, y_val, scale_pos_weight = scale_pos_weight, preprocessor = preprocessor)
    evaluate_and_publish(best, X_test, y_test, data_hash)

if __name__ == "__main__":
    main()
