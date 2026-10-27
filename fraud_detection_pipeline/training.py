
from __future__ import annotations

import warnings

import mlflow
import optuna

from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .config import CONFIG, SEED
from .models import (
    suggest_model,
    suggest_model_from_params,
)

warnings.filterwarnings("ignore", category=UserWarning)


def run_experiments(
    X_train,
    X_val,
    y_train,
    y_val,
    scale_pos_weight,
    preprocessor,
):
    """
    Tune models with Optuna using a fixed decision threshold.

    Selection criteria:
      - Threshold: 0.8
      - Minimum validation recall: 0.85
      - Minimum validation precision: 0.60
      - Maximize F1 among trials satisfying both constraints.
    """

    THRESHOLD = 0.6

    MIN_RECALL = float(
        CONFIG.get("min_validation_recall", 0.85)
    )
    MIN_PRECISION = float(
        CONFIG.get("min_validation_precision", 0.60)
    )

    model_names = CONFIG.get("models", ["XGBoost"])
    trials_per_model = int(
        CONFIG.get("trials_per_model", 40)
    )

    if not model_names:
        raise ValueError("CONFIG must contain at least one model.")

    if trials_per_model < 1:
        raise ValueError("trials_per_model must be at least 1.")

    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("Training and validation data cannot be empty.")

    if (y_train == 1).sum() == 0:
        raise ValueError("Training data contains no fraud examples.")

    candidates = []

    print("\nFraud detection training")
    print(f"Fixed threshold: {THRESHOLD}")
    print(f"Minimum recall: {MIN_RECALL:.2%}")
    print(f"Minimum precision: {MIN_PRECISION:.2%}")
    print(f"Optuna trials per model: {trials_per_model}")

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Optimizing {model_name}")
        print(f"{'=' * 60}")

        with mlflow.start_run(
            run_name=f"search_{model_name}"
        ) as parent:

            mlflow.log_params({
                "model_family": model_name,
                "trials_requested": trials_per_model,
                "selection_metric": "validation_f1",
                "decision_threshold": THRESHOLD,
                "minimum_validation_recall": MIN_RECALL,
                "minimum_validation_precision": MIN_PRECISION,
                "scale_pos_weight": float(scale_pos_weight),
            })

            def objective(trial):
                estimator = suggest_model(
                    trial,
                    model_name,
                    scale_pos_weight,
                )

                pipeline = Pipeline([
                    ("preprocessing", clone(preprocessor)),
                    ("classifier", estimator),
                ])

                with mlflow.start_run(
                    run_name=f"{model_name}_trial_{trial.number}",
                    nested=True,
                ):
                    pipeline.fit(X_train, y_train)

                    scores = pipeline.predict_proba(X_val)[:, 1]
                    predictions = (
                        scores >= THRESHOLD
                    ).astype(int)

                    recall = float(
                        recall_score(
                            y_val,
                            predictions,
                            zero_division=0,
                        )
                    )
                    precision = float(
                        precision_score(
                            y_val,
                            predictions,
                            zero_division=0,
                        )
                    )
                    f1 = float(
                        f1_score(
                            y_val,
                            predictions,
                            zero_division=0,
                        )
                    )

                    constraints_met = (
                        recall >= MIN_RECALL
                        and precision >= MIN_PRECISION
                    )

                    mlflow.log_metrics({
                        "validation_recall": recall,
                        "validation_precision": precision,
                        "validation_f1": f1,
                        "validation_threshold": THRESHOLD,
                        "constraints_met": int(constraints_met),
                    })

                    trial.set_user_attr(
                        "validation_recall", recall
                    )
                    trial.set_user_attr(
                        "validation_precision", precision
                    )
                    trial.set_user_attr(
                        "validation_f1", f1
                    )
                    trial.set_user_attr(
                        "validation_threshold", THRESHOLD
                    )
                    trial.set_user_attr(
                        "constraints_met", constraints_met
                    )

                    # Only feasible trials can be selected.
                    if not constraints_met:
                        return 0.0

                    return f1

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=SEED
                ),
            )

            study.optimize(
                objective,
                n_trials=trials_per_model,
                show_progress_bar=True,
            )

            feasible_trials = [
                trial
                for trial in study.trials
                if (
                    trial.state == optuna.trial.TrialState.COMPLETE
                    and trial.value is not None
                    and trial.user_attrs.get(
                        "constraints_met", False
                    )
                )
            ]

            if not feasible_trials:
                print(
                    f"WARNING: No {model_name} trial met both "
                    "validation constraints at threshold 0.8."
                )
                mlflow.log_param("feasible_trial_found", False)
                continue

            best_trial = max(
                feasible_trials,
                key=lambda trial: (
                    float(trial.user_attrs["validation_f1"]),
                    float(trial.user_attrs["validation_recall"]),
                    float(trial.user_attrs["validation_precision"]),
                ),
            )

            best_params = dict(best_trial.params)

            mlflow.log_param("feasible_trial_found", True)
            mlflow.log_param(
                "best_trial_number",
                best_trial.number,
            )
            mlflow.log_params({
                f"best_{key}": str(value)
                for key, value in best_params.items()
            })

            # Rebuild the winning estimator with the same class weight.
            best_estimator = suggest_model_from_params(
                model_name,
                best_params,
                scale_pos_weight,
            )

            best_pipeline = Pipeline([
                ("preprocessing", clone(preprocessor)),
                ("classifier", best_estimator),
            ])

            best_pipeline.fit(X_train, y_train)

            # Recheck the rebuilt model at the fixed threshold.
            val_scores = best_pipeline.predict_proba(X_val)[:, 1]
            val_predictions = (
                val_scores >= THRESHOLD
            ).astype(int)

            val_recall = float(
                recall_score(
                    y_val,
                    val_predictions,
                    zero_division=0,
                )
            )
            val_precision = float(
                precision_score(
                    y_val,
                    val_predictions,
                    zero_division=0,
                )
            )
            val_f1 = float(
                f1_score(
                    y_val,
                    val_predictions,
                    zero_division=0,
                )
            )

            mlflow.log_metrics({
                "selected_validation_recall": val_recall,
                "selected_validation_precision": val_precision,
                "selected_validation_f1": val_f1,
                "selected_validation_threshold": THRESHOLD,
            })

            if (
                val_recall < MIN_RECALL
                or val_precision < MIN_PRECISION
            ):
                print(
                    f"WARNING: Rebuilt {model_name} failed the "
                    "validation constraints. Skipping it."
                )
                continue

            candidates.append({
                "model_name": model_name,
                "pipeline": best_pipeline,
                "params": best_params,
                "validation_recall": val_recall,
                "validation_precision": val_precision,
                "validation_f1": val_f1,
                "validation_threshold": THRESHOLD,
                "parent_run_id": parent.info.run_id,
                "feature_columns": list(X_train.columns),
            })

    if not candidates:
        raise RuntimeError(
            "No model met both validation constraints at threshold 0.8. "
            "No valid candidate was selected. Consider more trials or "
            "revising the constraints based on validation results."
        )

    # All candidates satisfy the minimum precision and recall.
    best = max(
        candidates,
        key=lambda item: (
            item["validation_f1"],
            item["validation_recall"],
            item["validation_precision"],
        ),
    )

    print("\nSelected model:")
    print(f"Model: {best['model_name']}")
    print(
        f"Validation recall: "
        f"{best['validation_recall']:.4f}"
    )
    print(
        f"Validation precision: "
        f"{best['validation_precision']:.4f}"
    )
    print(
        f"Validation F1: "
        f"{best['validation_f1']:.4f}"
    )
    print(
        f"Selected threshold: "
        f"{best['validation_threshold']:.4f}"
    )

    return best

