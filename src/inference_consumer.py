
import json
import os
import time
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import psycopg
from kafka import KafkaConsumer
from prometheus_client import Counter, Gauge, Histogram, start_http_server

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "fraud_db"),
    "user": os.getenv("DB_USER", "fraud_user"),
    "password": os.getenv("DB_PASSWORD", "local_dev_password"),
}

MODEL_PATH = Path("E:\\fraud detection\\artifacts\\best_model.joblib")
MODEL_VERSION = "champion_fraud_model_v2"
THRESHOLD = 0.7
METRICS_PORT = 18000

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Prometheus metrics
MESSAGES_PROCESSED = Counter(
    "fraud_inference_messages_processed_total",
    "Kafka messages successfully processed and acknowledged",
)

PREDICTIONS_TOTAL = Counter(
    "fraud_inference_predictions_total",
    "New predictions successfully stored and acknowledged",
    ["prediction"],
)

PROCESSING_ERRORS = Counter(
    "fraud_inference_processing_errors_total",
    "Errors encountered while processing or acknowledging messages",
)

INFERENCE_LATENCY = Histogram(
    "fraud_inference_prediction_duration_seconds",
    "Time spent running model prediction",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

MESSAGE_LATENCY = Histogram(
    "fraud_inference_message_processing_duration_seconds",
    "Time spent processing a Kafka message, including database and offset commits",
)

CONSUMER_RUNNING = Gauge(
    "fraud_inference_consumer_running",
    "Whether the inference consumer is running (1=yes, 0=no)",
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}"
        )

    loaded_artifact = joblib.load(MODEL_PATH)
    # If the artifact is a dictionary bundle, extract the model/estimator key
    # REMOVE THESE COMMENTS and change the "model" to "loaded_artifact" IF U ARE USING A MODEL THAT COMES FROM OUR PIPELINE
    # I HAVE MADE THESE SCRIPTS WITH A MODEL THAT WAS BUILT WITHOUT THE PIPELINE AND WITHOUT THE ONE-HOT-ENCODINGS so im commenting these lines
    if isinstance(loaded_artifact, dict):
        print(f"Loaded artifact is a dictionary containing keys: {list(loaded_artifact.keys())}")
        # Change "model" to whatever key your training script used (e.g., "estimator", "clf", "classifier")
        model = loaded_artifact.get("pipeline") 
    else:
        model = loaded_artifact

    # Safety check
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"The loaded object (type: {type(model)}) does not have a 'predict_proba' method. "
            f"Check your dictionary keys or training script."
        ) 

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Fraud threshold: {THRESHOLD}")

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Fraud threshold: {THRESHOLD}")

    # Expose metrics on http://localhost:8000/metrics
    start_http_server(METRICS_PORT, addr="0.0.0.0")
    print(f"Prometheus metrics available on port {METRICS_PORT}")

    consumer = KafkaConsumer(
        "transactions",
        bootstrap_servers=[os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")],
        group_id="fraud-inference-v1",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )

    CONSUMER_RUNNING.set(1)
    print("Waiting for transaction events...")

    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            for message in consumer:
                message_started = time.perf_counter()
                event = message.value
                tx_id = event.get("transaction_id", "unknown")

                try:
                    features = pd.DataFrame(
                        [{name: event[name] for name in FEATURES}],
                        columns=FEATURES,
                    )

                    inference_started = time.perf_counter()
                    fraud_probability = float(
                        model.predict_proba(features)[0][1]
                    )
                    INFERENCE_LATENCY.observe(
                        time.perf_counter() - inference_started
                    )

                    prediction = int(
                        fraud_probability >= THRESHOLD
                    )

                    created_at = datetime.fromisoformat(
                        event["created_at"]
                    )

                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO inference_table (
                                transaction_id,
                                created_at,
                                type,
                                amount,
                                oldbalanceOrg,
                                newbalanceOrig,
                                oldbalanceDest,
                                newbalanceDest,
                                predicted_isFraud,
                                fraud_probability,
                                model_version
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s
                            )
                            ON CONFLICT (transaction_id)
                            DO NOTHING
                            """,
                            (
                                tx_id,
                                created_at,
                                event["type"],
                                event["amount"],
                                event["oldbalanceOrg"],
                                event["newbalanceOrig"],
                                event["oldbalanceDest"],
                                event["newbalanceDest"],
                                prediction,
                                fraud_probability,
                                MODEL_VERSION,
                            ),
                        )

                        # rowcount is 1 for a new row, 0 for a duplicate.
                        inserted = cur.rowcount == 1

                    # Preserve database-before-Kafka commit ordering.
                    conn.commit()
                    consumer.commit()

                    MESSAGES_PROCESSED.inc()

                    if inserted:
                        label = "fraud" if prediction == 1 else "non_fraud"
                        PREDICTIONS_TOTAL.labels(
                            prediction=label
                        ).inc()

                    MESSAGE_LATENCY.observe(
                        time.perf_counter() - message_started
                    )

                    print(
                        f"{tx_id} | "
                        f"probability={fraud_probability:.4f} | "
                        f"prediction={prediction}"
                    )

                except Exception as exc:
                    PROCESSING_ERRORS.inc()
                    conn.rollback()
                    print(
                        f"Error processing {tx_id}: {exc}. "
                        "The Kafka message was not acknowledged."
                    )
                    raise

    finally:
        CONSUMER_RUNNING.set(0)
        consumer.close()


if __name__ == "__main__":
    main()
