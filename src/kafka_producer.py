import json
import os
import time

import psycopg
from kafka import KafkaProducer
from prometheus_client import Counter, Gauge, start_http_server

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "fraud_db"),
    "user": os.getenv("DB_USER", "fraud_user"),
    "password": os.getenv("DB_PASSWORD", "local_dev_password"),
}

KAFKA_TOPIC = "transactions"
METRICS_PORT = 8002

# Prometheus metrics
MESSAGES_PUBLISHED = Counter(
    "fraud_kafka_messages_published_total",
    "Transactions published to Kafka and marked as published in PostgreSQL",
)

PRODUCER_ERRORS = Counter(
    "fraud_kafka_producer_errors_total",
    "Errors encountered while publishing transactions",
)

PRODUCER_RUNNING = Gauge(
    "fraud_kafka_producer_running",
    "Whether the Kafka producer process is running (1=yes, 0=no)",
)

producer = KafkaProducer(
    bootstrap_servers=[os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")],
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    acks="all",
    retries=5,
)

QUERY = """
SELECT transaction_id, created_at, type, amount,
       oldbalanceOrg, newbalanceOrig,
       oldbalanceDest, newbalanceDest
FROM transactions
WHERE published_to_kafka = FALSE
ORDER BY CAST(SUBSTRING(transaction_id FROM 7) AS BIGINT)
LIMIT 100
"""


def main():
    print("Starting Prometheus metrics server on port 8002...")
    start_http_server(METRICS_PORT)

    PRODUCER_RUNNING.set(1)
    print("Kafka producer started.")

    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            while True:
                try:
                    with conn.cursor() as cur:
                        cur.execute(QUERY)
                        rows = cur.fetchall()

                    if not rows:
                        time.sleep(0.2)
                        continue

                    for row in rows:
                        tx_id, created_at, *features = row

                        event = {
                            "transaction_id": tx_id,
                            "created_at": created_at.isoformat(),
                            "type": features[0],
                            "amount": features[1],
                            "oldbalanceOrg": features[2],
                            "newbalanceOrig": features[3],
                            "oldbalanceDest": features[4],
                            "newbalanceDest": features[5],
                        }

                        try:
                            producer.send(
                                KAFKA_TOPIC,
                                key=tx_id.encode("utf-8"),
                                value=event,
                            ).get(timeout=15)

                            with conn.cursor() as cur:
                                cur.execute(
                                    """
                                    UPDATE transactions
                                    SET published_to_kafka = TRUE
                                    WHERE transaction_id = %s
                                      AND published_to_kafka = FALSE
                                    """,
                                    (tx_id,),
                                )

                            conn.commit()
                            MESSAGES_PUBLISHED.inc()
                            print(f"Published {tx_id}")

                        except Exception as exc:
                            conn.rollback()
                            PRODUCER_ERRORS.inc()
                            print(f"Failed to publish {tx_id}: {exc}")
                            time.sleep(1)

                    time.sleep(0.2)

                except Exception as exc:
                    conn.rollback()
                    PRODUCER_ERRORS.inc()
                    print(f"Producer loop error: {exc}")
                    time.sleep(1)

    finally:
        PRODUCER_RUNNING.set(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Producer stopped.")
    finally:
        producer.close()
