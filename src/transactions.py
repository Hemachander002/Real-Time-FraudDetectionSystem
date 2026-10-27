
import os
import time
from pathlib import Path

import pandas as pd
import psycopg
from prometheus_client import Counter, Gauge, start_http_server

DATA_PATH = Path("E:\\fraud detection\\data\\test.csv")
METRICS_PORT = 8001

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "fraud_db"),
    "user": os.getenv("DB_USER", "fraud_user"),
    "password": os.getenv("DB_PASSWORD", "local_dev_password"),
}

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

BATCH_SIZE = 25
INTERVAL_SECONDS = 1

# Prometheus metrics
TRANSACTIONS_INSERTED = Counter(
    "fraud_transactions_inserted_total",
    "New transactions inserted into PostgreSQL",
)

INSERTION_ERRORS = Counter(
    "fraud_transactions_insertion_errors_total",
    "Transaction insertion errors",
)

SIMULATOR_RUNNING = Gauge(
    "fraud_transaction_simulator_running",
    "Whether the transaction simulator is running",
)


def main():
    # Start metrics server before the main processing loop.
    start_http_server(METRICS_PORT, addr="0.0.0.0")
    print(f"Prometheus metrics available on port {METRICS_PORT}")

    SIMULATOR_RUNNING.set(1)

    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH.resolve()}"
            )

        df = pd.read_csv(DATA_PATH)

        required = ["tx_id"] + FEATURES + ["isFraud"]
        missing = set(required) - set(df.columns)

        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")

        if df["tx_id"].isna().any() or df["tx_id"].duplicated().any():
            raise ValueError(
                "tx_id must be present and unique for every CSV row."
            )

        # Keep the original CSV order.
        df = df[required]

        with psycopg.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT transaction_id FROM transactions")
                existing_ids = {row[0] for row in cur.fetchall()}

            # Skip previously inserted transactions without changing order.
            pending = df[~df["tx_id"].isin(existing_ids)]

            print("Connected to PostgreSQL.")
            print(f"CSV records: {len(df):,}")
            print(f"Already inserted: {len(existing_ids):,}")
            print(f"Remaining records: {len(pending):,}")
            print(
                f"Sending batches of {BATCH_SIZE} transactions "
                f"with a {INTERVAL_SECONDS}-second interval. "
                "Press Ctrl+C to stop."
            )

            for start in range(0, len(pending), BATCH_SIZE):
                started = time.monotonic()
                batch_df = pending.iloc[start:start + BATCH_SIZE]

                records = [
                    (
                        str(row.tx_id),
                        str(row.type),
                        float(row.amount),
                        float(row.oldbalanceOrg),
                        float(row.newbalanceOrig),
                        float(row.oldbalanceDest),
                        float(row.newbalanceDest),
                    )
                    for row in batch_df.itertuples(index=False)
                ]

                try:
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO transactions (
                                transaction_id, type, amount,
                                oldbalanceOrg, newbalanceOrig,
                                oldbalanceDest, newbalanceDest
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (transaction_id) DO NOTHING
                            """,
                            records,
                        )

                        inserted = cur.rowcount

                    conn.commit()

                    # Count actual new rows, not attempted inserts.
                    TRANSACTIONS_INSERTED.inc(inserted)

                    print(
                        f"Batch ending at "
                        f"{batch_df.iloc[-1]['tx_id']} | "
                        f"new rows inserted={inserted}"
                    )

                except Exception:
                    conn.rollback()
                    INSERTION_ERRORS.inc()
                    raise

                elapsed = time.monotonic() - started
                time.sleep(max(0, INTERVAL_SECONDS - elapsed))

            print("CSV simulation completed.")

    finally:
        SIMULATOR_RUNNING.set(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
