import time
from pathlib import Path

import pandas as pd
import psycopg
import os

OUTCOMES_PATH = Path(
    "E:\\fraud detection\\data\\outcomes.csv"
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "fraud_db"),
    "user": os.getenv("DB_USER", "fraud_user"),
    "password": os.getenv("DB_PASSWORD", "local_dev_password"),
}

OUTCOME_INTERVAL_MINUTES = 1
POLL_INTERVAL_SECONDS = 2
BATCH_SIZE = 1000


def load_outcomes():
    """Load actual labels indexed by transaction ID."""
    if not OUTCOMES_PATH.exists():
        raise FileNotFoundError(
            f"Outcomes CSV not found: {OUTCOMES_PATH}"
        )

    df = pd.read_csv(OUTCOMES_PATH)
    df.columns = df.columns.str.strip().str.lower()

    required = {"tx_id", "isfraud"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in outcomes.csv: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[["tx_id", "isfraud"]].copy()
    df["tx_id"] = df["tx_id"].astype("string").str.strip()

    if df["tx_id"].isna().any() or df["tx_id"].eq("").any():
        raise ValueError(
            "outcomes.csv contains missing or empty tx_id values."
        )

    if df["tx_id"].duplicated().any():
        raise ValueError(
            "Duplicate tx_id values found in outcomes.csv."
        )

    labels = pd.to_numeric(df["isfraud"], errors="raise")

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(
            "isfraud must contain only 0 or 1."
        )

    df["isfraud"] = labels.astype(int)

    return dict(zip(df["tx_id"], df["isfraud"]))


def deliver_outcomes(conn, labels):
    """
    Insert actual outcomes for transactions that exist in the
    transactions table but do not yet have an outcome.
    """
    total_inserted = 0
    total_missing = 0
    latest_inserted_id = None
    missing_examples = []

    while True:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.transaction_id
                FROM transactions AS t
                LEFT JOIN outcome_table AS o
                    ON o.tx_id = t.transaction_id
                WHERE o.tx_id IS NULL
                ORDER BY t.created_at, t.transaction_id
                LIMIT %s
                """,
                (BATCH_SIZE,),
            )

            rows = cur.fetchall()

        if not rows:
            break

        records = []

        for (tx_id,) in rows:
            outcome = labels.get(tx_id)

            if outcome is None:
                total_missing += 1

                if len(missing_examples) < 5:
                    missing_examples.append(tx_id)

                continue

            records.append((tx_id, outcome))

        if records:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO outcome_table (tx_id, outcome)
                    VALUES (%s, %s)
                    ON CONFLICT (tx_id) DO NOTHING
                    """,
                    records,
                )

                inserted = cur.rowcount

            conn.commit()
            total_inserted += inserted

            if inserted:
                latest_inserted_id = records[-1][0]

        # If a batch contains IDs with no CSV label, they remain
        # unmatched. Stop this delivery to avoid repeatedly reading
        # those same IDs in an infinite loop.
        if total_missing:
            break

    print(
        f"Outcome delivery completed. "
        f"New outcomes recorded: {total_inserted}"
    )

    if latest_inserted_id:
        print(
            f"Latest inserted transaction: {latest_inserted_id}"
        )

    if total_missing:
        print(
            f"Warning: {total_missing} transactions have no matching "
            f"label in outcomes.csv. Examples: {missing_examples}"
        )


def main():
    labels = load_outcomes()

    print(f"Loaded {len(labels):,} actual outcomes.")
    print(
        f"Outcome delivery interval: "
        f"{OUTCOME_INTERVAL_MINUTES} minutes."
    )
    print("Outcome simulator started.")

    with psycopg.connect(**DB_CONFIG) as conn:
        # First delivery occurs after five minutes.
        next_delivery = (
            time.monotonic()
            + OUTCOME_INTERVAL_MINUTES * 60
        )

        while True:
            try:
                remaining = next_delivery - time.monotonic()

                if remaining > 0:
                    time.sleep(
                        min(POLL_INTERVAL_SECONDS, remaining)
                    )
                    continue

                deliver_outcomes(conn, labels)

            except Exception as exc:
                conn.rollback()
                print(f"Outcome simulation error: {exc}")

            # Schedule the next delivery five minutes from now.
            next_delivery = (
                time.monotonic()
                + OUTCOME_INTERVAL_MINUTES * 60
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOutcome simulator stopped.")

