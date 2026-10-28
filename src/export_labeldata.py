
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
import psycopg
import os

# PostgreSQL configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "fraud_db"),
    "user": os.getenv("DB_USER", "fraud_user"),
    "password": os.getenv("DB_PASSWORD", "local_dev_password"),
}

# S3 configuration
S3_BUCKET = "YOUR S3 BUCKET"
AWS_REGION = "YOUR REGION"
S3_PREFIX = "exports"

# Local export directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"

# Features used by the fraud detection model.
# transaction_id is retained for traceability, not as a model feature.
FEATURES = [
    "transaction_id",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

QUERY = """
SELECT
    t.transaction_id,
    t.type,
    t.amount,
    t.oldbalanceOrg,
    t.newbalanceOrig,
    t.oldbalanceDest,
    t.newbalanceDest,
    o.outcome AS isFraud
FROM transactions AS t
INNER JOIN outcome_table AS o
    ON t.transaction_id = o.tx_id
ORDER BY t.created_at, t.transaction_id;
"""


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Use UTC so filenames are consistent across environments.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    filename = f"labeled_transactions_{timestamp}.csv"
    local_path = EXPORT_DIR / filename
    s3_key = f"{S3_PREFIX}/{filename}"

    print("Connecting to PostgreSQL...")

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(QUERY)

            row_count = 0

            with local_path.open(
                "w", newline="", encoding="utf-8"
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(FEATURES + ["isFraud"])

                while True:
                    rows = cur.fetchmany(5000)

                    if not rows:
                        break

                    writer.writerows(rows)
                    row_count += len(rows)

    if row_count == 0:
        local_path.unlink(missing_ok=True)
        print(
            "No labeled transactions found. "
            "No CSV was uploaded."
        )
        return

    print(f"Exported {row_count:,} labeled transactions.")
    print(f"Local CSV: {local_path}")

    print("Uploading CSV to Amazon S3...")

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
    )

    s3.upload_file(
        str(local_path),
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "text/csv"},
    )

    print("Upload successful.")
    print(f"S3 location: s3://{S3_BUCKET}/{s3_key}")


if __name__ == "__main__":
    main()
