from pathlib import Path
import pandas as pd
from datetime import datetime
import shutil

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
BRONZE_DIR = BASE_DIR / "data" / "bronze"
REPORT_DIR = BASE_DIR / "reports" / "quality"

RAW_FILE = RAW_DIR / "vehicles_integrated.csv"
BRONZE_FILE = BRONZE_DIR / "vehicles_bronze.csv"
INGEST_LOG = REPORT_DIR / "ingestion_log.txt"

# Ensure folders exist
BRONZE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

    # Read source data
    df = pd.read_csv(RAW_FILE)

    # Save a direct copy into bronze layer
    shutil.copy2(RAW_FILE, BRONZE_FILE)

    # Log ingestion metadata
    log_lines = [
        "=== INGESTION LOG ===",
        f"Ingestion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source file: {RAW_FILE}",
        f"Bronze file: {BRONZE_FILE}",
        f"Row count: {df.shape[0]}",
        f"Column count: {df.shape[1]}",
        "",
        "Column names:",
        ", ".join(df.columns.tolist())
    ]

    with open(INGEST_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("Bronze ingestion completed successfully.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Bronze file saved to: {BRONZE_FILE}")
    print(f"Ingestion log saved to: {INGEST_LOG}")

if __name__ == "__main__":
    main()
