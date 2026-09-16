"""Load raw transaction data for the fraud detection pipeline."""
import argparse

import pandas as pd

from config import load_config


def load_raw_data(raw_path: str) -> pd.DataFrame:
    # TODO: adjust for your actual raw data format (csv/parquet/db query/etc.)
    return pd.read_csv(raw_path)


def main():
    parser = argparse.ArgumentParser(description="Load raw transaction data")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_raw_data(cfg["data"]["raw_path"])
    print(f"Loaded {len(df)} rows from {cfg['data']['raw_path']}")


if __name__ == "__main__":
    main()
