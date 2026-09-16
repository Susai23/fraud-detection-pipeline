"""Clean and prepare raw transaction data for feature engineering."""
import argparse

import pandas as pd

from config import load_config
from data_loader import load_raw_data


def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    # TODO: drop or impute missing values appropriately for your dataset
    df = df.dropna(subset=[cfg["data"]["target_column"]])

    # TODO: encode categorical columns (one-hot / target encoding / etc.)
    categorical_columns = cfg["features"]["categorical_columns"]
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw transaction data")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_df = load_raw_data(cfg["data"]["raw_path"])
    processed_df = preprocess(raw_df, cfg)
    processed_df.to_csv(cfg["data"]["processed_path"], index=False)
    print(f"Wrote {len(processed_df)} rows to {cfg['data']['processed_path']}")


if __name__ == "__main__":
    main()
