"""Score new, unlabeled transactions with a trained fraud model."""
import argparse

import joblib
import pandas as pd

from config import load_config
from preprocessing import preprocess


def main():
    parser = argparse.ArgumentParser(description="Score new transactions for fraud risk")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", required=True, help="CSV of new transactions to score")
    parser.add_argument("--output", default=None, help="Where to write scored output (default: stdout)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = joblib.load(cfg["model"]["output_path"])

    df = pd.read_csv(args.input)
    # TODO: preprocess() currently assumes a target column is present since it's
    # shared with training; for pure inference, adapt it to skip that step.
    processed = preprocess(df, cfg)

    id_column = cfg["data"].get("id_column")
    drop_columns = [c for c in [id_column, cfg["data"]["target_column"]] if c in processed.columns]
    X = processed.drop(columns=drop_columns)

    scores = model.predict_proba(X)[:, 1]
    result = df.copy()
    result["fraud_score"] = scores

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote scored transactions to {args.output}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
