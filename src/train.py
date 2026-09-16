"""Train a fraud classifier and persist it to disk."""
import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import load_config


def build_model(cfg: dict):
    model_cfg = cfg["model"]
    if model_cfg["type"] == "logistic_regression":
        return LogisticRegression(**model_cfg["params"])
    # TODO: add branches for other model types (random_forest, gradient_boosting, ...)
    raise ValueError(f"Unsupported model type: {model_cfg['type']}")


def main():
    parser = argparse.ArgumentParser(description="Train the fraud detection model")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg["data"]["processed_path"])

    target_column = cfg["data"]["target_column"]
    drop_columns = [target_column]
    if cfg["data"].get("id_column") in df.columns:
        drop_columns.append(cfg["data"]["id_column"])

    X = df.drop(columns=drop_columns)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    model = build_model(cfg)
    model.fit(X_train, y_train)

    joblib.dump(model, cfg["model"]["output_path"])
    print(f"Saved trained model to {cfg['model']['output_path']}")

    # Persist the holdout split so evaluate.py can reuse it.
    X_test.assign(**{target_column: y_test}).to_csv(
        cfg["data"]["processed_path"].replace(".csv", "_test.csv"), index=False
    )


if __name__ == "__main__":
    main()
