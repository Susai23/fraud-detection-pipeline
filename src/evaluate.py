"""Evaluate a trained fraud model on the held-out test split.

Fraud datasets are typically highly imbalanced, so accuracy alone is not
a meaningful metric here — precision/recall/PR-AUC matter more than
overall accuracy.
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fraud detection model")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_column = cfg["data"]["target_column"]

    test_path = cfg["data"]["processed_path"].replace(".csv", "_test.csv")
    df = pd.read_csv(test_path)

    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    model = joblib.load(cfg["model"]["output_path"])
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    with open(cfg["evaluation"]["report_path"], "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
