# Fraud Detection Pipeline

An end-to-end skeleton for a transaction fraud detection pipeline: data ingestion,
preprocessing, feature engineering, model training, evaluation, and inference.

This is a **scaffold** — the scripts define the pipeline stages and CLI interfaces,
but contain no trained model yet. Plug in a dataset (e.g. a labeled transactions
CSV with a `is_fraud` target column) and fill in the `TODO`s to make it run
end-to-end.

## Project Structure

```
fraud-detection-pipeline/
├── config/
│   └── config.yaml        # Pipeline configuration (paths, model params, thresholds)
├── data/
│   ├── raw/                # Raw, unmodified input data
│   └── processed/          # Cleaned / feature-engineered data
├── models/                  # Serialized trained models
├── notebooks/                # Exploratory analysis
├── src/
│   ├── data_loader.py       # Load raw data into a DataFrame
│   ├── preprocessing.py     # Cleaning, missing values, encoding
│   ├── features.py          # Feature engineering
│   ├── train.py              # Train a classifier and save it
│   ├── evaluate.py           # Evaluate a trained model on a holdout set
│   └── predict.py            # Score new transactions with a trained model
├── tests/
│   └── test_placeholder.py
└── requirements.txt
```

## Pipeline Stages

1. **Load** (`src/data_loader.py`) — read raw transaction data from `data/raw/`.
2. **Preprocess** (`src/preprocessing.py`) — handle missing values, encode
   categorical fields, scale numeric fields.
3. **Feature engineering** (`src/features.py`) — derive fraud-relevant signals
   (e.g. transaction velocity, amount z-scores, time-of-day).
4. **Train** (`src/train.py`) — fit a classifier (e.g. logistic regression /
   gradient boosting) on the processed, labeled data and persist it to `models/`.
5. **Evaluate** (`src/evaluate.py`) — score the held-out set (precision, recall,
   ROC-AUC, PR-AUC — fraud datasets are typically highly imbalanced, so
   accuracy alone is not a useful metric).
6. **Predict** (`src/predict.py`) — load a trained model and score new,
   unlabeled transactions.

## Getting Started

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Once you've added a dataset to data/raw/ and implemented the TODOs:
python src/train.py --config config/config.yaml
python src/evaluate.py --config config/config.yaml
python src/predict.py --config config/config.yaml --input data/raw/new_transactions.csv
```

## Configuration

Pipeline parameters (data paths, target column, model hyperparameters,
train/test split) live in `config/config.yaml`.

## Status

Scaffold only — no dataset or trained model is included yet.
