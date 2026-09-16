"""Feature engineering for fraud signals.

Fill these in based on the columns available in your dataset. Common
fraud-relevant signals include transaction velocity (count/amount of
recent transactions per account), deviation from an account's typical
spend, time-of-day / day-of-week patterns, and merchant-category risk.
"""
import pandas as pd


def add_amount_zscore(df: pd.DataFrame, amount_column: str, group_column: str = None) -> pd.DataFrame:
    df = df.copy()
    if group_column and group_column in df.columns:
        grouped = df.groupby(group_column)[amount_column]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, 1)
    else:
        mean = df[amount_column].mean()
        std = df[amount_column].std() or 1
    df[f"{amount_column}_zscore"] = (df[amount_column] - mean) / std
    return df


def add_time_features(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_column])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    return df


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # TODO: wire up the feature functions above (and any others you add)
    # based on cfg["features"] and the columns present in your dataset.
    return df
