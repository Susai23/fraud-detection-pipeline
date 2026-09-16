"""Placeholder test to keep pytest wired up until real fixtures/data exist."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import add_amount_zscore
import pandas as pd


def test_add_amount_zscore():
    df = pd.DataFrame({"amount": [10, 20, 30, 40]})
    result = add_amount_zscore(df, "amount")
    assert "amount_zscore" in result.columns
    assert len(result) == len(df)
