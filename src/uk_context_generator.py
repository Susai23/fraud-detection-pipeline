import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# A spread of real UK postcodes across regions — mix of urban and rural
# so crime/weather signals vary meaningfully, loosely paired to AccidentArea
URBAN_POSTCODES = [
    "SW1A 1AA", "M1 1AE", "B1 1AA", "LS1 1AA", "L1 1AA",
    "G1 1AA", "EH1 1AA", "CF10 1AA", "NE1 1AA", "S1 1AA",
    "BS1 1AA", "NG1 1AA", "OX1 1AA", "CB1 1AA", "BN1 1AA",
]
RURAL_POSTCODES = [
    "GL54 1AA", "YO61 1AA", "LD1 1AA", "IV2 1AA", "TQ9 1AA",
    "SA20 1AA", "PH15 1AA", "DT2 1AA", "NR35 1AA", "HR6 1AA",
]

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def assign_postcode(accident_area: str) -> str:
    if str(accident_area).strip().lower() == "urban":
        return random.choice(URBAN_POSTCODES)
    return random.choice(RURAL_POSTCODES)

def assign_recent_date(month_name: str, week_of_month: int, recent_years=(2023, 2024)) -> str:
    """
    Builds a plausible recent date using the claim's original Month,
    approximate week-of-month, and a randomly chosen recent year
    (so it falls inside API coverage windows).
    """
    year = random.choice(recent_years)
    month = MONTH_MAP.get(str(month_name).strip(), random.randint(1, 12))

    # Approximate day from week-of-month (1-5), clipped to valid days in that month
    approx_day = (int(week_of_month) - 1) * 7 + random.randint(1, 7)
    try:
        date = datetime(year, month, min(approx_day, 28))
    except ValueError:
        date = datetime(year, month, 15)  # fallback safe day

    return date.strftime("%Y-%m-%d")

def add_uk_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["postcode"] = df["AccidentArea"].apply(assign_postcode)
    df["claim_date"] = df.apply(
        lambda row: assign_recent_date(row["Month"], row["WeekOfMonth"]), axis=1
    )
    return df
