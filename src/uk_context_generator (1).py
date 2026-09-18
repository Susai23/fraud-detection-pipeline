import pandas as pd
import random
from datetime import datetime, timedelta

URBAN_POSTCODES = [
    "SW1A 1AA", "M1 1AE", "B1 1AA", "LS1 1AA", "L1 1AA",
    "G1 1AA", "EH1 1AA", "CF10 1AA", "NE1 1AA", "S1 1AA",
    "BS1 1AA", "NG1 1AA", "OX1 1AA", "CB1 1AA", "BN1 1AA",
]
RURAL_POSTCODES = [
    "GL54 1AA", "YO61 1AA", "LD1 1AA", "IV2 1AA", "TQ9 1AA",
    "SA20 1AA", "PH15 1AA", "DT2 1AA", "NR35 1AA", "HR6 1AA",
]

def assign_postcode(severity: str) -> str:
    """No urban/rural flag in this dataset, so assign randomly with an urban bias
    (most claims happen in populated areas)."""
    return random.choice(URBAN_POSTCODES if random.random() < 0.7 else RURAL_POSTCODES)

def remap_to_recent_date(original_date_str: str, recent_years=(2023, 2024)) -> str:
    """
    Keeps the month/day pattern of the original incident_date but shifts
    the year into a recent window so weather/crime APIs have coverage.
    """
    try:
        original = pd.to_datetime(original_date_str)
        year = random.choice(recent_years)
        # handle Feb 29 safely by falling back to Feb 28 if needed
        day = min(original.day, 28) if original.month == 2 else original.day
        new_date = datetime(year, original.month, day)
        return new_date.strftime("%Y-%m-%d")
    except Exception:
        # fallback: random date in range
        year = random.choice(recent_years)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return datetime(year, month, day).strftime("%Y-%m-%d")

def add_uk_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["postcode"] = df["incident_severity"].apply(assign_postcode)
    df["claim_date"] = df["incident_date"].apply(remap_to_recent_date)
    return df
