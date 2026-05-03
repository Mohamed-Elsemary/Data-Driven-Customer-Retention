"""
Data cleaning: type correction, missing-value imputation,
and categorical encoding.
"""

import logging
import pandas as pd
from config import CAT_COLS_TO_STRING, BINARY_MAP, TERNARY_COLS, TERNARY_MAP

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw DataFrame."""
    df = df.copy()

    # ── Type correction ────────────────────────────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    for col in CAT_COLS_TO_STRING:
        df[col] = df[col].astype("string")

    # ── Binary encoding (Yes/No → 1/0) ────────────────────────
    df["Partner"] = df["Partner"].astype(str).str.strip().map(BINARY_MAP)
    df["Dependents"] = df["Dependents"].astype(str).str.strip().map(BINARY_MAP)
    df["Churn"] = df["Churn"].astype(str).str.strip().map(BINARY_MAP)

    logger.info("Column dtypes after cleaning:\n%s", df.dtypes)
    logger.info("Duplicated rows: %d", df.duplicated().sum())
    logger.info("Unique customerIDs: %d", df["customerID"].nunique())

    return df
