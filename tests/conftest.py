"""
Shared pytest fixtures for the Telco Customer Churn pipeline tests.

Adds the parent directory (code/) to sys.path so that pipeline modules
can be imported the same way they import each other (flat imports).
"""

import os
import sys

import pandas as pd
import pytest

# ── Make pipeline modules importable ──────────────────────────
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


# ═══════════════════════════════════════════════════════════════
#  RAW DATAFRAME (mimics the Kaggle download)
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small, realistic raw DataFrame matching the Telco schema."""
    data = {
        "customerID": ["0001-A", "0002-B", "0003-C", "0004-D", "0005-E"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "Yes"],
        "tenure": [12, 0, 36, 48, 72],
        "PhoneService": ["Yes", "No", "Yes", "Yes", "Yes"],
        "MultipleLines": [
            "No",
            "No phone service",
            "Yes",
            "No",
            "Yes",
        ],
        "InternetService": [
            "Fiber optic",
            "DSL",
            "No",
            "Fiber optic",
            "DSL",
        ],
        "OnlineSecurity": [
            "No",
            "Yes",
            "No internet service",
            "No",
            "Yes",
        ],
        "OnlineBackup": [
            "Yes",
            "No",
            "No internet service",
            "No",
            "Yes",
        ],
        "DeviceProtection": [
            "No",
            "Yes",
            "No internet service",
            "Yes",
            "No",
        ],
        "TechSupport": [
            "No",
            "No",
            "No internet service",
            "Yes",
            "Yes",
        ],
        "StreamingTV": [
            "Yes",
            "No",
            "No internet service",
            "Yes",
            "No",
        ],
        "StreamingMovies": [
            "No",
            "Yes",
            "No internet service",
            "No",
            "Yes",
        ],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "Two year",
        ],
        "PaperlessBilling": ["Yes", "No", "No", "Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Bank transfer (automatic)",
        ],
        "MonthlyCharges": [70.35, 29.85, 20.05, 99.65, 45.25],
        "TotalCharges": ["844.2", "0", "721.8", "4783.2", "3258"],
        "Churn": ["Yes", "No", "No", "Yes", "No"],
    }
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════
#  CLEANED DATAFRAME (after data_cleaning.clean)
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def cleaned_df(raw_df):
    """Apply the real clean() function to the raw fixture."""
    from data_cleaning import clean

    return clean(raw_df)
