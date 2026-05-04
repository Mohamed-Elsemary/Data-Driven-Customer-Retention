"""
Shared pytest fixtures for the Telco Customer Churn pipeline tests.
"""

import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A 10-row realistic DataFrame matching the Telco schema.
    Large enough for meaningful train/test splits."""
    data = {
        "customerID": [f"C-{i:04d}" for i in range(10)],
        "gender": ["Male", "Female"] * 5,
        "SeniorCitizen": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes"],
        "tenure": [12, 0, 36, 48, 72, 5, 24, 60, 3, 40],
        "PhoneService": ["Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes"],
        "MultipleLines": [
            "No", "No phone service", "Yes", "No", "Yes",
            "No phone service", "No", "Yes", "No phone service", "No",
        ],
        "InternetService": [
            "Fiber optic", "DSL", "No", "Fiber optic", "DSL",
            "DSL", "Fiber optic", "No", "DSL", "Fiber optic",
        ],
        "OnlineSecurity": [
            "No", "Yes", "No internet service", "No", "Yes",
            "No", "No", "No internet service", "Yes", "No",
        ],
        "OnlineBackup": [
            "Yes", "No", "No internet service", "No", "Yes",
            "Yes", "No", "No internet service", "No", "Yes",
        ],
        "DeviceProtection": [
            "No", "Yes", "No internet service", "Yes", "No",
            "No", "Yes", "No internet service", "No", "Yes",
        ],
        "TechSupport": [
            "No", "No", "No internet service", "Yes", "Yes",
            "No", "No", "No internet service", "Yes", "No",
        ],
        "StreamingTV": [
            "Yes", "No", "No internet service", "Yes", "No",
            "Yes", "No", "No internet service", "No", "Yes",
        ],
        "StreamingMovies": [
            "No", "Yes", "No internet service", "No", "Yes",
            "No", "Yes", "No internet service", "Yes", "No",
        ],
        "Contract": [
            "Month-to-month", "One year", "Two year", "Month-to-month", "Two year",
            "Month-to-month", "One year", "Two year", "Month-to-month", "One year",
        ],
        "PaperlessBilling": [
            "Yes", "No", "No", "Yes", "No",
            "Yes", "No", "Yes", "No", "Yes",
        ],
        "PaymentMethod": [
            "Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)", "Bank transfer (automatic)",
            "Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)", "Electronic check",
        ],
        "MonthlyCharges": [70.35, 29.85, 20.05, 99.65, 45.25, 55.10, 65.40, 19.95, 35.50, 89.90],
        "TotalCharges": [
            "844.2", "0", "721.8", "4783.2", "3258",
            "275.5", "1569.6", "1197", "106.5", "3596",
        ],
        "Churn": ["Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def cleaned_df(raw_df):
    """Apply the real clean() function to the raw fixture."""
    from data_cleaning import clean

    return clean(raw_df)


@pytest.fixture
def encoded_df(cleaned_df):
    """Apply encode() to the cleaned fixture."""
    from feature_engineering import encode

    return encode(cleaned_df)


@pytest.fixture
def featured_df(encoded_df):
    """Apply add_engineered_features() to the encoded fixture."""
    from feature_engineering import add_engineered_features

    return add_engineered_features(encoded_df)
