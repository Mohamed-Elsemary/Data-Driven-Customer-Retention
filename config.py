"""
Shared constants, column lists, color palettes, and random seed
used across all pipeline modules.
"""

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.55  # classification probability threshold

# Column groups 
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_MAP = {"Yes": 1, "No": 0}

CAT_COLS_TO_STRING = [
    "customerID",
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "PhoneService",
    "PaperlessBilling",
]

TERNARY_COLS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TERNARY_MAP = {
    "Yes": 1,
    "No": 0,
    "No internet service": 0,
    "No phone service": 0,
}

ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "TechSupport"]

# Color palettes
CHURN_PALETTE = {0: "steelblue", 1: "coral"}
CHURN_LABEL_PALETTE = {"No Churn": "steelblue", "Churn": "coral"}

# Correlation / feature-selection thresholds
CORR_DROP_THRESHOLD = 0.85
PERM_STRONG_THRESHOLD = 0.01

# Grid-search parameter grids 
LR_PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "class_weight": ["balanced"],
}

DT_PARAM_GRID = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "class_weight": ["balanced"],
}

RF_PARAM_GRID = {
    "n_estimators": [200, 300],
    "max_depth": [8, 10, 15],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"],
}

XGB_PARAM_GRID = {
    "n_estimators": [200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1],
}

LGB_PARAM_GRID = {
    "n_estimators": [200, 300],
    "learning_rate": [0.01, 0.05],
    "num_leaves": [31, 50],
    "max_depth": [-1, 10, 15],
}

# Business-metric assumptions
AVG_CUSTOMER_LIFETIME_MONTHS = 12  # expected months a retained customer stays
