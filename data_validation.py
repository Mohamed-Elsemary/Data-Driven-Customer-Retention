import logging
import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Accuracy 
def check_accuracy(df: pd.DataFrame) -> None:
    logger.info("Categorical Invalid Accuracy Check:")
    logger.info("Gender invalid: %d", df[~df["gender"].isin(["Male", "Female"])].shape[0])
    logger.info("SeniorCitizen invalid: %d", df[~df["SeniorCitizen"].isin([0, 1])].shape[0])
    logger.info("Partner invalid: %d", df[~df["Partner"].isin(["Yes", "No"])].shape[0])
    logger.info("Dependents invalid: %d", df[~df["Dependents"].isin(["Yes", "No"])].shape[0])
    logger.info("PhoneService invalid: %d", df[~df["PhoneService"].isin(["Yes", "No"])].shape[0])
    logger.info(
        "MultipleLines invalid: %d",
        df[~df["MultipleLines"].isin(["Yes", "No", "No phone service"])].shape[0],
    )
    logger.info(
        "InternetService invalid: %d",
        df[~df["InternetService"].isin(["DSL", "Fiber optic", "No"])].shape[0],
    )
    logger.info(
        "Contract invalid: %d",
        df[~df["Contract"].isin(["Month-to-month", "One year", "Two year"])].shape[0],
    )
    logger.info(
        "PaperlessBilling invalid: %d",
        df[~df["PaperlessBilling"].isin(["Yes", "No"])].shape[0],
    )
    logger.info("Churn invalid: %d", df[~df["Churn"].isin(["Yes", "No"])].shape[0])

    logger.info("\nService-related accuracy checks:")
    for col in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        logger.info(
            "%s invalid: %d",
            col,
            df[~df[col].isin(["Yes", "No", "No internet service"])].shape[0],
        )

    logger.info("\nNumeric accuracy checks:")
    logger.info("Negative tenure: %d", (df["tenure"] < 0).sum())
    logger.info("Negative MonthlyCharges: %d", (df["MonthlyCharges"] < 0).sum())

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    logger.info("Missing TotalCharges: %d", df["TotalCharges"].isna().sum())
    logger.info("Negative TotalCharges: %d", (df["TotalCharges"] < 0).sum())

   
    logger.info("\ncustomerID format issues checks:")
    logger.info(
        "Invalid customerID format: %d",
        df[~df["customerID"].astype(str).str.match("^[0-9A-Za-z-]+$")].shape[0],
    )

    # Inspect missing TotalCharges
    logger.info("Sample missing TotalCharges row:\n%s", df[df["TotalCharges"].isna()].head(1).T)
    missing_df = df[df["TotalCharges"].isna()]
    logger.info("Missing TotalCharges count: %d", len(missing_df))
    logger.info("All are new customers (tenure=0): %s", (missing_df["tenure"] == 0).all())


# Consistency 
def check_consistency(df: pd.DataFrame) -> None:
   # All customers with No internet service correctly have No internet service in all associated features, indicating strong internal consistency.
    n = df[
        (df["InternetService"] == "No")
        & (
            (df["OnlineSecurity"] != "No internet service")
            | (df["OnlineBackup"] != "No internet service")
            | (df["DeviceProtection"] != "No internet service")
            | (df["TechSupport"] != "No internet service")
            | (df["StreamingTV"] != "No internet service")
            | (df["StreamingMovies"] != "No internet service")
        )
    ].shape[0]

    logger.info("InternetService vs add-ons inconsistencies: %d", n)

    # PhoneService vs MultipleLines
    n2 = df[(df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")].shape[0]
    logger.info("PhoneService vs MultipleLines inconsistencies: %d", n2)

    # Reverse check
    n3 = df[
        (df["InternetService"] != "No")
        & (
            (df["OnlineSecurity"] == "No internet service")
            | (df["OnlineBackup"] == "No internet service")
        )
    ].shape[0]
    logger.info("Has internet but says 'No internet service': %d", n3)

    # Tenure > 0 but TotalCharges missing
    n4 = df[(df["tenure"] > 0) & (df["TotalCharges"].isna())].shape[0]
    logger.info("tenure > 0 with missing TotalCharges: %d", n4)


# Completeness
def check_completeness(df: pd.DataFrame) -> None:
    logger.info("Missing values per column:\n%s", df.isnull().sum())

    logger.info(
        "Missing TotalCharges detail:\n%s",
        df[df["TotalCharges"].isna()][["tenure", "MonthlyCharges", "Contract", "Churn"]],
    )

    pct = ((df.isnull().sum() / len(df)) * 100)[df.isnull().sum() > 0]
    if len(pct):
        logger.info("Missing %% for non-zero columns:\n%s", pct)

    critical_cols = ["customerID", "gender", "tenure", "MonthlyCharges", "Churn"]
    logger.info("Critical-column nulls:\n%s", df[critical_cols].isnull().sum())


# Duplicates 
def check_duplicates(df: pd.DataFrame) -> None:
    logger.info("Duplicate rows: %d", df.duplicated().sum())
    logger.info("Duplicate customerIDs: %d", df["customerID"].duplicated().sum())


# Outlier Detection
def detect_outliers(df: pd.DataFrame) -> None:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Distribution of {col}")
        fig.savefig(os.path.join(PLOTS_DIR, f"validation_dist_{col}.png"), dpi=120)
        plt.close(fig)

    Q1 = df["TotalCharges"].quantile(0.25)
    Q3 = df["TotalCharges"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df["TotalCharges"] < lower_bound) | (df["TotalCharges"] > upper_bound)]
    logger.info("Q1: %.2f, Q3: %.2f, IQR: %.2f", Q1, Q3, IQR)
    logger.info("lower_bound: %.2f", lower_bound)
    logger.info("upper_bound: %.2f", upper_bound)
    logger.info("Outliers found: %d", len(outliers))


# Distribution Profile
def distribution_profile(df: pd.DataFrame) -> None:
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    logger.info("--- DISTRIBUTION PROFILE ---")
    logger.info("Statistical Summary:\n%s", df[cols].describe())
    logger.info("Skewness:\n%s", df[cols].skew())
    logger.info("Kurtosis:\n%s", df[cols].kurtosis())


# Relationship Profile
def relationship_profile(df: pd.DataFrame) -> None:
    cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    logger.info("Pearson Correlation:\n%s", df[cols].corr(method="pearson"))
    logger.info("Spearman Correlation:\n%s", df[cols].corr(method="spearman"))

    fig, ax = plt.subplots()
    cax = ax.matshow(df[cols].corr())
    fig.colorbar(cax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cols, rotation=45)
    ax.set_yticks(range(3))
    ax.set_yticklabels(cols)
    ax.set_title("Correlation Matrix")
    fig.savefig(os.path.join(PLOTS_DIR, "validation_correlation_matrix.png"), dpi=120)
    plt.close(fig)
    logger.info("Correlation matrix plot saved.")


def run_all_validations(df: pd.DataFrame) -> None:
    check_accuracy(df)
    check_consistency(df)
    check_completeness(df)
    check_duplicates(df)
    detect_outliers(df)
    distribution_profile(df)
    relationship_profile(df)

