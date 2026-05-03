"""
Feature engineering: encoding, engineered features, correlation
analysis, PCA, permutation importance, and final feature selection.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from config import (
    TERNARY_COLS, TERNARY_MAP, ADDON_COLS,
    RANDOM_STATE, TEST_SIZE, CORR_DROP_THRESHOLD, PERM_STRONG_THRESHOLD,
)

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  ENCODING
# ═══════════════════════════════════════════════════════════════

def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all encoding steps and drop customerID."""
    df = df.copy()
    df = df.drop(columns=["customerID"])

    # Binary encoding for 2-category columns
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype(int)
    df["PhoneService"] = df["PhoneService"].map({"Yes": 1, "No": 0}).astype(int)
    df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0}).astype(int)
    assert df[["gender", "PhoneService", "PaperlessBilling"]].isnull().sum().sum() == 0

    # Ternary → binary (collapse "No internet/phone service" into 0)
    for col in TERNARY_COLS:
        df[col] = df[col].astype(str).map(TERNARY_MAP).astype(int)
    assert df[TERNARY_COLS].isnull().sum().sum() == 0

    # One-hot encoding for nominal categories
    df = pd.get_dummies(df, columns=["InternetService", "PaymentMethod"], drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype("int8")

    logger.info("Final DataFrame shape: %s", df.shape)
    logger.info("Dtypes:\n%s", df.dtypes)
    logger.info("Any nulls: %d", df.isnull().sum().sum())
    return df


# ═══════════════════════════════════════════════════════════════
#  ENGINEERED FEATURES
# ═══════════════════════════════════════════════════════════════

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all custom features. Expects an encoded DataFrame."""
    df = df.copy()

    # Feature 1: Total Add-On Services
    df["_Total_AddOns-Services"] = df[ADDON_COLS].sum(axis=1)

    # Feature 2: Cost Per Service
    df["CoreServices_Count"] = df["PhoneService"] + (1 - df["InternetService_No"])
    df["True_Total_Services"] = df["CoreServices_Count"] + df["_Total_AddOns-Services"]
    df["_Cost_Per_Service"] = df["MonthlyCharges"] / df["True_Total_Services"]
    df = df.drop(columns=["CoreServices_Count", "True_Total_Services"])

    # Feature 3: Is AutoPay
    is_bank_transfer = (
        (df["PaymentMethod_Electronic check"] == 0)
        & (df["PaymentMethod_Mailed check"] == 0)
        & (df["PaymentMethod_Credit card (automatic)"] == 0)
    )
    df["_Is_AutoPay"] = (
        (df["PaymentMethod_Credit card (automatic)"] == 1) | is_bank_transfer
    ).astype(int)

    # Feature 4: Loyalty Score
    tenure_mean = df["tenure"].mean()
    monthly_mean = df["MonthlyCharges"].mean()
    df["_LoyaltyScore"] = (
        (df["tenure"] > tenure_mean) & (df["MonthlyCharges"] < monthly_mean)
    ).astype(int)
    logger.info("LoyaltyScore created (tenure mean: %.1f)", tenure_mean)

    # Feature 5: High Friction Payment
    df["_HighFriction_Payment"] = (
        (df["PaperlessBilling"] == 1)
        & (df["PaymentMethod_Electronic check"] == 1)
    ).astype(int)

    # Feature 6: Household Stability
    df["_Household_Stability"] = df["Partner"] + df["Dependents"]

    return df


# ═══════════════════════════════════════════════════════════════
#  FEATURE-ENGINEERING PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_feature_engineering_charts(df: pd.DataFrame) -> None:
    """Visualisations that accompany feature engineering."""

    # Total Add-Ons boxplot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Charge Per Service - Overpaying Signal", fontsize=16, fontweight="bold")
    sns.boxplot(
        data=df, x="_Total_AddOns-Services", y="MonthlyCharges",
        hue="Churn", palette={0: "#5b9bd5", 1: "#e05c5c"}, ax=ax,
    )
    ax.set_title("Distribution of Monthly Charges per Add-on Count")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["No Churn", "Churn"], title="Status", loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_addons_boxplot.png"), dpi=120)
    plt.close()

    # Cost Per Service
    custom_palette = {0: "steelblue", 1: "coral", "0": "steelblue", "1": "coral"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Impact of 'Cost Per Service' on Churn", fontsize=16, fontweight="bold")
    sns.kdeplot(data=df, x="_Cost_Per_Service", hue="Churn", fill=True,
                palette=custom_palette, ax=axes[0], common_norm=False)
    axes[0].set_title("Density of Cost Per Service")
    sns.boxplot(data=df, x="Churn", y="_Cost_Per_Service", hue="Churn",
                legend=False, palette=custom_palette, ax=axes[1])
    axes[1].set_title("Median Cost Per Service by Churn Status")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_cost_per_service.png"), dpi=120)
    plt.close()

    # AutoPay
    autopay_churn = df.groupby("_Is_AutoPay")["Churn"].mean() * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Manual Payment (0)", "AutoPay (1)"], y=autopay_churn.values,
                palette=["#e05c5c", "#5b9bd5"])
    plt.title("The Set and Forget Effect: AutoPay vs. Churn Rate", fontsize=14, fontweight="bold")
    plt.ylabel("Churn Rate"); plt.ylim(0, 45)
    for i, v in enumerate(autopay_churn.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_autopay_churn.png"), dpi=120)
    plt.close()

    # Loyalty Score
    loyalty_churn = df.groupby("_LoyaltyScore")["Churn"].mean() * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=loyalty_churn.index, y=loyalty_churn.values, palette="Set2")
    plt.title("Churn Rate by Loyalty Score", fontsize=14, fontweight="bold")
    plt.xlabel("Loyalty Score"); plt.ylabel("Churn Rate (%)")
    for i, val in enumerate(loyalty_churn.values):
        plt.text(i, val + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")
    plt.savefig(os.path.join(PLOTS_DIR, "fe_loyalty_churn.png"), dpi=120)
    plt.close()

    # High Friction Payment
    friction_churn = df.groupby("_HighFriction_Payment")["Churn"].mean() * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Low/Normal Friction (0)", "High Friction (1)"],
                y=friction_churn.values, palette=["#5b9bd5", "#e05c5c"])
    plt.title("The 'Payment Friction' Effect on Churn", fontsize=14, fontweight="bold")
    plt.ylabel("Churn Rate (%)"); plt.ylim(0, 55)
    for i, v in enumerate(friction_churn.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_friction_churn.png"), dpi=120)
    plt.close()

    # Household Stability
    household_churn = df.groupby("_Household_Stability")["Churn"].mean() * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Single (0)", "Partial Family (1)", "Full Family (2)"],
                y=household_churn.values, palette="viridis")
    plt.title("Churn Rate by Household Stability", fontsize=14, fontweight="bold")
    plt.ylabel("Churn Rate"); plt.ylim(0, 40)
    for i, v in enumerate(household_churn.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_household_churn.png"), dpi=120)
    plt.close()

    logger.info("Feature-engineering plots saved to %s", PLOTS_DIR)


# ═══════════════════════════════════════════════════════════════
#  SPLIT & FREQUENCY ENCODE
# ═══════════════════════════════════════════════════════════════

def split_and_encode(df: pd.DataFrame):
    """Train/test split + frequency encoding for Contract."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    freq_map = X_train["Contract"].value_counts(normalize=True)
    X_train["Contract"] = X_train["Contract"].map(freq_map)
    X_test["Contract"] = X_test["Contract"].map(freq_map)

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════
#  CORRELATION ANALYSIS & DROP
# ═══════════════════════════════════════════════════════════════

def correlation_analysis(X_train, y_train, X_test):
    """Plot correlations, drop highly correlated features, return updated splits."""
    df_corr = X_train.copy()
    df_corr["Churn"] = y_train

    corr_matrix = df_corr.drop(columns="Churn").corr(method="spearman")

    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 10},
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix (Spearman)", fontsize=16)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_correlation_matrix.png"), dpi=120)
    plt.close()

    upper = corr_matrix.abs().where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_DROP_THRESHOLD)]
    logger.info("Features to drop (corr > 0.85): %s", to_drop)

    features_to_drop = ["TotalCharges", "_Household_Stability"]
    X_train = X_train.drop(columns=features_to_drop)
    X_test = X_test.drop(columns=features_to_drop)
    df_corr = df_corr.drop(columns=features_to_drop)

    return X_train, X_test, df_corr


# ═══════════════════════════════════════════════════════════════
#  FEATURE SELECTION (Spearman + PCA + Permutation Importance)
# ═══════════════════════════════════════════════════════════════

def spearman_feature_selection(df_corr: pd.DataFrame):
    """Return selected_features from Spearman correlation with Churn."""
    df_corr = pd.get_dummies(df_corr, drop_first=True)
    target_corr = df_corr.corr(method="spearman")["Churn"].sort_values(ascending=False)
    logger.info("Spearman correlations with Churn:\n%s", target_corr)

    plt.figure(figsize=(8, 10))
    sns.heatmap(target_corr.to_frame(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation with Churn")
    plt.savefig(os.path.join(PLOTS_DIR, "fe_spearman_all.png"), dpi=120)
    plt.close()

    top_features = target_corr.abs().sort_values(ascending=False).head(11).index
    plt.figure(figsize=(6, 8))
    sns.heatmap(target_corr.loc[top_features].to_frame(), annot=True,
                cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Top Features Correlated with Churn")
    plt.savefig(os.path.join(PLOTS_DIR, "fe_spearman_top.png"), dpi=120)
    plt.close()

    selected_features = [
        "Contract", "tenure", "InternetService_Fiber optic",
        "_HighFriction_Payment", "_Cost_Per_Service",
        "InternetService_No", "_LoyaltyScore", "_Is_AutoPay",
    ]
    logger.info("Spearman-selected features:")
    for f in selected_features:
        logger.info("  • %s", f)
    return selected_features


def pca_analysis(X_train, y_train, all_features):
    """Full PCA analysis with scree plot, 2-D scatter, and loadings."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train[all_features])

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    n_95 = int(np.argmax(cumvar >= 95) + 1)
    logger.info("Components for 95%% variance: %d / %d", n_95, len(all_features))

    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(all_features) + 1),
                pca_full.explained_variance_ratio_ * 100,
                color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot"); axes[0].axhline(5, color="red", ls="--", lw=0.8)

    axes[1].plot(range(1, len(all_features) + 1), cumvar, marker="o", color="coral")
    axes[1].axhline(95, color="green", ls="--", lw=0.8)
    axes[1].axvline(n_95, color="green", ls=":", lw=0.8)
    axes[1].set_xlabel("Number of Components"); axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Variance — PCA")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_pca_scree.png"), dpi=120)
    plt.close()

    # 2-D scatter
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    for cls, label, color in [(0, "No Churn", "steelblue"), (1, "Churn", "coral")]:
        mask = y_train == cls
        plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                    label=label, alpha=0.4, s=15, color=color)
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("2-D PCA — Class Separability"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_pca_2d.png"), dpi=120)
    plt.close()

    # Loadings heatmap
    loadings = pd.DataFrame(
        pca_full.components_[:n_95], columns=all_features,
        index=[f"PC{i+1}" for i in range(n_95)],
    )
    plt.figure(figsize=(max(16, len(all_features)), n_95 + 2))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5)
    plt.title(f"PCA Loadings — Top {n_95} Components"); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fe_pca_loadings.png"), dpi=120)
    plt.close()


def permutation_feature_importance(X_train, X_test, y_train, y_test,
                                   all_features, selected_features):
    """Run permutation importance and return final_features list."""
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train[all_features], y_train)

    perm = permutation_importance(
        rf, X_test[all_features], y_test,
        n_repeats=30, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    logger.info("Permutation Importance (ROC-AUC drop):\n%s", perm_df.to_string(index=False))

    plt.figure(figsize=(10, max(6, len(all_features) // 2)))
    plt.barh(perm_df["Feature"][::-1], perm_df["Importance"][::-1],
             xerr=perm_df["Std"][::-1], color="steelblue", edgecolor="white", capsize=4)
    plt.axvline(0, color="black", lw=0.8, ls="--")
    plt.xlabel("Mean Decrease in ROC-AUC (± std)")
    plt.title("Permutation Feature Importance — All Features (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "permutation_importance_all.png"), dpi=120)
    plt.close()

    strong = perm_df[perm_df["Importance"] >= PERM_STRONG_THRESHOLD]["Feature"].tolist()
    moderate = perm_df[
        (perm_df["Importance"] > 0) & (perm_df["Importance"] < PERM_STRONG_THRESHOLD)
    ]["Feature"].tolist()
    weak = perm_df[perm_df["Importance"] <= 0]["Feature"].tolist()

    logger.info("Tier 1 — Strong (≥ 0.01): %s", strong)
    logger.info("Tier 2 — Moderate (0 < x < 0.01): %s", moderate)
    logger.info("Tier 3 — Weak / Negligible: %s", weak)

    final_features = strong + moderate
    logger.info("Final features (%d):", len(final_features))
    for f in final_features:
        logger.info("  • %s", f)

    # Benchmark
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_bench = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    for label, feats in [
        (f"Spearman only ({len(selected_features)})", selected_features),
        (f"Permutation-filtered ({len(final_features)})", final_features),
    ]:
        scores = cross_val_score(rf_bench, X_train[feats], y_train,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
        logger.info("  %-50s  ROC-AUC = %.4f ± %.4f", label, scores.mean(), scores.std())

    return final_features
