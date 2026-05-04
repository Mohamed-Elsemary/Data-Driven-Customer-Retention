"""
Exploratory Data Analysis — univariate, bivariate, and
multivariate visualisations.

All functions expect a *cleaned* DataFrame (after data_cleaning.clean).
"""

import logging
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import CHURN_LABEL_PALETTE, CHURN_PALETTE, NUMERIC_COLS

logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  UNIVARIATE
# ═══════════════════════════════════════════════════════════════


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(len(NUMERIC_COLS), 2, figsize=(14, 4 * len(NUMERIC_COLS)))

    for i, col in enumerate(NUMERIC_COLS):
        sns.histplot(df[col], kde=True, ax=axes[i, 0], color="steelblue")
        axes[i, 0].set_title(f"{col} — distribution")
        axes[i, 0].set_xlabel(col)

        sns.boxplot(x=df[col], ax=axes[i, 1], color="lightblue")
        axes[i, 1].set_title(f"{col} — boxplot")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "univariate_numeric.png"), dpi=120)
    plt.close()
    logger.info("Numeric stats:\n%s", df[NUMERIC_COLS].describe().T)


def plot_binary_distributions(df: pd.DataFrame) -> None:
    binary_cols = [
        "SeniorCitizen",
        "PhoneService",
        "Partner",
        "PaperlessBilling",
        "Dependents",
        "Churn",
    ]
    fig, axes = plt.subplots(1, len(binary_cols), figsize=(14, 4))
    for i, col in enumerate(binary_cols):
        counts = df[col].value_counts().sort_index()
        axes[i].bar(["No (0)", "Yes (1)"], counts.values, color=["steelblue", "coral"])
        axes[i].set_title(col)
        axes[i].set_ylabel("Count")
        for j, v in enumerate(counts.values):
            axes[i].text(
                j,
                v + 30,
                f"{v}\n({v / len(df) * 100:.1f}%)",
                ha="center",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "univariate_binary.png"), dpi=120)
    plt.close()


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    cat_cols = [
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
    ]
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        counts = df[col].value_counts()
        axes[i].barh(counts.index.tolist(), counts.values, color="steelblue")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("Count")
        for j, v in enumerate(counts.values):
            axes[i].text(v + 20, j, f"{v}", va="center", fontsize=8)
    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "univariate_categorical.png"), dpi=120)
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  BIVARIATE
# ═══════════════════════════════════════════════════════════════


def plot_numeric_vs_churn(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i, col in enumerate(NUMERIC_COLS):
        sns.stripplot(
            data=df,
            x="Churn",
            y=col,
            hue="Churn",
            palette=CHURN_PALETTE,
            ax=axes[i],
            alpha=0.4,
            jitter=True,
            legend=False,
        )
        axes[i].set_title(f"{col} by Churn")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_numeric_vs_churn.png"), dpi=120)
    plt.close()


def plot_tenure_churn(df: pd.DataFrame) -> None:
    """Tenure lifecycle churn analysis."""
    tmp = df.copy()

    def categorize_tenure(t):
        if t <= 12:
            return "Under 1 year"
        elif t <= 24:
            return "Between 1 and 2 years"
        else:
            return "2 or more years"

    tmp["Tenure_Group"] = tmp["tenure"].apply(categorize_tenure)
    group_counts = tmp["Tenure_Group"].value_counts()
    churn_counts = tmp.groupby(["Tenure_Group", "Churn"]).size().unstack(fill_value=0)

    groups = ["Under 1 year", "Between 1 and 2 years", "2 or more years"]
    for group in groups:
        total = group_counts[group]
        churned = churn_counts.loc[group, 1]
        stayed = churn_counts.loc[group, 0]
        churn_rate = (churned / total) * 100
        logger.info("%s:", group)
        logger.info("  Total Customers: %d", total)
        logger.info("  Churned: %d (%.1f%%)", churned, churn_rate)
        logger.info("  Stayed:  %d (%.1f%%)", stayed, (stayed / total) * 100)

    churn_rates = tmp.groupby("Tenure_Group")["Churn"].mean() * 100
    churn_rates = churn_rates.reindex(groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=churn_rates.index, y=churn_rates.values, palette="Reds_r", ax=ax)
    for i, v in enumerate(churn_rates.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=12)
    ax.set_xlabel("Tenure Lifecycle", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 55)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_tenure_churn.png"), dpi=120)
    plt.close()


def plot_gender_churn(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["Churn_label"] = df["Churn"].map({0: "No Churn", 1: "Churn"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    gender_rate = df_plot.groupby("gender")["Churn"].mean() * 100
    axes[1].bar(
        gender_rate.index,
        gender_rate.values,
        color=["orchid", "steelblue"],
        edgecolor="white",
    )
    axes[1].set_title("Churn rate (%) by gender")
    axes[1].set_ylim(0, 40)
    for i, v in enumerate(gender_rate.values):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center")

    gender_churn = df_plot.groupby(["gender", "Churn_label"]).size().unstack()
    gender_churn.plot(
        kind="bar",
        ax=axes[0],
        color=["steelblue", "coral"],
        edgecolor="white",
    )
    axes[0].set_title("Gender vs Churn — counts")
    axes[0].set_xlabel("")
    axes[0].set_xticklabels(["Female", "Male"], rotation=0)
    axes[0].legend(title="")
    for container in axes[0].containers:
        axes[0].bar_label(container, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_gender_churn.png"), dpi=120)
    plt.close()


def plot_monthly_charges_churn(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["Churn_label"] = df_plot["Churn"].map({0: "No Churn", 1: "Churn"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Impact of Monthly Charges on Churn", fontsize=16, fontweight="bold")

    sns.kdeplot(
        data=df_plot,
        x="MonthlyCharges",
        hue="Churn_label",
        fill=True,
        palette=CHURN_LABEL_PALETTE,
        ax=axes[0],
        common_norm=False,
    )
    axes[0].set_title("Distribution of Monthly Charges")
    axes[0].set_xlabel("Monthly Charges ($)")
    axes[0].set_ylabel("Density")

    sns.boxplot(
        data=df_plot,
        x="Churn_label",
        y="MonthlyCharges",
        palette=CHURN_LABEL_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Median & Spread of Monthly Charges")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Monthly Charges ($)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_monthly_charges_churn.png"), dpi=120)
    plt.close()


def plot_total_charges_churn(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["Churn_label"] = df_plot["Churn"].map({0: "No Churn", 1: "Churn"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Impact of Total Charges on Churn", fontsize=16, fontweight="bold")

    sns.kdeplot(
        data=df_plot,
        x="TotalCharges",
        hue="Churn_label",
        fill=True,
        palette=CHURN_LABEL_PALETTE,
        ax=axes[0],
        common_norm=False,
    )
    axes[0].set_title("Distribution of Total Charges")
    axes[0].set_xlabel("Total Charges ($)")
    axes[0].set_ylabel("Density")

    sns.boxplot(
        data=df_plot,
        x="Churn_label",
        y="TotalCharges",
        palette=CHURN_LABEL_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Median & Spread of Total Charges")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Total Charges ($)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_total_charges_churn.png"), dpi=120)
    plt.close()


def plot_payment_method_churn(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["Churn_label"] = df_plot["Churn"].map({0: "No Churn", 1: "Churn"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Impact of Payment Method on Churn", fontsize=16, fontweight="bold")

    pm_rate = df_plot.groupby("PaymentMethod")["Churn"].mean() * 100
    pm_rate = pm_rate.sort_values(ascending=False)

    axes[0].bar(
        pm_rate.index,
        pm_rate.values,
        color=["red", "steelblue", "steelblue", "steelblue"],
        edgecolor="white",
    )
    axes[0].set_title("Churn Rate by Payment Method")
    axes[0].set_ylim(0, 60)
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(pm_rate.values):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    pm_churn = df_plot.groupby(["PaymentMethod", "Churn_label"]).size().unstack()
    pm_churn = pm_churn.loc[pm_rate.index]
    pm_churn.plot(
        kind="bar",
        ax=axes[1],
        color=["steelblue", "coral"],
        edgecolor="white",
        width=0.7,
    )
    axes[1].set_title("Payment Method vs Churn - Counts")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(title="")
    for container in axes[1].containers:
        axes[1].bar_label(container, fontsize=10, padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_payment_method_churn.png"), dpi=120)
    plt.close()


def plot_addon_churn(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Add-On Services — Individual Churn Rates", fontsize=16, fontweight="bold")

    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    for ax, col in zip(axes.flatten(), addon_cols):
        churn_rate = df.groupby(col)["Churn"].mean() * 100
        x_labels = churn_rate.index.astype(str).tolist()
        colors = ["#5b9bd5", "#e05c5c", "#f0a050"][: len(x_labels)]
        bars = ax.bar(x_labels, churn_rate.values, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, churn_rate.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_title(f"{col} vs Churn", fontsize=13)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, 55)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_addon_churn.png"), dpi=120)
    plt.close()


def plot_streaming_churn(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Streaming Services — Path to StreamingBundle", fontsize=15, fontweight="bold")

    churn_tv = df.groupby("StreamingTV")["Churn"].mean() * 100
    x_labels_tv = churn_tv.index.astype(str).tolist()
    colors_tv = ["#5b9bd5", "#e05c5c", "#f0a050"][: len(x_labels_tv)]
    bars_tv = axes[0].bar(x_labels_tv, churn_tv.values, color=colors_tv, width=0.5)
    for bar, v in zip(bars_tv, churn_tv.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold"
        )
    axes[0].set_title("StreamingTV vs Churn")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_ylim(0, 55)

    churn_mv = df.groupby("StreamingMovies")["Churn"].mean() * 100
    x_labels_mv = churn_mv.index.astype(str).tolist()
    colors_mv = ["#5b9bd5", "#e05c5c", "#f0a050"][: len(x_labels_mv)]
    bars_mv = axes[1].bar(x_labels_mv, churn_mv.values, color=colors_mv, width=0.5)
    for bar, v in zip(bars_mv, churn_mv.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold"
        )
    axes[1].set_title("StreamingMovies vs Churn")
    axes[1].set_ylim(0, 55)

    combo = df.groupby(["StreamingTV", "StreamingMovies"])["Churn"].mean() * 100
    combo_matrix = combo.unstack()
    sns.heatmap(combo_matrix, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=axes[2], linewidths=0.5)
    axes[2].set_title("StreamingTV × StreamingMovies Churn Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bivariate_streaming_churn.png"), dpi=120)
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  MULTIVARIATE
# ═══════════════════════════════════════════════════════════════


def plot_correlation_heatmaps(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["Churn_label"] = df_plot["Churn"].map({0: "No Churn", 1: "Churn"})

    num_bin_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "Churn",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    mask = np.triu(np.ones(shape=(len(num_bin_cols), len(num_bin_cols)), dtype=bool))

    for ax, method in zip(axes, ["spearman", "pearson"]):
        corr = df_plot[num_bin_cols].corr(method=method)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"{method.capitalize()} correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "multivariate_correlation_heatmaps.png"), dpi=120)
    plt.close()


def plot_internet_monthly_charges(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=df_plot, x="InternetService", y="MonthlyCharges", palette="Set2", ax=ax)
    ax.set_title("Monthly Charges by Internet Service Type", fontsize=16, fontweight="bold")
    ax.set_xlabel("Internet Service", fontsize=12, fontweight="bold")
    ax.set_ylabel("Monthly Charges", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "multivariate_internet_monthly_charges.png"), dpi=120)
    plt.close()


def plot_contract_internet_churn(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["Churn_label"] = df_plot["Churn"].map({0: "No Churn", 1: "Churn"})

    pivot = df_plot.groupby(["Contract", "InternetService"])["Churn"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Churn rate - Contract × Internet service")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "multivariate_contract_internet_churn.png"), dpi=120)
    plt.close()


def plot_tenure_charge_heatmap(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["tenure_band"] = pd.cut(
        df_plot["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0–12m", "13–24m", "25–48m", "49–72m"],
        right=True,
    )
    df_plot["charge_band"] = pd.cut(
        df_plot["MonthlyCharges"],
        bins=[0, 35, 65, 90, 120],
        labels=["Low", "Mid", "High", "Premium"],
    )
    pivot2 = df_plot.groupby(["tenure_band", "charge_band"])["Churn"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot2, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Churn rate — tenure band × charge band")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "multivariate_tenure_charge_heatmap.png"), dpi=120)
    plt.close()


def plot_pairplot(df: pd.DataFrame) -> None:
    pair_df = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].copy()
    pair_df["Churn"] = pair_df["Churn"].map({0: "No Churn", 1: "Churn"})
    g = sns.pairplot(
        pair_df,
        hue="Churn",
        palette=CHURN_LABEL_PALETTE,
        plot_kws={"alpha": 0.3, "s": 10},
        diag_kind="kde",
    )
    g.figure.suptitle("Pair plot - numeric features by Churn", y=1.01)
    g.savefig(os.path.join(PLOTS_DIR, "multivariate_pairplot.png"), dpi=120)
    plt.close()


def plot_payment_contract_churn(df: pd.DataFrame) -> None:
    grp = df.groupby(["PaymentMethod", "Contract"])["Churn"].mean().unstack() * 100
    grp.plot(kind="bar", figsize=(12, 5), colormap="coolwarm", edgecolor="white")
    plt.title("Churn rate (%) — Payment method × Contract type")
    plt.ylabel("Churn rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Contract", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "multivariate_payment_contract.png"), dpi=120)
    plt.close()


# ── Run all EDA ────────────────────────────────────────────────
def run_full_eda(df: pd.DataFrame) -> None:
    """Execute every EDA visualisation."""
    logger.info("=" * 60)
    logger.info("  UNIVARIATE ANALYSIS")
    logger.info("=" * 60)
    plot_numeric_distributions(df)
    plot_binary_distributions(df)
    plot_categorical_distributions(df)

    logger.info("=" * 60)
    logger.info("  BIVARIATE ANALYSIS")
    logger.info("=" * 60)
    plot_numeric_vs_churn(df)
    plot_tenure_churn(df)
    plot_gender_churn(df)
    plot_monthly_charges_churn(df)
    plot_total_charges_churn(df)
    plot_payment_method_churn(df)
    plot_addon_churn(df)
    plot_streaming_churn(df)

    logger.info("=" * 60)
    logger.info("  MULTIVARIATE ANALYSIS")
    logger.info("=" * 60)
    plot_correlation_heatmaps(df)
    plot_internet_monthly_charges(df)
    plot_contract_internet_churn(df)
    plot_tenure_charge_heatmap(df)
    plot_pairplot(df)
    plot_payment_contract_churn(df)
