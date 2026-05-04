import logging
import os
import kagglehub
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def download_and_load() -> pd.DataFrame:
    path = kagglehub.dataset_download("abbas829/telco-customer-churn-dataset")
    logger.info("Path to dataset files: %s", path)

    dataset_files = os.listdir(path)
    logger.info("Files downloaded: %s", dataset_files)

    csv_filename = [f for f in dataset_files if f.endswith(".csv")][0]
    full_csv_path = os.path.join(path, csv_filename)

    df = pd.read_csv(full_csv_path)
    return df


def show_churn_distribution(df: pd.DataFrame) -> None:
    """Quick bar chart of Churn vs No Churn — saved to plots/."""
    counts = df["Churn"].value_counts()
    labels = ["No Churn", "Churn"]
    colors = ["#4CAF50", "#F44336"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=colors)

    total = len(df)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{(height / total) * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.title("Churn vs No Churn Distribution")
    plt.ylabel("Number of Customers")

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "churn_distribution.png"), dpi=120)
    plt.close()
    logger.info("Churn distribution plot saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = download_and_load()
    logger.info("Head:\n%s", df.head())
    logger.info("Info:\n%s", df.dtypes)
    show_churn_distribution(df)
