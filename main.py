"""
Main pipeline orchestrator — runs the full end-to-end workflow:
  1. Load data
  2. Validate data quality
  3. Clean data
  4. Exploratory Data Analysis
  5. Feature engineering & selection
  6. Model training & evaluation
"""

import logging
import sys

from data_cleaning import clean
from data_loading import download_and_load, show_churn_distribution
from data_validation import run_all_validations
from eda import run_full_eda
from feature_engineering import (
    add_engineered_features,
    correlation_analysis,
    encode,
    pca_analysis,
    permutation_feature_importance,
    plot_feature_engineering_charts,
    spearman_feature_selection,
    split_and_encode,
)
from modeling import (
    cross_validate_best,
    evaluate_models,
    grid_search_all,
    save_outputs,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Set up root logger: console + file output."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    fh = logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def main():
    _configure_logging()

    # ── 1. Load ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 1: LOADING DATA")
    logger.info("=" * 60)
    df_raw = download_and_load()
    logger.info("First 5 rows:\n%s", df_raw.head())
    logger.info("DataFrame info:\n%s", df_raw.dtypes)
    show_churn_distribution(df_raw)

    # ── 2. Validate ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 2: DATA VALIDATION")
    logger.info("=" * 60)
    run_all_validations(df_raw)

    # ── 3. Clean ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 3: DATA CLEANING")
    logger.info("=" * 60)
    df = clean(df_raw)

    # ── 4. EDA ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 4: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)
    run_full_eda(df)

    # ── 5. Feature Engineering ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 5: FEATURE ENGINEERING")
    logger.info("=" * 60)
    df = encode(df)
    df = add_engineered_features(df)
    plot_feature_engineering_charts(df)

    X_train, X_test, y_train, y_test = split_and_encode(df)
    X_train, X_test, df_corr = correlation_analysis(X_train, y_train, X_test)
    selected_features = spearman_feature_selection(df_corr)

    all_features = X_train.columns.tolist()
    pca_analysis(X_train, y_train, all_features)
    final_features = permutation_feature_importance(
        X_train,
        X_test,
        y_train,
        y_test,
        all_features,
        selected_features,
    )

    # ── 6. Modelling ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  STEP 6: MODELLING")
    logger.info("=" * 60)
    setup_mlflow()

    monthly_charges_test = X_test["MonthlyCharges"].copy()
    X_train_final = X_train[final_features].copy()
    X_test_final = X_test[final_features].copy()

    best_models = grid_search_all(X_train_final, y_train)
    results, results_df = evaluate_models(
        best_models,
        X_train_final,
        X_test_final,
        y_train,
        y_test,
        monthly_charges_test,
    )

    # Cross-validate the last (LightGBM) model with custom threshold
    cross_validate_best(best_models["LightGBM"], X_train_final, y_train)

    # ── 7. Save ────────────────────────────────────────────────
    save_outputs(df, results)
    logger.info("✅ Pipeline complete!")


if __name__ == "__main__":
    main()
