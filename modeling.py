"""
Model training: grid-search tuning, evaluation,
cross-validated scoring, and MLflow experiment tracking.
"""

import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    RANDOM_STATE, THRESHOLD,
    LR_PARAM_GRID, DT_PARAM_GRID, RF_PARAM_GRID,
    XGB_PARAM_GRID, LGB_PARAM_GRID,
    RETENTION_CAMPAIGN_COST, AVG_CUSTOMER_LIFETIME_MONTHS,
)

logger = logging.getLogger(__name__)


def _make_skf():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def setup_mlflow(experiment_name="Telco-Customer-Churn"):
    """Initialise MLflow experiment (uses default ./mlruns store)."""
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow experiment : %s", experiment_name)
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())


def _sanitize_params(params: dict) -> dict:
    """Convert model .get_params() to MLflow-safe flat strings."""
    return {k: str(v)[:250] for k, v in params.items()}


def _compute_business_metrics(y_true, y_pred, monthly_charges):
    """Compute business-oriented churn metrics from confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tp_mask = (y_true == 1) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    avg_charge_tp = float(monthly_charges[tp_mask].mean()) if tp_mask.sum() > 0 else 0.0
    avg_charge_fn = float(monthly_charges[fn_mask].mean()) if fn_mask.sum() > 0 else 0.0

    # Business 1 — Revenue Saved: correctly-flagged churners × CLV
    revenue_saved = tp * avg_charge_tp * AVG_CUSTOMER_LIFETIME_MONTHS

    # Business 2 — Net Retention Value: revenue saved minus campaign spend
    campaign_cost = (tp + fp) * RETENTION_CAMPAIGN_COST
    net_retention_value = revenue_saved - campaign_cost

    # Business 3 — Cost per True Churn Detected
    cost_per_detected = campaign_cost / tp if tp > 0 else 0.0

    # Business 4 — Revenue Leakage: missed churners × CLV
    revenue_leakage = fn * avg_charge_fn * AVG_CUSTOMER_LIFETIME_MONTHS

    return {
        "Revenue_Saved": round(revenue_saved, 2),
        "Net_Retention_Value": round(net_retention_value, 2),
        "Cost_Per_Detection": round(cost_per_detected, 2),
        "Revenue_Leakage": round(revenue_leakage, 2),
    }


# ═══════════════════════════════════════════════════════════════
#  GRID SEARCH
# ═══════════════════════════════════════════════════════════════

def grid_search_all(X_train, y_train):
    """Run GridSearchCV for LR, DT, RF, XGB, and LightGBM. Return best estimators."""
    skf = _make_skf()

    # Logistic Regression
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=LR_PARAM_GRID,
        scoring="f1", cv=skf, n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)

    # Decision Tree
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=DT_PARAM_GRID,
        scoring="f1", cv=skf, n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)

    # Random Forest
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=RF_PARAM_GRID,
        scoring="f1", cv=skf, n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)

    # XGBoost
    xgb_grid = GridSearchCV(
        XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
            random_state=RANDOM_STATE,
        ),
        param_grid=XGB_PARAM_GRID,
        scoring="f1", cv=skf, n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train)

    # LightGBM
    lgb_grid = GridSearchCV(
        LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        param_grid=LGB_PARAM_GRID,
        scoring="f1", cv=skf, n_jobs=-1,
    )
    lgb_grid.fit(X_train, y_train)

    best_models = {
        "Logistic Regression": lr_grid.best_estimator_,
        "Decision Tree": dt_grid.best_estimator_,
        "Random Forest": rf_grid.best_estimator_,
        "XGBoost": xgb_grid.best_estimator_,
        "LightGBM": lgb_grid.best_estimator_,
    }

    logger.info("Best parameters:")
    for name, grid in [("LR", lr_grid), ("DT", dt_grid), ("RF", rf_grid),
                        ("XGB", xgb_grid), ("LGB", lgb_grid)]:
        logger.info("  %s: %s", name, grid.best_params_)

    return best_models


# ═══════════════════════════════════════════════════════════════
#  EVALUATION  +  MLFLOW LOGGING
# ═══════════════════════════════════════════════════════════════

def evaluate_models(best_models, X_train, X_test, y_train, y_test,
                    monthly_charges_test):
    """Fit each model, evaluate on test set, and log everything to MLflow."""
    results = {}

    for name, model in best_models.items():
        with mlflow.start_run(run_name=name):
            # ── Tag ────────────────────────────────────────────
            mlflow.set_tag("model_name", name)

            # ── Train & predict ────────────────────────────────
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            # ── Log hyper-parameters ───────────────────────────
            mlflow.log_params(_sanitize_params(model.get_params()))

            # ── Standard metrics ───────────────────────────────
            standard = {
                "Accuracy":  accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds),
                "Recall":    recall_score(y_test, preds),
                "F1":        f1_score(y_test, preds),
                "ROC-AUC":   roc_auc_score(y_test, probs),
            }
            mlflow.log_metrics({
                "accuracy":  standard["Accuracy"],
                "precision": standard["Precision"],
                "recall":    standard["Recall"],
                "f1":        standard["F1"],
                "roc_auc":   standard["ROC-AUC"],
            })

            # ── Business metrics ───────────────────────────────
            business = _compute_business_metrics(
                y_test.values, preds, monthly_charges_test.values,
            )
            mlflow.log_metrics({
                "revenue_saved":          business["Revenue_Saved"],
                "net_retention_value":    business["Net_Retention_Value"],
                "cost_per_detection":     business["Cost_Per_Detection"],
                "revenue_leakage":        business["Revenue_Leakage"],
            })

            # ── Save model artifact ────────────────────────────
            mlflow.sklearn.log_model(model, artifact_path="model")

            results[name] = {**standard, **business}
            logger.info("  ✓ %s logged to MLflow", name)

    results_df = pd.DataFrame(results).T
    logger.info("\n%s", results_df.to_string())
    return results, results_df


# ═══════════════════════════════════════════════════════════════
#  CROSS-VALIDATION WITH CUSTOM THRESHOLD
# ═══════════════════════════════════════════════════════════════

def cross_validate_best(model, X_train, y_train, threshold=THRESHOLD):
    """5-fold CV with a custom probability threshold."""
    skf = _make_skf()
    f1_scores, roc_scores = [], []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= threshold).astype(int)

        f1_scores.append(f1_score(y_val, preds))
        roc_scores.append(roc_auc_score(y_val, probs))

    logger.info("F1  mean: %.4f  std: %.4f", np.mean(f1_scores), np.std(f1_scores))
    logger.info("ROC mean: %.4f  std: %.4f", np.mean(roc_scores), np.std(roc_scores))


# ═══════════════════════════════════════════════════════════════
#  SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════

def save_outputs(df: pd.DataFrame, results: dict) -> None:
    """Save processed data and model results to CSV."""
    df.to_csv("churn_data_processed.csv", index=False)
    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    results_df.to_csv("model_results.csv", index=False)
    logger.info("Files saved: churn_data_processed.csv, model_results.csv")
