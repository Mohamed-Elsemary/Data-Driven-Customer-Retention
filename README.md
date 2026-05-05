# Data-Driven Customer Retention

End-to-end machine learning pipeline for predicting customer churn in a telecom company. Covers data validation, cleaning, EDA, feature engineering, model training (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM), and experiment tracking with MLflow.

## Dataset

Telco Customer Churn Dataset from Kaggle, downloaded automatically via `kagglehub`.

- Source: https://www.kaggle.com/datasets/abbas829/telco-customer-churn-dataset
- 7,043 customer records, 21 features
- Binary target: Churn (Yes/No)

## Project Structure

```
.
├── main.py                  # Pipeline orchestrator
├── config.py                # Constants, column lists, hyperparameter grids
├── data_loading.py          # Automated dataset download via kagglehub
├── data_validation.py       # 7-dimension data quality checks
├── data_cleaning.py         # Type conversion, binary mapping, null handling
├── eda.py                   # Univariate, bivariate, multivariate analysis
├── feature_engineering.py   # Encoding, derived features, feature selection
├── modeling.py              # Grid search, evaluation, MLflow logging
├── tests/
│   ├── conftest.py              # Shared fixtures (raw, cleaned, encoded, featured)
│   ├── test_data_cleaning.py    # Data cleaning unit tests
│   ├── test_feature_engineering.py  # Encoding and feature formula tests
│   ├── test_modeling.py         # Business metrics unit tests
│   └── test_integration.py      # End-to-end pipeline tests
├── Makefile                 # Task automation
├── pyproject.toml           # Dependencies and tool configuration
└── .github/workflows/ci.yml # CI/CD pipeline
```

## Requirements

- Python 3.11
- All dependencies listed in `pyproject.toml`

## Setup

```bash
pip install -e ".[dev]"
```

Or using the Makefile:

```bash
make setup
```

## Usage

Run the full pipeline (data download through model training):

```bash
make pipeline
```

Or directly:

```bash
python main.py
```

View MLflow experiment results:

```bash
mlflow ui
```

## Makefile Targets

| Target              | Description                                      |
|---------------------|--------------------------------------------------|
| `make setup`        | Install all dependencies                         |
| `make format`       | Auto-format code with Black                      |
| `make check`        | Check formatting without changes                 |
| `make lint`         | Run Flake8 linter                                |
| `make isort`        | Sort imports with isort                           |
| `make test`         | Run full test suite (28 tests)                   |
| `make test-unit`    | Run unit tests only                              |
| `make test-integration` | Run integration tests only                   |
| `make pipeline`     | Run the full ML pipeline                         |
| `make clean`        | Delete generated files (cache, plots, logs, etc) |
| `make all`          | Run isort, format, check, lint, and test         |

## Tests

28 tests covering four areas:

- **Data cleaning** (6 tests): Binary mapping, type conversion, whitespace handling, non-destructive mutation
- **Feature engineering** (13 tests): Encoding correctness, derived feature formulas, train/test split integrity, frequency encoding leakage prevention
- **Business metrics** (3 tests): Revenue Saved and Revenue Leakage calculations against hand-computed values, zero-TP edge case
- **Integration** (6 tests): End-to-end pipeline, NaN non-propagation, numeric dtype enforcement, determinism, business metrics wiring

Run all tests:

```bash
make test
```

## CI/CD

GitHub Actions runs on every push/PR to `main`:

1. **Code Quality** -- isort check, Black format check, Flake8 lint
2. **Tests** -- unit and integration tests (runs only if step 1 passes)

## Pipeline Stages

1. **Data Loading** -- Download dataset from Kaggle via kagglehub
2. **Data Validation** -- Completeness, uniqueness, distribution, outlier checks
3. **Data Cleaning** -- Type conversion, binary mapping, null imputation
4. **EDA** -- Univariate, bivariate, and multivariate analysis with saved plots
5. **Feature Engineering** -- Encoding, 6 derived features, correlation-based and permutation-based feature selection
6. **Modeling** -- Grid search over 5 classifiers, evaluation, MLflow tracking

## Models

| Model               | Role                          |
|----------------------|-------------------------------|
| Logistic Regression  | Baseline                      |
| Decision Tree        | Interpretable non-linear      |
| Random Forest        | Ensemble, reduced overfitting |
| XGBoost              | Gradient boosting             |
| LightGBM             | Fast gradient boosting        |

All models are tuned with GridSearchCV (5-fold stratified CV, F1 scoring) and logged to MLflow with standard metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and business metrics (Revenue Saved, Revenue Leakage).