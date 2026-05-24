"""
Central configuration for the credit default analysis system.
All paths, parameters, and model settings are defined here.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOG_FILE = os.path.join(BASE_DIR, "experiment.log")

TRAIN_FILE = os.path.join(DATA_RAW, "cs-training.csv")
CLEANED_FILE = os.path.join(DATA_PROCESSED, "cleaned_data.csv")
PROFILE_REPORT = os.path.join(REPORTS_DIR, "data_profile.html")

# ── Data ───────────────────────────────────────────────────────────────────
MAX_SAMPLES = 10000
TARGET_COL = "SeriousDlqin2yrs"
RANDOM_STATE = 42

# Columns with known missing values (from Data Dictionary)
MISSING_COLS = ["MonthlyIncome", "NumberOfDependents"]

# All feature columns (excluding target)
FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# ── Preprocessing ──────────────────────────────────────────────────────────
# Missing value imputation method: "rf" | "knn" | "iterative" | "median"
IMPUTATION_METHOD = "knn"
KNN_NEIGHBORS = 5
RF_IMPUTE_N_ESTIMATORS = 100

# Outlier detection: "iqr" | "zscore" | "lof" | "none"
OUTLIER_METHOD = "iqr"
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
LOF_NEIGHBORS = 20
LOF_CONTAMINATION = 0.05

# Outlier handling: "remove" | "clip" | "impute"
OUTLIER_HANDLING = "clip"

# Scaling: "standard" | "minmax" | "none"
SCALING = "standard"

# ── Feature Engineering ────────────────────────────────────────────────────
CORRELATION_METHODS = ["pearson", "spearman", "kendall"]
CORRELATION_THRESHOLD = 0.7  # warn if |r| > threshold
VARIANCE_THRESHOLD = 0.01

# Feature selection: "rfe" | "lasso" | "rf_importance" | "none"
FEATURE_SELECTION = "rf_importance"
N_FEATURES_SELECT = 6  # number of features to keep

# PCA
PCA_N_COMPONENTS = 5  # set to 0 to skip PCA

# ── Model Training ─────────────────────────────────────────────────────────
TEST_SIZE = 0.2
CV_FOLDS = 5

# Models to train: subset of ["lr", "rf", "svm", "nb"]
MODELS = ["lr", "rf", "svm"]

# Hyperparameter tuning
USE_GRID_SEARCH = True
GRID_SEARCH_CV = 3
GRID_SEARCH_SCORING = "roc_auc"

# Grid search param grids
PARAM_GRIDS = {
    "lr": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0, 1],
        "max_iter": [5000],
    },
    "rf": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced", None],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "class_weight": ["balanced", None],
        "probability": [True],
    },
    "nb": {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
    },
}

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
