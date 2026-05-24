"""
数据预处理模块 (创新点2)
- 多种缺失值补齐方法 (KNN, Iterative, RF, 中位数/众数)
- 异常值检测与处理 (IQR, Z-score, LOF)
- 数据标准化 (StandardScaler, MinMaxScaler)
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
try:
    from sklearn.impute import IterativeImputer
except ImportError:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats as scipy_stats
import joblib

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Missing value imputation
# ═══════════════════════════════════════════════════════════════════════════

def _impute_column_rf(df, target_col, cat_cols=None):
    """Impute missing values using Random Forest (for continuous cols)."""
    cat_cols = cat_cols or []
    df_result = df.copy()
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isna().sum() == 0:
            continue
        logger.info("  RF imputing: %s", col)
        # Use complete rows as training data
        complete_mask = df[col].notna()
        features = [c for c in df.columns if c != col]
        X_train = pd.get_dummies(df.loc[complete_mask, features], drop_first=True)
        y_train = df.loc[complete_mask, col]
        X_pred = pd.get_dummies(df.loc[~complete_mask, features], drop_first=True)
        # Align columns
        X_train, X_pred = X_train.align(X_pred, join="left", axis=1, fill_value=0)

        if col in cat_cols:
            model = RandomForestClassifier(
                n_estimators=config.RF_IMPUTE_N_ESTIMATORS,
                random_state=config.RANDOM_STATE,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=config.RF_IMPUTE_N_ESTIMATORS,
                random_state=config.RANDOM_STATE,
            )
        model.fit(X_train, y_train)
        predicted = model.predict(X_pred)
        df_result.loc[~complete_mask, col] = predicted
    return df_result


def impute_missing_values(df, method=None, target_col=None, cat_cols=None):
    """Impute missing values using the specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with missing values.
    method : str
        "rf", "knn", "iterative", or "median".
    target_col : str
        Target column name (excluded from imputation).
    cat_cols : list
        Categorical/discrete columns for which RF classification is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    method = method or config.IMPUTATION_METHOD
    target_col = target_col or config.TARGET_COL
    cat_cols = cat_cols or ["NumberOfDependents"]

    logger.info("Imputing missing values using method: %s", method)

    df_imputed = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    if method == "rf":
        df_imputed = _impute_column_rf(df_imputed, target_col, cat_cols)

    elif method == "knn":
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_imputed[feature_cols])
        imputer = KNNImputer(n_neighbors=config.KNN_NEIGHBORS)
        scaled_imputed = imputer.fit_transform(scaled)
        scaled_imputed = scaler.inverse_transform(scaled_imputed)
        df_imputed[feature_cols] = scaled_imputed

    elif method == "iterative":
        imputer = IterativeImputer(
            random_state=config.RANDOM_STATE, max_iter=20
        )
        df_imputed[feature_cols] = imputer.fit_transform(df_imputed[feature_cols])

    elif method == "median":
        num_cols = df_imputed[feature_cols].select_dtypes(include=[np.number]).columns
        imp_num = SimpleImputer(strategy="median")
        df_imputed[num_cols] = imp_num.fit_transform(df_imputed[num_cols])

    else:
        raise ValueError(f"Unknown imputation method: {method}")

    logger.info("Remaining missing values: %d", df_imputed.isna().sum().sum())
    return df_imputed


def plot_imputation_comparison(df_original, df_imputed, col, save_path=None):
    """Plot distribution comparison before/after imputation."""
    save_dir = config.FIGURES_DIR
    save_path = save_path or os.path.join(save_dir, f"impute_{col}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Before (drop NA)
    df_original[col].dropna().hist(bins=40, ax=axes[0], alpha=0.7, color="steelblue")
    axes[0].set_title(f"{col} – Before Imputation (non-null)")
    axes[0].set_xlabel(col)
    # After
    df_imputed[col].hist(bins=40, ax=axes[1], alpha=0.7, color="darkorange")
    axes[1].set_title(f"{col} – After Imputation")
    axes[1].set_xlabel(col)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Imputation comparison saved → %s", save_path)


# ═══════════════════════════════════════════════════════════════════════════
# Outlier detection and handling
# ═══════════════════════════════════════════════════════════════════════════

def _get_numeric_cols(df, target_col):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]


def detect_outliers_iqr(df, target_col=None, multiplier=None):
    """Detect outliers using IQR method. Returns boolean mask."""
    target_col = target_col or config.TARGET_COL
    multiplier = multiplier or config.IQR_MULTIPLIER
    numeric_cols = _get_numeric_cols(df, target_col)

    outlier_mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        col_mask = (df[col] < lower) | (df[col] > upper)
        outlier_mask = outlier_mask | col_mask
        n_out = col_mask.sum()
        if n_out > 0:
            logger.info("  IQR outliers in %s: %d (%.2f%%)", col, n_out, n_out / len(df) * 100)
    return outlier_mask


def detect_outliers_zscore(df, target_col=None, threshold=None):
    """Detect outliers using Z-score method."""
    target_col = target_col or config.TARGET_COL
    threshold = threshold or config.ZSCORE_THRESHOLD
    numeric_cols = _get_numeric_cols(df, target_col)

    outlier_mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        z = np.abs(scipy_stats.zscore(df[col].dropna()))
        # Align back
        col_outlier = pd.Series(False, index=df.index)
        valid_idx = df[col].dropna().index
        col_outlier.loc[valid_idx] = z > threshold
        outlier_mask = outlier_mask | col_outlier
        n_out = col_outlier.sum()
        if n_out > 0:
            logger.info("  Z-score outliers in %s: %d (%.2f%%)", col, n_out, n_out / len(df) * 100)
    return outlier_mask


def detect_outliers_lof(df, target_col=None, n_neighbors=None, contamination=None):
    """Detect outliers using Local Outlier Factor."""
    target_col = target_col or config.TARGET_COL
    n_neighbors = n_neighbors or config.LOF_NEIGHBORS
    contamination = contamination or config.LOF_CONTAMINATION
    numeric_cols = _get_numeric_cols(df, target_col)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols].dropna())
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(scaled)

    outlier_mask = pd.Series(False, index=df.index)
    outlier_mask.loc[df[numeric_cols].dropna().index] = labels == -1
    n_out = outlier_mask.sum()
    logger.info("  LOF outliers detected: %d (%.2f%%)", n_out, n_out / len(df) * 100)
    return outlier_mask


def detect_outliers(df, method=None, target_col=None):
    """Detect outliers using the specified method.

    Returns
    -------
    pd.Series
        Boolean mask where True = outlier.
    """
    method = method or config.OUTLIER_METHOD
    target_col = target_col or config.TARGET_COL
    logger.info("Detecting outliers: method=%s", method)

    if method == "iqr":
        return detect_outliers_iqr(df, target_col)
    elif method == "zscore":
        return detect_outliers_zscore(df, target_col)
    elif method == "lof":
        return detect_outliers_lof(df, target_col)
    elif method == "none":
        return pd.Series(False, index=df.index)
    else:
        raise ValueError(f"Unknown outlier method: {method}")


def handle_outliers(df, outlier_mask, handling=None, target_col=None):
    """Handle detected outliers.

    Parameters
    ----------
    handling : str
        "remove", "clip", or "impute".
    """
    handling = handling or config.OUTLIER_HANDLING
    target_col = target_col or config.TARGET_COL
    n_out = outlier_mask.sum()
    logger.info("Handling %d outliers: strategy=%s", n_out, handling)

    if handling == "remove":
        return df[~outlier_mask].copy()

    elif handling == "clip":
        df_result = df.copy()
        numeric_cols = _get_numeric_cols(df_result, target_col)
        for col in numeric_cols:
            Q1 = df_result[col].quantile(0.25)
            Q3 = df_result[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - config.IQR_MULTIPLIER * IQR
            upper = Q3 + config.IQR_MULTIPLIER * IQR
            df_result[col] = df_result[col].clip(lower, upper)
        return df_result

    elif handling == "impute":
        df_result = df.copy()
        numeric_cols = _get_numeric_cols(df_result, target_col)
        for col in numeric_cols:
            df_result.loc[outlier_mask, col] = np.nan
        return impute_missing_values(df_result, method=config.IMPUTATION_METHOD, target_col=target_col)

    else:
        raise ValueError(f"Unknown outlier handling: {handling}")


# ═══════════════════════════════════════════════════════════════════════════
# Scaling
# ═══════════════════════════════════════════════════════════════════════════

def scale_features(df, target_col=None, scaling=None):
    """Scale numeric features."""
    scaling = scaling or config.SCALING
    target_col = target_col or config.TARGET_COL
    logger.info("Scaling features: method=%s", scaling)

    if scaling == "none":
        return df

    df_scaled = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
    # Save scaler for later use
    scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved → %s", scaler_path)
    return df_scaled


# ═══════════════════════════════════════════════════════════════════════════
# Full preprocessing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_preprocessing(df, imputation_method=None, outlier_method=None,
                      outlier_handling=None, scaling=None):
    """Run the full preprocessing pipeline.

    Returns
    -------
    df_cleaned : pd.DataFrame
    """
    logger.info("=" * 50)
    logger.info("Preprocessing Pipeline Start")
    logger.info("=" * 50)

    # Step 1: Impute missing values
    df_imputed = impute_missing_values(df, method=imputation_method)

    # Plot comparison for columns that had missing values
    for col in config.MISSING_COLS:
        if col in df.columns and df[col].isna().sum() > 0:
            plot_imputation_comparison(df, df_imputed, col)

    # Step 2: Detect outliers
    outlier_mask = detect_outliers(df_imputed, method=outlier_method)

    # Step 3: Handle outliers
    df_clean = handle_outliers(df_imputed, outlier_mask, handling=outlier_handling)
    logger.info("After outlier handling: %d rows", len(df_clean))

    # Step 4: Save cleaned data
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    df_clean.to_csv(config.CLEANED_FILE, index=False)
    logger.info("Cleaned data saved → %s", config.CLEANED_FILE)

    # Step 5: Scale (return both scaled and unscaled; let caller decide)
    df_scaled = scale_features(df_clean, scaling=scaling)

    logger.info("Preprocessing Pipeline Complete")
    return df_scaled, df_clean
