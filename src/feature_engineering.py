"""
特征分析与特征工程模块
- 多相关系数分析 (Pearson, Spearman, Kendall)
- 相关性热图
- 特征选择: 过滤式、包裹式(RFE)、嵌入式(Lasso/RF重要性)
- PCA 降维
- 交叉验证评估特征选择前后性能
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
    VarianceThreshold,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Correlation analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_correlations(df, target_col=None, methods=None):
    """Compute correlation matrix for multiple methods.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from method name to correlation matrix.
    """
    target_col = target_col or config.TARGET_COL
    methods = methods or config.CORRELATION_METHODS
    results = {}

    feature_cols = [c for c in config.FEATURE_COLS if c in df.columns]
    all_cols = [target_col] + feature_cols

    for method in methods:
        logger.info("Computing %s correlation...", method)
        corr = df[all_cols].corr(method=method)
        results[method] = corr
        # Print feature-target correlations sorted
        target_corr = corr[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        print(f"\n--- {method.upper()} Feature-Target Correlations (sorted) ---")
        print(target_corr.round(4).to_string())

    return results


def detect_high_correlations(corr_matrix, threshold=None):
    """Detect and warn about highly correlated feature pairs."""
    threshold = threshold or config.CORRELATION_THRESHOLD
    high_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
    if high_pairs:
        logger.warning("High correlations detected (|r| > %.2f):", threshold)
        for c1, c2, v in sorted(high_pairs, key=lambda x: -abs(x[2])):
            logger.warning("  %s ↔ %s : %.4f", c1, c2, v)
    return high_pairs


def plot_correlation_heatmap(corr_matrix, method_name, save_path=None):
    """Plot a correlation heatmap."""
    save_path = save_path or os.path.join(
        config.FIGURES_DIR, f"correlation_heatmap_{method_name}.png"
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Correlation Heatmap – {method_name.capitalize()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Correlation heatmap saved → %s", save_path)


# ═══════════════════════════════════════════════════════════════════════════
# Feature selection
# ═══════════════════════════════════════════════════════════════════════════

def _get_xy(df, target_col):
    target_col = target_col or config.TARGET_COL
    feature_cols = [c for c in config.FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def select_by_variance(df, target_col=None, threshold=None):
    """Filter features by variance threshold."""
    target_col = target_col or config.TARGET_COL
    threshold = threshold or config.VARIANCE_THRESHOLD
    X, y, feature_cols = _get_xy(df, target_col)

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    selected = selector.get_support()
    kept = [c for c, s in zip(feature_cols, selected) if s]
    removed = [c for c, s in zip(feature_cols, selected) if not s]
    logger.info("Variance threshold: kept %d, removed %d", len(kept), len(removed))
    logger.info("  Removed: %s", removed)
    return kept, removed, selector


def select_by_rfe(df, target_col=None, n_features=None, estimator="rf"):
    """Recursive Feature Elimination."""
    target_col = target_col or config.TARGET_COL
    n_features = n_features or config.N_FEATURES_SELECT
    X, y, feature_cols = _get_xy(df, target_col)

    if estimator == "rf":
        est = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    else:
        est = LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE)

    rfe = RFE(est, n_features_to_select=n_features)
    rfe.fit(X, y)

    ranking = pd.Series(rfe.ranking_, index=feature_cols).sort_values()
    print(f"\n--- RFE Feature Ranking (top {n_features} kept) ---")
    print(ranking.to_string())

    selected = [c for c, r in zip(feature_cols, rfe.support_) if r]
    return selected, ranking, rfe


def select_by_lasso(df, target_col=None):
    """L1-regularized (Lasso) feature selection."""
    target_col = target_col or config.TARGET_COL
    X, y, feature_cols = _get_xy(df, target_col)

    lasso = Lasso(alpha=0.01, random_state=config.RANDOM_STATE, max_iter=5000)
    lasso.fit(X, y)

    importance = pd.Series(np.abs(lasso.coef_), index=feature_cols).sort_values(ascending=False)
    print("\n--- Lasso Feature Importance ---")
    print(importance.round(6).to_string())

    selected = importance[importance > 0].index.tolist()
    logger.info("Lasso selected %d features", len(selected))
    return selected, importance, lasso


def select_by_rf_importance(df, target_col=None, n_features=None):
    """Random Forest feature importance selection."""
    target_col = target_col or config.TARGET_COL
    n_features = n_features or config.N_FEATURES_SELECT
    X, y, feature_cols = _get_xy(df, target_col)

    rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- Random Forest Feature Importance ---")
    print(importance.round(6).to_string())

    selected = importance.head(n_features).index.tolist()
    logger.info("RF importance: selected %s", selected)
    return selected, importance, rf


def run_feature_selection(df, method=None, target_col=None):
    """Run feature selection using the configured method.

    Returns
    -------
    selected_features : list
    """
    method = method or config.FEATURE_SELECTION
    target_col = target_col or config.TARGET_COL
    logger.info("Feature selection: method=%s", method)

    if method == "variance":
        selected, _, _ = select_by_variance(df, target_col)
    elif method == "rfe":
        selected, _, _ = select_by_rfe(df, target_col)
    elif method == "lasso":
        selected, _, _ = select_by_lasso(df, target_col)
    elif method == "rf_importance":
        selected, _, _ = select_by_rf_importance(df, target_col)
    elif method == "none":
        selected = [c for c in config.FEATURE_COLS if c in df.columns]
    else:
        raise ValueError(f"Unknown feature selection: {method}")

    # Save selected features
    output_path = os.path.join(config.REPORTS_DIR, "selected_features.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(selected))
    logger.info("Selected features saved → %s", output_path)

    return selected


# ═══════════════════════════════════════════════════════════════════════════
# PCA
# ═══════════════════════════════════════════════════════════════════════════

def run_pca(df, target_col=None, n_components=None):
    """Run PCA and return transformed data + explained variance."""
    target_col = target_col or config.TARGET_COL
    n_components = n_components or config.PCA_N_COMPONENTS

    if n_components <= 0:
        logger.info("PCA skipped (n_components <= 0)")
        return df, None, None

    X, y, feature_cols = _get_xy(df, target_col)

    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained)
    logger.info("PCA: %d components", n_components)
    for i, (ev, cv) in enumerate(zip(explained, cumsum)):
        logger.info("  PC%d: %.4f (cumulative: %.4f)", i + 1, ev, cv)

    # Plot explained variance
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(explained) + 1), explained, alpha=0.7, label="Individual")
    ax.plot(range(1, len(cumsum) + 1), cumsum, "ro-", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(config.FIGURES_DIR, "pca_variance.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    # Build DataFrame with PCs
    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    df_pca[target_col] = y.values

    return df_pca, pca, explained


# ═══════════════════════════════════════════════════════════════════════════
# Cross-validation feature evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_feature_selection(df_before, df_after, target_col=None, cv=None):
    """Compare model performance before and after feature selection using cross-validation."""
    target_col = target_col or config.TARGET_COL
    cv = cv or config.CV_FOLDS
    logger.info("Evaluating feature selection with %d-fold CV...", cv)

    model = LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE)

    X_before = df_before[[c for c in config.FEATURE_COLS if c in df_before.columns]]
    y_before = df_before[target_col]
    scores_before = cross_val_score(model, X_before, y_before, cv=cv, scoring="roc_auc")

    features_after = [c for c in df_after.columns if c != target_col]
    X_after = df_after[features_after]
    y_after = df_after[target_col]
    scores_after = cross_val_score(model, X_after, y_after, cv=cv, scoring="roc_auc")

    print("\n--- Feature Selection Cross-Validation Comparison ---")
    print(f"  Before ({X_before.shape[1]} features): AUC = {scores_before.mean():.4f} (±{scores_before.std():.4f})")
    print(f"  After  ({X_after.shape[1]} features): AUC = {scores_after.mean():.4f} (±{scores_after.std():.4f})")

    return scores_before, scores_after


# ═══════════════════════════════════════════════════════════════════════════
# Full feature engineering pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_feature_engineering(df, corr_methods=None, selection_method=None,
                            pca_n_components=None, target_col=None):
    """Run the full feature engineering pipeline.

    Returns
    -------
    df_selected : pd.DataFrame
        DataFrame with selected features + target.
    selected_features : list
    """
    logger.info("=" * 50)
    logger.info("Feature Engineering Pipeline Start")
    logger.info("=" * 50)

    # Step 1: Correlation analysis
    correlations = compute_correlations(df, target_col, methods=corr_methods)
    for method_name, corr_mat in correlations.items():
        plot_correlation_heatmap(corr_mat, method_name)
        if method_name == "pearson":
            detect_high_correlations(corr_mat)

    # Save feature-target correlations
    pearson_corr = correlations.get("pearson", list(correlations.values())[0])
    target_corr = pearson_corr[config.TARGET_COL].drop(config.TARGET_COL)
    target_corr.to_csv(os.path.join(config.REPORTS_DIR, "feature_target_corr.csv"), header=["correlation"])

    # Step 2: Feature selection
    selected = run_feature_selection(df, method=selection_method, target_col=target_col)
    df_selected = df[selected + [config.TARGET_COL]].copy()

    # Step 3: PCA (optional comparison)
    if pca_n_components is None:
        pca_n_components = config.PCA_N_COMPONENTS
    if pca_n_components > 0:
        df_pca, pca_obj, _ = run_pca(df, target_col, pca_n_components)

    # Step 4: Evaluate feature selection
    evaluate_feature_selection(df, df_selected, target_col)

    logger.info("Feature Engineering Pipeline Complete")
    return df_selected, selected
