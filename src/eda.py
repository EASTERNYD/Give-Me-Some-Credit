"""
数据加载与整体预览模块
- ydata-profiling 交互式报告
- 统计量计算 (分位数、偏度、峰度)
- 目标变量频数分析
- missingno 缺失矩阵
- 箱线图集合
"""
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

import config

logger = logging.getLogger(__name__)


def load_data(path=None, max_samples=None):
    """Load the credit default dataset."""
    path = path or config.TRAIN_FILE
    max_samples = max_samples or config.MAX_SAMPLES
    logger.info("Loading data from %s (max %d samples)", path, max_samples)
    df = pd.read_csv(path, nrows=max_samples)
    # Drop unnamed index column if present
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def generate_profile_report(df, output_path=None):
    """Generate interactive ydata-profiling HTML report."""
    output_path = output_path or config.PROFILE_REPORT
    logger.info("Generating profile report → %s", output_path)
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, title="Credit Default Data Profile", explorative=True)
        profile.to_file(output_path)
        logger.info("Profile report saved.")
    except ImportError:
        logger.warning("ydata-profiling not installed; skipping profile report.")


def compute_statistics(df, feature_cols=None):
    """Compute extended statistics: quantiles, skewness, kurtosis for numeric columns."""
    feature_cols = feature_cols or config.FEATURE_COLS
    logger.info("Computing extended statistics for %d features.", len(feature_cols))

    stats_df = pd.DataFrame(index=feature_cols)
    for col in feature_cols:
        if col in df.columns:
            series = df[col].dropna()
            stats_df.loc[col, "count"] = len(series)
            stats_df.loc[col, "mean"] = series.mean()
            stats_df.loc[col, "std"] = series.std()
            stats_df.loc[col, "min"] = series.min()
            stats_df.loc[col, "q25"] = series.quantile(0.25)
            stats_df.loc[col, "median"] = series.quantile(0.50)
            stats_df.loc[col, "q75"] = series.quantile(0.75)
            stats_df.loc[col, "max"] = series.max()
            stats_df.loc[col, "skewness"] = series.skew()
            stats_df.loc[col, "kurtosis"] = series.kurtosis()
            stats_df.loc[col, "missing"] = df[col].isna().sum()
            stats_df.loc[col, "missing_pct"] = df[col].isna().mean() * 100

    print("\n" + "=" * 70)
    print("Extended Statistics Summary")
    print("=" * 70)
    print(stats_df.round(4).to_string())
    return stats_df


def analyze_target(df, target_col=None):
    """Analyze binary target variable: frequency, proportion."""
    target_col = target_col or config.TARGET_COL
    counts = df[target_col].value_counts()
    props = df[target_col].value_counts(normalize=True) * 100

    print("\n" + "=" * 70)
    print(f"Target Variable: {target_col}")
    print("=" * 70)
    for val in sorted(counts.index):
        print(f"  Class {val}: {counts[val]} samples ({props[val]:.2f}%)")

    imbalance_ratio = counts[0] / counts[1] if counts[1] > 0 else float("inf")
    print(f"  Imbalance ratio (0:1) = {imbalance_ratio:.2f}:1")
    return counts, props


def plot_missing_matrix(df, save_path=None):
    """Plot missing value matrix using missingno."""
    save_path = save_path or os.path.join(config.FIGURES_DIR, "missing_matrix.png")
    try:
        import missingno as msno
        fig, ax = plt.subplots(figsize=(12, 6))
        msno.matrix(df, ax=ax, sparkline=False)
        ax.set_title("Missing Value Matrix", fontsize=14)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Missing matrix saved → %s", save_path)
    except ImportError:
        # fallback with seaborn heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, ax=ax)
        ax.set_title("Missing Value Matrix (fallback)")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_boxplot_panel(df, feature_cols=None, save_path=None):
    """Plot a panel of boxplots for outlier detection."""
    feature_cols = feature_cols or config.FEATURE_COLS
    save_path = save_path or os.path.join(config.FIGURES_DIR, "boxplot_panel.png")
    logger.info("Generating boxplot panel.")

    cols_present = [c for c in feature_cols if c in df.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(cols_present) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_present):
        df[col].dropna().plot.box(ax=axes[i], vert=True)
        axes[i].set_title(col, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Boxplots – Outlier Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Boxplot panel saved → %s", save_path)


def run_eda(df, skip_profiling=False):
    """Run the full EDA pipeline."""
    logger.info("=" * 50)
    logger.info("EDA Pipeline Start")
    logger.info("=" * 50)

    # Basic info
    print("\n>>> Data Info <<<")
    df.info()

    print("\n>>> Basic Describe <<<")
    print(df.describe().to_string())

    # Extended statistics
    stats = compute_statistics(df)

    # Target analysis
    analyze_target(df)

    # Missing matrix
    plot_missing_matrix(df)

    # Boxplot panel
    plot_boxplot_panel(df)

    # Profile report (can be slow for large data)
    if not skip_profiling:
        generate_profile_report(df)

    logger.info("EDA Pipeline Complete")
    return stats
