"""
数据可视化与结果展示模块
- 单变量: 年龄分布、月收入箱线图、逾期次数条形图 (按违约分组)
- 多变量: 相关性热图、平行坐标图、特征重要性条形图
- 模型评估: ROC曲线、混淆矩阵热图、预测概率分布直方图
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

logger = logging.getLogger(__name__)

# Plot style
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")


# ═══════════════════════════════════════════════════════════════════════════
# Univariate
# ═══════════════════════════════════════════════════════════════════════════

def plot_age_distribution(df, target_col=None, save_path=None):
    """Age distribution histogram/KDE grouped by default status."""
    target_col = target_col or config.TARGET_COL
    save_path = save_path or os.path.join(config.FIGURES_DIR, "age_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, label, color in zip(axes, [0, 1], ["#2ecc71", "#e74c3c"]):
        subset = df[df[target_col] == label]["age"]
        ax.hist(subset, bins=30, density=True, alpha=0.5, color=color, label=f"Class {label}")
        subset.plot.kde(ax=ax, color=color, lw=2)
        ax.set_title(f"Age Distribution – {'Non-default' if label == 0 else 'Default'}")
        ax.set_xlabel("Age")
        ax.legend()

    fig.suptitle("Age Distribution by Target Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Age distribution saved → %s", save_path)


def plot_income_boxplot(df, target_col=None, save_path=None):
    """Monthly income boxplot grouped by default status."""
    target_col = target_col or config.TARGET_COL
    save_path = save_path or os.path.join(config.FIGURES_DIR, "income_boxplot.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df[df["MonthlyIncome"] > 0].copy()
    df_plot["Status"] = df_plot[target_col].map({0: "Non-default", 1: "Default"})
    sns.boxplot(x="Status", y="MonthlyIncome", data=df_plot, hue="Status", palette=["#2ecc71", "#e74c3c"], legend=False, ax=ax)
    ax.set_title("Monthly Income by Default Status (excluding $0)", fontsize=14)
    ax.set_ylabel("Monthly Income")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Income boxplot saved → %s", save_path)


def plot_pastdue_barchart(df, target_col=None, save_path=None):
    """Bar chart for overdue count features, grouped by target."""
    target_col = target_col or config.TARGET_COL
    save_path = save_path or os.path.join(config.FIGURES_DIR, "pastdue_barchart.png")

    overdue_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, overdue_cols):
        if col not in df.columns:
            continue
        agg = df.groupby(target_col)[col].mean()
        ax.bar(["Non-default", "Default"], agg.values, color=["#2ecc71", "#e74c3c"])
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("Average Count")

    fig.suptitle("Average Overdue Counts by Default Status", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Past-due bar chart saved → %s", save_path)


# ═══════════════════════════════════════════════════════════════════════════
# Multivariate
# ═══════════════════════════════════════════════════════════════════════════

def plot_parallel_coordinates(df, target_col=None, feature_cols=None, save_path=None, max_samples=80):
    """Parallel coordinates with thin individual lines + bold class centroids."""
    target_col = target_col or config.TARGET_COL
    feature_cols = feature_cols or [c for c in config.FEATURE_COLS if c in df.columns]
    save_path = save_path or os.path.join(config.FIGURES_DIR, "parallel_coordinates.png")

    from sklearn.preprocessing import MinMaxScaler

    # Normalize all data to [0,1] for parallel coordinates
    scaler = MinMaxScaler()
    all_cols = feature_cols + [target_col]
    df_all = df[all_cols].dropna()

    # Sample a small set for faint individual lines
    n_per_class = min(max_samples, len(df_all))
    df_sample = pd.concat([
        df_all[df_all[target_col] == c].sample(n_per_class, random_state=config.RANDOM_STATE)
        for c in [0, 1]
    ])

    normalized = scaler.fit_transform(df_sample[feature_cols])
    df_norm = pd.DataFrame(normalized, columns=feature_cols)
    df_norm["Class"] = df_sample[target_col].values.astype(int)

    # Compute class centroids
    normalized_all = scaler.transform(df_all[feature_cols])
    df_all_norm = pd.DataFrame(normalized_all, columns=feature_cols)
    df_all_norm["Class"] = df_all[target_col].values.astype(int)
    means = df_all_norm.groupby("Class").mean()
    stds = df_all_norm.groupby("Class").std() * 0.5  # ±0.5 std band

    fig, ax = plt.subplots(figsize=(14, 6))

    # Faint individual lines
    colors = {0: "#2ecc71", 1: "#e74c3c"}
    labels = {0: "Non-default", 1: "Default"}
    for cls in [0, 1]:
        subset = df_norm[df_norm["Class"] == cls][feature_cols]
        for _, row in subset.iterrows():
            ax.plot(range(len(feature_cols)), row.values, color=colors[cls],
                    alpha=0.12, linewidth=0.6)

    # Bold centroid lines + band
    x = range(len(feature_cols))
    for cls in [0, 1]:
        mu = means.loc[cls].values
        sd = stds.loc[cls].values
        ax.fill_between(x, mu - sd, mu + sd, color=colors[cls], alpha=0.15)
        ax.plot(x, mu, color=colors[cls], linewidth=2.8, label=f"{labels[cls]} (mean)")

    ax.set_xticks(x)
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Value (0–1)", fontsize=11)
    ax.set_title("Parallel Coordinates – Default vs Non-Default (centroids ±0.5σ)", fontsize=14)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Parallel coordinates saved → %s", save_path)


def plot_feature_importance(importance_series, title="Feature Importance", save_path=None):
    """Bar chart of feature importance from tree-based model."""
    save_path = save_path or os.path.join(config.FIGURES_DIR, "feature_importance.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    importance_series.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance saved → %s", save_path)


# ═══════════════════════════════════════════════════════════════════════════
# Model evaluation – supplementary
# ═══════════════════════════════════════════════════════════════════════════

def plot_probability_distribution(trained_models, X_test, y_test, save_path=None):
    """Histogram of predicted probabilities by class to inspect calibration."""
    save_path = save_path or os.path.join(config.FIGURES_DIR, "probability_distribution.png")

    n = len(trained_models)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    from src.model_evaluation import MODEL_DISPLAY_NAMES

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (key, model) in zip(axes, trained_models.items()):
        if not hasattr(model, "predict_proba"):
            continue
        y_proba = model.predict_proba(X_test)[:, 1]
        name = MODEL_DISPLAY_NAMES.get(key, key)

        for cls, color, label in zip([0, 1], ["#2ecc71", "#e74c3c"], ["Non-default", "Default"]):
            subset = y_proba[y_test == cls]
            ax.hist(subset, bins=30, alpha=0.5, color=color, label=label, density=True)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted Probability")
        ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Predicted Probability Distribution by Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Probability distributions saved → %s", save_path)


# ═══════════════════════════════════════════════════════════════════════════
# Full visualization pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_visualization(df, trained_models=None, X_test=None, y_test=None):
    """Run comprehensive visualization pipeline."""
    logger.info("=" * 50)
    logger.info("Visualization Pipeline Start")
    logger.info("=" * 50)

    # Univariate
    plot_age_distribution(df)
    if "MonthlyIncome" in df.columns:
        plot_income_boxplot(df)
    plot_pastdue_barchart(df)

    # Multivariate
    plot_parallel_coordinates(df)

    # Model-related visualizations
    if trained_models and X_test is not None and y_test is not None:
        plot_probability_distribution(trained_models, X_test, y_test)

    logger.info("Visualization Pipeline Complete")
