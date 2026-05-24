"""
模型验证与评估模块 (创新点6)
- 混淆矩阵、Accuracy、Precision、Recall、F1-score
- ROC 曲线及 AUC
- PR 曲线 (适用于类别不平衡)
- Log Loss
- 多模型评估对比报告
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    log_loss,
    classification_report,
)

import config

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "svm": "SVM",
    "nb": "Naive Bayes",
}


def evaluate_model(model, X_test, y_test, model_name=None):
    """Compute all evaluation metrics for a single model.

    Returns
    -------
    dict
        Metric name → value.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics["confusion_matrix"] = cm
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_test, y_pred, zero_division=0)

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        metrics["avg_precision"] = average_precision_score(y_test, y_proba)
        metrics["log_loss"] = log_loss(y_test, y_proba)
        metrics["y_proba"] = y_proba
    else:
        metrics["roc_auc"] = None
        metrics["avg_precision"] = None
        metrics["log_loss"] = None
        metrics["y_proba"] = None

    metrics["y_pred"] = y_pred

    # Print summary
    name = model_name or "Model"
    display_name = MODEL_DISPLAY_NAMES.get(model_name, name)
    print(f"\n{'='*50}")
    print(f"  {display_name} – Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-score:    {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return metrics


def evaluate_all_models(trained_models, X_test, y_test):
    """Evaluate all trained models and produce a comparison report.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    logger.info("=" * 50)
    logger.info("Model Evaluation Pipeline Start")
    logger.info("=" * 50)

    all_metrics = {}
    for key, model in trained_models.items():
        try:
            metrics = evaluate_model(model, X_test, y_test, model_name=key)
            all_metrics[key] = metrics
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", key, e, exc_info=True)

    # Build comparison table
    rows = []
    for key, m in all_metrics.items():
        rows.append({
            "Model": MODEL_DISPLAY_NAMES.get(key, key),
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-score": round(m["f1"], 4),
            "ROC AUC": round(m.get("roc_auc") or 0, 4),
            "Avg Precision": round(m.get("avg_precision") or 0, 4),
            "Log Loss": round(m.get("log_loss") or 0, 4),
        })

    comparison = pd.DataFrame(rows).set_index("Model")
    print(f"\n{'='*70}")
    print("Model Comparison Report")
    print(f"{'='*70}")
    print(comparison.to_string())

    # Save to CSV
    csv_path = os.path.join(config.REPORTS_DIR, "model_comparison.csv")
    comparison.to_csv(csv_path)
    logger.info("Comparison report saved → %s", csv_path)

    return comparison, all_metrics


def plot_roc_curves(trained_models, X_test, y_test, save_path=None):
    """Plot multi-model ROC curve comparison."""
    save_path = save_path or os.path.join(config.FIGURES_DIR, "roc_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trained_models)))

    for (key, model), color in zip(trained_models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            name = MODEL_DISPLAY_NAMES.get(key, key)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – Model Comparison", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved → %s", save_path)


def plot_confusion_matrices(trained_models, X_test, y_test, save_path=None):
    """Plot confusion matrix heatmaps for each model."""
    save_path = save_path or os.path.join(config.FIGURES_DIR, "confusion_matrices.png")

    n = len(trained_models)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (key, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        name = MODEL_DISPLAY_NAMES.get(key, key)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for j in range(len(trained_models), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrices saved → %s", save_path)


def plot_pr_curves(trained_models, X_test, y_test, save_path=None):
    """Plot multi-model Precision-Recall curves."""
    save_path = save_path or os.path.join(config.FIGURES_DIR, "pr_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trained_models)))

    for (key, model), color in zip(trained_models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            name = MODEL_DISPLAY_NAMES.get(key, key)
            ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves – Model Comparison", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("PR curves saved → %s", save_path)


def run_evaluation(trained_models, X_test, y_test):
    """Run the full evaluation pipeline."""
    logger.info("=" * 50)
    logger.info("Evaluation Pipeline Start")
    logger.info("=" * 50)

    comparison, all_metrics = evaluate_all_models(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_pr_curves(trained_models, X_test, y_test)

    logger.info("Evaluation Pipeline Complete")
    return comparison, all_metrics
