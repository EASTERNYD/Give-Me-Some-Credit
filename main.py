#!/usr/bin/env python
"""
信用卡违约分析实验系统 — 主执行脚本
======================================
按照 config.py 中的配置运行全部流程:
  1. 数据加载与预览 (创新点1)
  2. 数据预处理 (创新点2)
  3. 特征工程与选择 (创新点3+4)
  4. 多模型训练 (创新点5)
  5. 模型评估 (创新点6)
  6. 数据可视化 (创新点7)

Usage:
    python main.py                    # 完整流程
    python main.py --skip-profiling   # 跳过 ydata-profiling 报告
    python main.py --models lr,rf     # 只训练指定模型
"""
import argparse
import logging
import os
import sys
import time

import pandas as pd

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src import eda
from src import data_preprocessing
from src import feature_engineering
from src import model_training
from src import model_evaluation
from src import visualization


def setup_logging():
    """Configure logging to both file and console."""
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Credit Default Analysis System")
    parser.add_argument(
        "--skip-profiling", action="store_true",
        help="Skip ydata-profiling HTML report generation",
    )
    parser.add_argument(
        "--models", type=str, default=",".join(config.MODELS),
        help=f"Comma-separated model keys to train. Options: lr, rf, svm, nb. Default: {','.join(config.MODELS)}",
    )
    parser.add_argument(
        "--imputation", type=str, default=config.IMPUTATION_METHOD,
        help="Imputation method: rf, knn, iterative, median",
    )
    parser.add_argument(
        "--outlier", type=str, default=config.OUTLIER_METHOD,
        help="Outlier detection method: iqr, zscore, lof, none",
    )
    parser.add_argument(
        "--feature-selection", type=str, default=config.FEATURE_SELECTION,
        help="Feature selection method: variance, rfe, lasso, rf_importance, none",
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip hyperparameter tuning for faster run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Credit Default Analysis System – Starting")
    logger.info("=" * 60)

    # Resolve model keys
    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    logger.info("Configuration: models=%s, imputation=%s, outlier=%s, feature_selection=%s",
                model_keys, args.imputation, args.outlier, args.feature_selection)

    try:
        # ═════════════════════════════════════════════════════════════════
        # Step 1: Data Loading & EDA (创新点1)
        # ═════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Data Loading & EDA")
        logger.info("=" * 60)

        df = eda.load_data()
        eda.run_eda(df, skip_profiling=args.skip_profiling)

        # ═════════════════════════════════════════════════════════════════
        # Step 2: Data Preprocessing (创新点2)
        # ═════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Data Preprocessing")
        logger.info("=" * 60)

        df_scaled, df_clean = data_preprocessing.run_preprocessing(
            df,
            imputation_method=args.imputation,
            outlier_method=args.outlier,
        )

        # ═════════════════════════════════════════════════════════════════
        # Step 3: Feature Engineering (创新点3+4)
        # ═════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Feature Engineering")
        logger.info("=" * 60)

        df_selected, selected_features = feature_engineering.run_feature_engineering(
            df_scaled,
            selection_method=args.feature_selection,
        )

        # ═════════════════════════════════════════════════════════════════
        # Step 4: Model Training (创新点5)
        # ═════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Model Training")
        logger.info("=" * 60)

        trained_models, (X_train, X_test, y_train, y_test) = model_training.train_models(
            df_selected,
            model_keys=model_keys,
            tune=not args.no_tune,
        )

        if not trained_models:
            logger.error("No models were successfully trained. Exiting.")
            sys.exit(1)

        # ═════════════════════════════════════════════════════════════════
        # Step 5: Model Evaluation (创新点6)
        # ═════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Model Evaluation")
        logger.info("=" * 60)

        comparison, all_metrics = model_evaluation.run_evaluation(
            trained_models, X_test, y_test
        )

        # ═════════════════════════════════════════════════════════════════
        # Step 6: Visualization (创新点7)
        # ═════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Visualization")
        logger.info("=" * 60)

        # Feature importance from Random Forest (if available)
        rf_model = trained_models.get("rf")
        if rf_model and hasattr(rf_model, "feature_importances_"):
            feat_cols = [c for c in df_selected.columns if c != config.TARGET_COL]
            importances = pd.Series(rf_model.feature_importances_, index=feat_cols).sort_values(ascending=False)
            visualization.plot_feature_importance(importances)

        visualization.run_visualization(
            df_clean,
            trained_models=trained_models,
            X_test=X_test,
            y_test=y_test,
        )

        # ═════════════════════════════════════════════════════════════════
        # Summary
        # ═════════════════════════════════════════════════════════════════
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("ALL STEPS COMPLETE")
        logger.info(f"Total elapsed time: {elapsed:.2f}s")
        logger.info("=" * 60)

        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Log file: {config.LOG_FILE}")
        print(f"  Figures: {config.FIGURES_DIR}")
        print(f"  Reports: {config.REPORTS_DIR}")
        print(f"  Models: {config.MODELS_DIR}")
        print(f"{'='*60}")

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
