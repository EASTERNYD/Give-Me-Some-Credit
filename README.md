# 信用卡违约分析

基于 Kaggle "Give Me Some Credit" 竞赛数据，构建的完整信用卡客户违约分析与预测系统。涵盖从数据加载、探索、预处理、特征工程、多模型训练到评估的全流程机器学习 

## 功能特性

- **数据预览** — `ydata-profiling` 交互式报告、`missingno` 缺失矩阵、箱线图面板、分位数/偏度/峰度统计
- **缺失值处理** — 支持 KNN / 迭代插补 / 随机森林 / 中位数 四种补齐方法，含填补前后分布对比
- **相关性分析** — Pearson / Spearman / Kendall 三种相关系数，热图可视化，高相关特征自动警告
- **特征选择与降维** — 方差阈值、RFE、Lasso、随机森林重要性 + PCA，含交叉验证评估
- **多模型训练** — 逻辑回归、随机森林、SVM + GridSearchCV 超参数调优，模型自动保存
- **多指标评估** — 混淆矩阵、Accuracy / Precision / Recall / F1、ROC/AUC、PR 曲线、Log Loss
- **丰富可视化** — 年龄分布、收入箱线、逾期条形、平行坐标、特征重要性、ROC/PR 曲线、概率分布

## 技术栈

| 类别 | 库 |
|------|-----|
| 数据处理 | pandas, numpy, scipy |
| 机器学习 | scikit-learn |
| 可视化 | matplotlib, seaborn, missingno |
| 数据报告 | ydata-profiling |
| 工具 | joblib, logging |

## 目录结构

```
credit_default_analysis/
├── data/
│   ├── raw/                    # 原始数据 cs-training.csv
│   └── processed/              # 预处理输出 cleaned_data.csv
├── notebooks/                  # Jupyter notebook（可选）
├── src/
│   ├── eda.py                  # 数据探索与预览
│   ├── data_preprocessing.py   # 缺失值补齐 + 异常值处理 + 标准化
│   ├── feature_engineering.py  # 相关性分析 + 特征选择 + PCA
│   ├── model_training.py       # 多模型训练 + 超参数调优
│   ├── model_evaluation.py     # 多指标评估 + 对比报告
│   └── visualization.py        # 6 类可视化图表
├── config.py                   # 集中配置（路径、参数、模型列表等）
├── main.py                     # 主执行脚本（一键运行全流程）
├── figures/                    # 输出图表
├── models/                     # 输出模型文件 (.pkl)
├── reports/                    # 输出报告 (.csv, .html, .txt)
├── requirements.txt
├── LICENSE
└── README.md
```

## 快速开始

### 环境要求

- Python 3.8+

### 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd credit_default_analysis

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

将 `cs-training.csv` 放入 `data/raw/` 目录。

### 运行

```bash
# 完整流程
python main.py

# 跳过 profiling 报告（更快）
python main.py --skip-profiling

# 自定义模型和参数
python main.py --models lr,rf --imputation knn --outlier iqr --feature-selection rf_importance

# 快速运行（跳过超参数调优）
python main.py --no-tune --skip-profiling
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--skip-profiling` | 跳过 ydata-profiling 报告 | 关闭 |
| `--models` | 模型列表 (lr, rf, svm, nb) | lr,rf,svm |
| `--imputation` | 缺失值方法 (knn, iterative, rf, median) | knn |
| `--outlier` | 异常值检测 (iqr, zscore, lof, none) | iqr |
| `--feature-selection` | 特征选择 (variance, rfe, lasso, rf_importance, none) | rf_importance |
| `--no-tune` | 跳过超参数调优 | 关闭 |

## 输出文件

| 文件 | 说明 |
|------|------|
| `data_profile.html` | ydata-profiling 数据概览报告 |
| `missing_matrix.png` | 缺失值矩阵图 |
| `boxplot_panel.png` | 特征箱线图面板 |
| `cleaned_data.csv` | 预处理后数据 |
| `correlation_heatmap_*.png` | Pearson/Spearman/Kendall 相关性热图 |
| `feature_target_corr.csv` | 特征-目标相关系数表 |
| `selected_features.txt` | 特征选择结果 |
| `pca_variance.png` | PCA 解释方差图 |
| `model_comparison.csv` | 各模型评估指标对比表 |
| `roc_curves.png` | 多模型 ROC 曲线对比 |
| `pr_curves.png` | 多模型 PR 曲线对比 |
| `confusion_matrices.png` | 各模型混淆矩阵热图 |
| `feature_importance.png` | 特征重要性条形图 |
| `age_distribution.png` | 年龄分布图 |
| `income_boxplot.png` | 收入箱线图 |
| `pastdue_barchart.png` | 逾期次数条形图 |
| `parallel_coordinates.png` | 平行坐标图 |
| `probability_distribution.png` | 预测概率分布图 |
| `impute_*.png` | 缺失值填补前后分布对比 |
| `experiment.log` | 实验日志 |

## AI 辅助开发说明

本项目在开发过程中使用了 AI 编程助手（Claude Code）辅助完成以下工作：

- 项目结构设计与模块划分
- 代码生成与调试
- 文档撰写

所有 AI 生成的代码均经过人工审查和测试验证，确保功能正确性和代码质量。

## 许可证

本项目基于 MIT License 开源，详见 [LICENSE](LICENSE)。
