#!/usr/bin/env bash
# run_model.sh — A股量化策略完整 walk-forward 训练与回测脚本
#
# 用法:
#   ./scripts/run_model.sh                    # 完整训练（全量 walk-forward + 回测）
#
# 注：此脚本用于模型完整训练和性能评估，耗时较长（5-10 分钟）
#    日常预测使用 ./scripts/run_daily.sh （快速单日预测，~30秒）
#
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"

TODAY=$(date +%Y-%m-%d)

echo "=============================================="
echo "  A股量化策略 — 完整 Walk-Forward 训练"
echo "  执行日期: $TODAY"
echo "=============================================="

# Step 1: 增量更新数据
echo ""
echo "[Step 1/6] 增量更新数据..."
python "$SRC_DIR/py00_fetch_stock_data.py"

# Step 2: 数据清洗（全量）
echo ""
echo "[Step 2/6] 数据清洗..."
python "$SRC_DIR/py01_data_loader.py"

# Step 3: 特征工程（全量）
echo ""
echo "[Step 3/6] 特征工程..."
python "$SRC_DIR/py02_features.py"

# Step 4: 全量 walk-forward 训练 + 预测
echo ""
echo "[Step 4/6] Walk-Forward 训练与预测..."
python "$SRC_DIR/py03_model.py"

# Step 5: 回测报告
echo ""
echo "[Step 5/6] 生成回测报告..."
python "$SRC_DIR/py04_backtest.py"

# Step 6: 可视化图表
echo ""
echo "[Step 6/6] 生成可视化图表..."
python "$SRC_DIR/py06_report.py"

echo ""
echo "=============================================="
echo "  完成! 训练模型与回测结果已保存"
echo "=============================================="
