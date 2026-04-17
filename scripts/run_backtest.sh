#!/usr/bin/env bash
# run_backtest.sh — A股量化策略完整 walk-forward 训练与回测脚本
#
# 用法:
#   ./scripts/run_backtest.sh                    # 完整训练（全量 walk-forward + 回测）
#
# 注：此脚本用于模型完整训练和性能评估，耗时较长（90 分钟）
#    日常预测使用 ./scripts/run_strategy.sh （快速单日预测，~30秒）
#
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"

TODAY=$(date +%Y-%m-%d)
START_TIME=$(date +%s)

echo "=============================================="
echo "  A股量化策略 — 完整 Walk-Forward 训练"
echo "  执行日期: $TODAY"
echo "=============================================="

# Step 1: 增量更新数据
echo ""
echo "[Step 1/7] 增量更新数据..."
python "$SRC_DIR/py00_fetch_stock_data.py"

# Step 2: 数据清洗（全量）
echo ""
echo "[Step 2/7] 数据清洗..."
python "$SRC_DIR/py01_data_clean.py"

# Step 3: 特征工程（全量）
echo ""
echo "[Step 3/7] 特征工程..."
python "$SRC_DIR/py02_features.py"

# Step 4: 全量 walk-forward 训练 + 预测
echo ""
echo "[Step 4/7] Walk-Forward 训练与预测..."
python "$SRC_DIR/py03_model.py"

# Step 5: 回测报告
echo ""
echo "[Step 5/7] 生成回测报告..."
python "$SRC_DIR/py05_backtest.py"

# Step 6: 可视化图表
echo ""
echo "[Step 6/7] 生成可视化图表..."
python "$SRC_DIR/py06_report.py"

# Step 7: 生成策略报告（含历史归档）
echo ""
echo "[Step 7/7] 生成今日交易策略报告..."
python "$SRC_DIR/py04_today.py"

echo ""
echo "=============================================="
echo "  完成! 训练模型与回测结果已保存"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))
echo "  总耗时: ${MINUTES} 分 ${SECONDS} 秒"
echo "=============================================="
