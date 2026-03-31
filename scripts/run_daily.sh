#!/usr/bin/env bash
# run_daily.sh — A股量化策略每日一键运行脚本
#
# 用法:
#   ./scripts/run_daily.sh          # 增量更新数据 + 快速预测 + 生成策略
#   ./scripts/run_daily.sh --full   # 增量更新数据 + 完整walk-forward训练 + 回测 + 生成策略和图表
#
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"

echo "=============================================="
echo "  A股量化策略 — 每日运行 $(date +%Y-%m-%d)"
echo "=============================================="

MODE="${1:-quick}"

# Step 1: 增量更新数据
echo ""
echo "[Step 1/4] 增量更新数据..."
python "$SRC_DIR/py00_fetch_stock_data.py" --update

# Step 2: 数据清洗
echo ""
echo "[Step 2/4] 数据清洗..."
python "$SRC_DIR/py01_data_loader.py"

# Step 3: 特征工程
echo ""
echo "[Step 3/4] 特征工程..."
python "$SRC_DIR/py02_features.py"

# Step 4: 模型预测 + 生成策略
echo ""
if [ "$MODE" = "--full" ]; then
    echo "[Step 4/4] 完整walk-forward训练 + 回测 + 策略生成..."
    python "$SRC_DIR/py03_model.py"
    python "$SRC_DIR/py04_backtest.py"
    python "$SRC_DIR/py05_today.py"
    python "$SRC_DIR/py06_report.py"
else
    echo "[Step 4/4] 快速预测 + 策略生成..."
    python "$SRC_DIR/py07_quick_predict.py"
    python "$SRC_DIR/py05_today.py"
fi

echo ""
echo "=============================================="
echo "  完成! 策略报告: output/today_strategy.txt"
echo "=============================================="
cat "$PROJECT_DIR/output/today_strategy.txt"
