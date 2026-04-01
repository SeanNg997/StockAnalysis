#!/usr/bin/env bash
# run_daily.sh — A股量化策略每日一键运行脚本
#
# 用法:
#   ./scripts/run_daily.sh                # 增量更新数据 + 快速预测 + 生成全市场策略
#   ./scripts/run_daily.sh --full         # 增量更新数据 + 完整walk-forward训练 + 回测 + 生成策略和图表
#   ./scripts/run_daily.sh 600000         # 增量更新数据 + 快速预测 + 生成单股票策略报告
#   ./scripts/run_daily.sh --full 000001   # 完整模式 + 单股票策略报告
#
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"

# 解析参数：--full 和/或 股票代码
FULL_MODE=false
STOCK_CODE=""

for arg in "$@"; do
    case "$arg" in
        --full)
            FULL_MODE=true
            ;;
        *)
            STOCK_CODE="$arg"
            ;;
    esac
done

echo "=============================================="
echo "  A股量化策略 — 每日运行 $(date +%Y-%m-%d)"
if [ -n "$STOCK_CODE" ]; then
    echo "  目标股票: $STOCK_CODE"
fi
echo "=============================================="

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
if [ "$FULL_MODE" = true ]; then
    echo "[Step 4/4] 完整walk-forward训练 + 回测 + 策略生成..."
    python "$SRC_DIR/py03_model.py"
    python "$SRC_DIR/py04_backtest.py"
else
    echo "[Step 4/4] 快速预测 + 策略生成..."
    python "$SRC_DIR/py07_quick_predict.py"
fi

# 生成策略报告
if [ -n "$STOCK_CODE" ]; then
    python "$SRC_DIR/py05_today.py" "$STOCK_CODE"
else
    python "$SRC_DIR/py05_today.py"
    if [ "$FULL_MODE" = true ]; then
        python "$SRC_DIR/py06_report.py"
    fi
fi

echo ""
echo "=============================================="
if [ -n "$STOCK_CODE" ]; then
    # 提取6位数字代码用于显示文件名
    SHORT_CODE=$(echo "$STOCK_CODE" | grep -oE '[0-9]{6}')
    echo "  完成! 策略报告: output/today_strategy_${SHORT_CODE}.md"
else
    echo "  完成! 策略报告: output/today_strategy.md"
fi
echo "=============================================="
