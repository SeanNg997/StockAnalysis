#!/usr/bin/env bash
# run_daily.sh — A股量化策略每日预测脚本（快速单日预测，不含回测）
#
# 用法:
#   ./scripts/run_daily.sh                         # 最新日期全市场策略
#   ./scripts/run_daily.sh --date 2025-03-14       # 指定日期全市场策略
#   ./scripts/run_daily.sh 600000                  # 最新日期单股票策略报告
#   ./scripts/run_daily.sh --date 2025-03-14 600000  # 指定日期单股票策略报告
#
# 注：此脚本仅用于预测指定日期的决策，速度快（~30秒）
#    若需要完整的 walk-forward 训练和回测，请运行 ./scripts/run_model.sh
#
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"

# 解析参数：--date YYYY-MM-DD、以及可选的股票代码
DATE_ARG=""
STOCK_CODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)
            DATE_ARG="$2"
            shift 2
            ;;
        *)
            STOCK_CODE="$1"
            shift
            ;;
    esac
done

TODAY=$(date +%Y-%m-%d)
DISPLAY_DATE="${DATE_ARG:-$TODAY}"

echo "=============================================="
echo "  A股量化策略 — 每日预测 $DISPLAY_DATE"
if [ -n "$DATE_ARG" ]; then
    echo "  [历史回溯模式] 数据截止: $DATE_ARG"
fi
if [ -n "$STOCK_CODE" ]; then
    echo "  目标股票: $STOCK_CODE"
fi
echo "=============================================="

# 构建 --date 参数（若指定）
DATE_FLAG=""
[ -n "$DATE_ARG" ] && DATE_FLAG="--date $DATE_ARG"

# Step 1: 增量更新数据（历史模式跳过，数据已存在）
echo ""
if [ -n "$DATE_ARG" ]; then
    echo "[Step 1/4] 历史模式：跳过数据更新，使用现有 CSV 数据"
else
    echo "[Step 1/4] 增量更新数据..."
    python "$SRC_DIR/py00_fetch_stock_data.py" --update
fi

# Step 2: 数据清洗（截断到指定日期）
echo ""
echo "[Step 2/4] 数据清洗..."
python "$SRC_DIR/py01_data_loader.py" $DATE_FLAG

# Step 3: 特征工程（截断到指定日期）
echo ""
echo "[Step 3/4] 特征工程..."
python "$SRC_DIR/py02_features.py" $DATE_FLAG

# Step 4: 单日快速预测（只预测指定日期，不做 walk-forward）
echo ""
echo "[Step 4/4] 单日快速预测..."
python "$SRC_DIR/py03_model.py" $DATE_FLAG

# Step 5: 生成策略报告
echo ""
echo "[Step 5] 生成策略报告..."
if [ -n "$STOCK_CODE" ]; then
    python "$SRC_DIR/py05_today.py" $DATE_FLAG "$STOCK_CODE"
else
    python "$SRC_DIR/py05_today.py" $DATE_FLAG
    python "$SRC_DIR/py06_report.py"
fi

echo ""
echo "=============================================="
if [ -n "$STOCK_CODE" ]; then
    SHORT_CODE=$(echo "$STOCK_CODE" | grep -oE '[0-9]{6}')
    echo "  完成! 策略报告: output/today_strategy_${SHORT_CODE}.md"
else
    echo "  完成! 策略报告: output/today_strategy.md"
fi
echo "=============================================="
