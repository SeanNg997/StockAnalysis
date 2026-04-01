#!/usr/bin/env bash
# run_daily_review.sh — 盘后决策评估一键脚本
#
# 用法:
#   ./scripts/run_daily_review.sh          # 正常模式（需 16:00 之后）
#   ./scripts/run_daily_review.sh --force  # 跳过时间检查（调试用）
#
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
OUTPUT_DIR="$PROJECT_DIR/output"
STRATEGY_TXT="$OUTPUT_DIR/today_strategy.txt"

FORCE_FLAG=""
for arg in "$@"; do
    [ "$arg" = "--force" ] && FORCE_FLAG="--force"
done

TODAY=$(date +%Y-%m-%d)

echo "=============================================="
echo "  A股量化策略 — 盘后决策评估 $TODAY"
echo "=============================================="

# ── Step 1: 检查盘前报告是否为今日 ─────────────────────────────
if [ ! -f "$STRATEGY_TXT" ]; then
    echo ""
    echo "[Step 1] 未找到盘前策略文件，正在生成今日盘前报告..."
    bash "$PROJECT_DIR/scripts/run_daily.sh"
else
    STRATEGY_DATE=$(grep -oP '决策基准日[:：]\s*\K\d{4}-\d{2}-\d{2}' "$STRATEGY_TXT" | head -1)
    if [ "$STRATEGY_DATE" != "$TODAY" ]; then
        echo ""
        echo "[Step 1] 盘前报告日期为 $STRATEGY_DATE（非今日 $TODAY），正在生成今日盘前报告..."
        bash "$PROJECT_DIR/scripts/run_daily.sh"
    else
        echo ""
        echo "[Step 1] 盘前报告已是今日（$TODAY），跳过生成。"
    fi
fi

# ── Step 2: 运行盘后评估 ─────────────────────────────────────
echo ""
echo "[Step 2] 运行盘后决策评估..."
python "$SRC_DIR/py08_review.py" $FORCE_FLAG

echo ""
echo "=============================================="
REVIEW_FILE="$OUTPUT_DIR/today_review_$(date +%Y%m%d).png"
if [ -f "$REVIEW_FILE" ]; then
    echo "  完成! 评估图: $REVIEW_FILE"
else
    echo "  完成!"
fi
echo "=============================================="
