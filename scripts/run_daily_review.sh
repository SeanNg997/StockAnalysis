#!/usr/bin/env bash
# run_daily_review.sh — 盘后决策评估一键脚本
#
# 用法:
#   ./scripts/run_daily_review.sh                    # 正常模式（需 16:00 之后）
#   ./scripts/run_daily_review.sh --force            # 跳过时间检查（调试用）
#   ./scripts/run_daily_review.sh --date 2025-03-14  # 评估指定历史日期的策略决策（自动跳过时间检查）
#
# --date 含义：策略基准日（即 today_strategy.md 中的"决策基准日"）
#   评估的执行日由策略文件中的"决策应用日期"自动确定
#
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
OUTPUT_DIR="$PROJECT_DIR/output"
STRATEGY_MD="$OUTPUT_DIR/today_strategy.md"

FORCE_FLAG=""
DATE_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE_FLAG="--force"; shift ;;
        --date)  DATE_ARG="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# 指定历史日期时自动跳过时间检查
DATE_FLAG=""
if [ -n "$DATE_ARG" ]; then
    DATE_FLAG="--date $DATE_ARG"
    FORCE_FLAG="--force"
fi

DISPLAY_DATE="${DATE_ARG:-$(date +%Y-%m-%d)}"

echo "=============================================="
echo "  A股量化策略 — 盘后决策评估 $DISPLAY_DATE"
echo "=============================================="

# ── Step 1: 确保盘前报告与目标基准日匹配 ─────────────────────────
if [ ! -f "$STRATEGY_MD" ]; then
    echo ""
    echo "[Step 1] 未找到盘前策略文件，正在生成盘前报告..."
    bash "$PROJECT_DIR/scripts/run_daily.sh" $DATE_FLAG
else
    # 从策略文件提取决策基准日
    STRATEGY_DATE=$(python3 -c "
import re, sys
content = open('$STRATEGY_MD', encoding='utf-8').read()
m = re.search(r'\*{0,2}决策基准日\*{0,2}[:：]\*{0,2}\s*(\d{4}-\d{2}-\d{2})', content)
print(m.group(1) if m else '')
")
    if [ "$STRATEGY_DATE" != "$DISPLAY_DATE" ]; then
        echo ""
        echo "[Step 1] 策略基准日为 $STRATEGY_DATE（目标 $DISPLAY_DATE），正在重新生成盘前报告..."
        bash "$PROJECT_DIR/scripts/run_daily.sh" $DATE_FLAG
    else
        echo ""
        echo "[Step 1] 盘前报告已匹配目标日期（$DISPLAY_DATE），跳过生成。"
    fi
fi

# ── Step 2: 运行盘后评估 ─────────────────────────────────────
# 不传 --date 给 py07_review.py，执行日由策略文件中的"决策应用日期"自动确定
echo ""
echo "[Step 2] 运行盘后决策评估..."
python "$SRC_DIR/py07_review.py" $FORCE_FLAG

echo ""
echo "=============================================="
# 从策略文件提取决策应用日期作为评估文件名
EXEC_DATE=$(python3 -c "
import re
content = open('$STRATEGY_MD', encoding='utf-8').read()
m = re.search(r'\*{0,2}决策应用日期\*{0,2}[:：]\*{0,2}\s*(\d{4}-\d{2}-\d{2})', content)
print(m.group(1).replace('-', '') if m else '')
")
REVIEW_MD="$OUTPUT_DIR/today_review_${EXEC_DATE}.md"
REVIEW_PNG="$OUTPUT_DIR/today_review_${EXEC_DATE}.png"
if [ -f "$REVIEW_MD" ]; then
    echo "  完成! 评估报告: $REVIEW_MD"
fi
if [ -f "$REVIEW_PNG" ]; then
    echo "  评估图表: $REVIEW_PNG"
fi
echo "=============================================="
