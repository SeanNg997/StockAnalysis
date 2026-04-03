#!/usr/bin/env bash
# run_daily_review.sh — 盘后决策评估一键脚本
#
# 用法:
#   ./scripts/run_daily_review.sh                    # 自动模式：review 上一个已收盘交易日的决策
#   ./scripts/run_daily_review.sh --force            # 跳过时间检查（调试用）
#   ./scripts/run_daily_review.sh --date 2025-03-14  # 评估指定执行日(T+1)的策略（自动跳过时间检查）
#
# 日期逻辑（T/T+1 定义）：
#   T日   = 决策日（盘后运行策略生成预测的那天）
#   T+1日 = 执行日（按T日决策实际买入并收盘的那天，是评估目标）
#
#   review 的永远是"上一个已完全收盘的执行日(T+1)"，因为：
#   - 今天是周末/节假日 → 无行情数据，自动回退
#   - 今天是交易日但 < 15:00 → 今天尚未收盘，自动回退到上一个T+1
#   - 今天是交易日且 >= 15:00 → 今天可作为T+1参与评估
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

# 确定展示用的决策日(T)和执行日(T+1)
# 规则：T+1 必须是已完全收盘的交易日（今天 >= 15:00 才算今天收盘）
if [ -n "$DATE_ARG" ]; then
    DISPLAY_EXEC="$DATE_ARG"
    DISPLAY_DECISION=$(python3 -c "
import os, sys, pandas as pd
from datetime import datetime, date, timedelta
pkl = os.path.join('$PROJECT_DIR', 'data', 'predictions.pkl')
exec_day = datetime.strptime('$DATE_ARG', '%Y-%m-%d').date()
if os.path.exists(pkl):
    df = pd.read_pickle(pkl)
    all_dates = sorted(df['date'].dt.date.unique())
    candidates = [d for d in all_dates if d < exec_day]
    if candidates:
        print(candidates[-1])
        sys.exit(0)
# fallback：前一个工作日
d = exec_day - timedelta(days=1)
while d.weekday() >= 5:
    d -= timedelta(days=1)
print(d)
" 2>/dev/null || echo "unknown")
else
    # 自动模式：找上一个"已收盘的T+1"对应的决策日T
    read DISPLAY_DECISION DISPLAY_EXEC <<< $(python3 -c "
import os, sys, glob as _glob
import pandas as pd
from datetime import datetime, date, timedelta

now = datetime.now()
today = now.date()
market_closed = now.hour > 15 or (now.hour == 15 and now.minute >= 0)

pkl = os.path.join('$PROJECT_DIR', 'data', 'predictions.pkl')
data_dir = os.path.join('$PROJECT_DIR', 'data')

# 从 CSV 收集已有行情的交易日
csv_files = sorted(_glob.glob(os.path.join(data_dir, 'Stock_dailyK_*.csv')))
csv_dates = set()
for f in csv_files[-4:]:
    try:
        tmp = pd.read_csv(f, encoding='utf-8-sig', usecols=['date'])
        tmp['date'] = pd.to_datetime(tmp['date'])
        csv_dates.update(tmp['date'].dt.date.unique())
    except Exception:
        pass

def find_next_trading_day(d):
    if csv_dates:
        future = sorted(dt for dt in csv_dates if dt > d)
        if future:
            return future[0]
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt

def exec_day_closed(exec_day):
    if exec_day < today:
        return True
    if exec_day == today:
        return market_closed
    return False

if os.path.exists(pkl):
    df = pd.read_pickle(pkl)
    all_dates = sorted(df['date'].dt.date.unique())
    for decision_date in reversed(all_dates):
        exec_day = find_next_trading_day(decision_date)
        if exec_day_closed(exec_day):
            print(decision_date, exec_day)
            sys.exit(0)

# fallback：往前推工作日
d = today - timedelta(days=1)
while d.weekday() >= 5:
    d -= timedelta(days=1)
decision = d - timedelta(days=1)
while decision.weekday() >= 5:
    decision -= timedelta(days=1)
print(decision, d)
" 2>/dev/null || echo "unknown unknown")
fi

echo "=============================================="
echo "  A股量化策略 — 盘后决策评估"
echo "  决策日(T):   $DISPLAY_DECISION"
echo "  执行日(T+1): $DISPLAY_EXEC"
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
    if [ "$STRATEGY_DATE" != "$DISPLAY_DECISION" ]; then
        echo ""
        echo "[Step 1] 策略基准日为 $STRATEGY_DATE（目标 $DISPLAY_DECISION），正在重新生成盘前报告..."
        bash "$PROJECT_DIR/scripts/run_daily.sh" $DATE_FLAG
    else
        echo ""
        echo "[Step 1] 盘前报告已匹配目标日期（$DISPLAY_DECISION），跳过生成。"
    fi
fi

# ── Step 2: 运行盘后评估 ─────────────────────────────────────
echo ""
echo "[Step 2] 运行盘后决策评估..."
python "$SRC_DIR/py07_review.py" $FORCE_FLAG $DATE_FLAG

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
