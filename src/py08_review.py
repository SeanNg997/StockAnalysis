"""
py08_review.py — 盘后决策评估模块
====================================
职责：
1. 解析今日盘前决策（output/today_strategy.md）
2. 读取今日实际行情（最新月度CSV）
3. 生成对比可视化图（output/today_review_YYYYMMDD.png）

评估口径：
  策略为"T+1开盘买入 → 持有5个交易日"
  今日评估 = 买入价（今日开盘）→ 收盘价（浮动盈亏，持有中）
  完整收益需持有期满才能确认

用法：
  python src/py08_review.py                        # 评估今日
  python src/py08_review.py --force                # 跳过时间检查（调试用）
  python src/py08_review.py --date 2025-03-15      # 评估指定历史日期（自动跳过时间检查）
"""

import os
import sys
import glob
import re
from datetime import datetime, date, time as dtime

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.family'] = ['PingFang HK', 'Hiragino Sans GB', 'STHeiti',
                                       'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
STRATEGY_MD = os.path.join(OUTPUT_DIR, 'today_strategy.md')

# 评估时间窗口：收盘(15:00) + 1小时 = 16:00
REVIEW_HOUR = 16


# ── 时间检查 ─────────────────────────────────────────────────────

def check_review_time(force: bool = False, eval_date: date = None) -> None:
    """检查当前时间是否满足评估条件（16:00之后）。
    指定历史日期时自动跳过时间检查。"""
    if force or (eval_date is not None and eval_date < datetime.now().date()):
        return
    now = datetime.now()
    if now.weekday() >= 5:
        sys.exit("今日为周末，非交易日，无需评估。")
    if now.hour < REVIEW_HOUR:
        remain = REVIEW_HOUR - now.hour
        sys.exit(f"当前时间 {now.strftime('%H:%M')}，请在 {REVIEW_HOUR}:00 之后运行评估（还需等待约 {remain} 小时）。")


# ── 解析盘前决策 ──────────────────────────────────────────────────

def parse_strategy(md_path: str) -> tuple[date, date, list[dict], bool]:
    """
    解析 today_strategy.md，返回：
    - strategy_date: 决策基准日（执行日前一天）
    - exec_date: 决策应用日期（执行日）
    - top5: 推荐买入TOP5列表（每项含代码、名称、预测收益率、建议仓位、收盘价）
    - is_empty: 是否建议空仓

    支持两种格式：
    1. 新格式（TOP N 子标题 + 小表格）：
       ### TOP 1 — 名称　`代码`
       | 收盘价 | 预测收益率 | 置信度 | 建议仓位 |
       | ¥36.01 | **+0.1659%** | 84.72% | 50% |

    2. 旧格式（大表格）：
       | 1 | sh.600053 | 九鼎投资 | 15.04 | +0.3247% | 0.9341 | 20.0% |
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取执行日（格式：**执行日**：YYYY-MM-DD）
    m_exec = re.search(r'\*{0,2}执行日\*{0,2}[:：]\s*(\d{4}-\d{2}-\d{2})', content)
    if not m_exec:
        # 兼容旧格式
        m_exec = re.search(r'\*{0,2}决策应用日期\*{0,2}[:：]\*{0,2}\s*(\d{4}-\d{2}-\d{2})', content)
    if not m_exec:
        raise ValueError("无法从策略文件中解析执行日期")
    exec_date = datetime.strptime(m_exec.group(1), '%Y-%m-%d').date()

    # 决策基准日 = 执行日前一天（简化处理，跳过周末）
    from datetime import timedelta
    strategy_date = exec_date - timedelta(days=1)
    while strategy_date.weekday() >= 5:
        strategy_date -= timedelta(days=1)

    # 也尝试解析显式的决策基准日
    m_base = re.search(r'\*{0,2}决策基准日\*{0,2}[:：]\*{0,2}\s*(\d{4}-\d{2}-\d{2})', content)
    if m_base:
        strategy_date = datetime.strptime(m_base.group(1), '%Y-%m-%d').date()

    # 检查是否空仓
    if '建议空仓' in content:
        return strategy_date, exec_date, [], True

    # 尝试新格式解析：### TOP N — 名称　`代码`
    top5 = []
    top_blocks = re.findall(
        r'###\s+TOP\s+(\d+)\s*[—–-]\s*(\S+)\s+`([^`]+)`(.*?)(?=###|---|$)',
        content, re.DOTALL
    )

    if top_blocks:
        for rank_str, name, code, block in top_blocks:
            # 解析小表格中的数据行
            # | ¥36.01 | **+0.1659%** | 84.72% | 50% |
            m = re.search(
                r'\|\s*[¥￥]?([\d.]+)\s*\|\s*\*{0,2}([+-]?[\d.]+)%\*{0,2}\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|',
                block
            )
            if m:
                top5.append({
                    'rank': int(rank_str),
                    '代码': code,
                    '名称': name,
                    'pred_close': float(m.group(1)),
                    'pred_return': float(m.group(2)) / 100,
                    'confidence': float(m.group(3)) / 100,
                    'weight': float(m.group(4)) / 100,
                })
            if len(top5) >= 5:
                break
    else:
        # 旧格式解析
        in_top5 = False
        for line in content.splitlines():
            if '推荐买入' in line and 'TOP 5' in line:
                in_top5 = True
                continue
            if in_top5 and line.startswith('##') and '推荐买入' not in line:
                break
            if not in_top5:
                continue
            if '代码' in line or '----' in line or '排名' in line:
                continue
            m = re.match(
                r'\|\s*(\d+)\s*\|\s*((?:sh|sz)\.\d{6})\s*\|\s*(\S+)\s*\|\s*([\d.]+)\s*\|\s*([+-]?[\d.]+)%\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*\|',
                line
            )
            if m:
                top5.append({
                    'rank': int(m.group(1)),
                    '代码': m.group(2),
                    '名称': m.group(3),
                    'pred_close': float(m.group(4)),
                    'pred_return': float(m.group(5)) / 100,
                    'confidence': float(m.group(6)),
                    'weight': float(m.group(7)) / 100,
                })
            if len(top5) == 5:
                break

    return strategy_date, exec_date, top5, False


# ── 读取今日实际行情 ──────────────────────────────────────────────

def load_today_data(target_date: date) -> pd.DataFrame:
    """从月度CSV中读取指定日期的行情数据"""
    yyyymm = target_date.strftime('%Y%m')
    csv_path = os.path.join(DATA_DIR, f'Stock_dailyK_{yyyymm}.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到行情文件: {csv_path}，请先运行数据更新。")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date']).dt.date
    today_df = df[df['date'] == target_date].copy()

    if today_df.empty:
        raise ValueError(f"{target_date} 无行情数据，可能是非交易日或数据尚未更新。")

    return today_df


# ── 合并决策与行情 ─────────────────────────────────────────────────

def merge_decision_with_actual(top5: list[dict], today_df: pd.DataFrame) -> pd.DataFrame:
    """将盘前决策与今日实际行情合并，计算各项指标"""
    rows = []
    for item in top5:
        code = item['代码']
        actual = today_df[today_df['代码'] == code]
        row = item.copy()

        if actual.empty:
            row.update({
                'actual_open': np.nan, 'actual_close': np.nan,
                'actual_pctChg': np.nan, 'intraday_return': np.nan,
                'weighted_return': np.nan, 'status': '停牌/无数据',
            })
        else:
            r = actual.iloc[0]
            open_p = float(r['open'])
            close_p = float(r['close'])
            pct_chg = float(r['pctChg'])       # 今日涨跌幅（%）
            intraday = (close_p - open_p) / open_p  # 开盘→收盘内日收益

            # 判断涨跌停
            limit_up = pct_chg >= 9.9
            limit_down = pct_chg <= -9.9
            status = '涨停' if limit_up else ('跌停' if limit_down else '正常')

            row.update({
                'actual_open': open_p,
                'actual_close': close_p,
                'actual_pctChg': pct_chg / 100,    # 转为小数
                'intraday_return': intraday,
                'weighted_return': intraday * item['weight'],
                'status': status,
            })
        rows.append(row)

    return pd.DataFrame(rows)


# ── 计算市场基准 ──────────────────────────────────────────────────

def calc_market_stats(today_df: pd.DataFrame) -> dict:
    """计算今日全市场统计"""
    df = today_df.dropna(subset=['pctChg']).copy()
    df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
    df = df.dropna(subset=['pctChg'])
    return {
        'total': len(df),
        'up_count': (df['pctChg'] > 0).sum(),
        'down_count': (df['pctChg'] < 0).sum(),
        'flat_count': (df['pctChg'] == 0).sum(),
        'mean_pct': df['pctChg'].mean() / 100,
        'median_pct': df['pctChg'].median() / 100,
    }


# ── 可视化 ────────────────────────────────────────────────────────

def _pct_fmt(v: float) -> str:
    return f"{v:+.2%}" if not np.isnan(v) else "N/A"

def _color(v: float, nan_color: str = '#888888') -> str:
    if np.isnan(v):
        return nan_color
    return '#E8423E' if v >= 0 else '#1DBB50'  # A股红涨绿跌


def plot_review(result_df: pd.DataFrame, market: dict,
                strategy_date: date, today: date, out_path: str) -> None:
    """生成盘后评估图"""
    fig = plt.figure(figsize=(16, 10), facecolor='#1C1C1E')
    fig.patch.set_facecolor('#1C1C1E')

    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.07, right=0.97, top=0.88, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, :])   # 主柱状图（跨两列）
    ax2 = fig.add_subplot(gs[1, 0])   # 市场环境
    ax3 = fig.add_subplot(gs[1, 1])   # 详细数据表

    dark_bg = '#2C2C2E'
    text_color = '#F5F5F7'
    grid_color = '#3A3A3C'

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)

    # ── 图1：推荐 TOP 5 对比柱状图 ──────────────────────────────
    n = len(result_df)
    x = np.arange(n)
    w = 0.28

    pred_vals = result_df['pred_return'].values
    actual_vals = result_df['actual_pctChg'].fillna(0).values
    intraday_vals = result_df['intraday_return'].fillna(0).values

    bars1 = ax1.bar(x - w, pred_vals, w, label='盘前预测收益率',
                    color='#5E81F4', alpha=0.9, zorder=3)
    bars2 = ax1.bar(x, actual_vals, w, label='今日实际涨跌幅',
                    color=[_color(v) for v in actual_vals], alpha=0.9, zorder=3)
    bars3 = ax1.bar(x + w, intraday_vals, w, label='今日开盘→收盘',
                    color=[_color(v, '#888888') for v in intraday_vals],
                    alpha=0.7, zorder=3)

    # 柱顶标注数值
    for bars, vals in [(bars1, pred_vals), (bars2, actual_vals), (bars3, intraday_vals)]:
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            y_pos = bar.get_height() + (0.0003 if val >= 0 else -0.0008)
            ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f"{val:+.2%}", ha='center', va='bottom' if val >= 0 else 'top',
                     fontsize=8, color=text_color, fontweight='bold')

    ax1.set_xticks(x)
    labels = [f"{r['名称']}\n{r['代码'].split('.')[1]}" for _, r in result_df.iterrows()]
    ax1.set_xticklabels(labels, color=text_color, fontsize=9)
    ax1.axhline(0, color=grid_color, linewidth=0.8, zorder=2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax1.tick_params(axis='y', colors=text_color)
    ax1.set_ylabel('收益率', color=text_color, fontsize=9)
    ax1.legend(fontsize=8, labelcolor=text_color, facecolor=dark_bg,
               edgecolor=grid_color, loc='upper right')
    ax1.set_title(f'推荐 TOP 5 — 盘前预测 vs 今日实际  ({today})',
                  color=text_color, fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', color=grid_color, linewidth=0.5, zorder=1)

    # ── 图2：市场环境对比 ────────────────────────────────────────
    categories = ['上涨', '平盘', '下跌']
    counts = [market['up_count'], market['flat_count'], market['down_count']]
    colors_mkt = ['#E8423E', '#888888', '#1DBB50']
    bars_mkt = ax2.bar(categories, counts, color=colors_mkt, alpha=0.85, zorder=3)
    for bar, cnt in zip(bars_mkt, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(cnt), ha='center', va='bottom', color=text_color, fontsize=9)

    # 叠加组合平均收益注释
    portfolio_avg = result_df['actual_pctChg'].mean()
    mkt_avg = market['mean_pct']
    note = (f"组合均值: {_pct_fmt(portfolio_avg)}\n"
            f"市场均值: {_pct_fmt(mkt_avg)}\n"
            f"上涨占比: {market['up_count']/market['total']:.1%}")
    ax2.text(0.97, 0.97, note, transform=ax2.transAxes,
             ha='right', va='top', color=text_color, fontsize=8,
             bbox=dict(facecolor='#3A3A3C', edgecolor='none', alpha=0.8, pad=5))

    ax2.set_title('今日市场概况', color=text_color, fontsize=10, fontweight='bold')
    ax2.set_ylabel('股票数量', color=text_color, fontsize=9)
    ax2.grid(axis='y', color=grid_color, linewidth=0.5, zorder=1)

    # ── 图3：详细数据表 ──────────────────────────────────────────
    ax3.axis('off')
    col_labels = ['代码', '名称', '盘前预测', '今日涨跌', '开→收', '仓位', '状态']
    table_data = []
    for _, r in result_df.iterrows():
        table_data.append([
            r['代码'].split('.')[1],
            r['名称'],
            _pct_fmt(r['pred_return']),
            _pct_fmt(r['actual_pctChg']),
            _pct_fmt(r['intraday_return']),
            f"{r['weight']:.0%}",
            r['status'],
        ])

    # 加总行
    total_weighted = result_df['weighted_return'].sum()
    avg_actual = result_df['actual_pctChg'].mean()
    table_data.append(['—', '组合汇总', '—', _pct_fmt(avg_actual),
                       _pct_fmt(total_weighted), '100%', ''])

    tbl = ax3.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    # 表格样式
    header_color = '#3A3A3C'
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_facecolor(dark_bg if row_idx > 0 else header_color)
        cell.set_text_props(color=text_color)
        cell.set_edgecolor(grid_color)
        # 最后一行（汇总）加深背景
        if row_idx == len(table_data):
            cell.set_facecolor('#3A3A3C')

    ax3.set_title('详细数据', color=text_color, fontsize=10, fontweight='bold', pad=12)

    # ── 总标题 ────────────────────────────────────────────────────
    portfolio_intraday = result_df['weighted_return'].sum()
    verdict = "盈利" if portfolio_intraday > 0 else ("持平" if portfolio_intraday == 0 else "亏损")
    verdict_color = _color(portfolio_intraday)
    title_color = verdict_color if portfolio_intraday != 0 else text_color
    fig.suptitle(
        f"盘后决策评估  |  盘前日期: {strategy_date}  →  执行日: {today}"
        f"  |  组合加权收益: {_pct_fmt(portfolio_intraday)}  [{verdict}]",
        color=title_color, fontsize=13, fontweight='bold', y=0.95
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"评估图已保存: {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────

def run_review(force: bool = False, eval_date: date = None) -> None:
    check_review_time(force, eval_date)

    if not os.path.exists(STRATEGY_MD):
        sys.exit(f"找不到策略文件: {STRATEGY_MD}，请先运行盘前报告。")

    strategy_date, exec_date, top5, is_empty = parse_strategy(STRATEGY_MD)
    print(f"盘前决策日期: {strategy_date}")
    print(f"决策应用日期: {exec_date}")

    if is_empty:
        print("今日盘前建议空仓，无需评估。")
        return

    if not top5:
        print("未解析到有效的推荐股票，请检查策略文件格式。")
        return

    print(f"共解析到 {len(top5)} 只推荐股票: {[s['代码'] for s in top5]}")

    # 评估日 = 策略中的决策应用日期（执行日），而非当前系统日期
    # --date 参数仅在显式指定时覆盖（用于历史回测场景）
    today = eval_date if eval_date is not None else exec_date
    print(f"评估日期（执行日）: {today}")

    try:
        today_df = load_today_data(today)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(
            f"无法读取执行日 {today} 的行情数据: {e}\n"
            f"评估需要执行日（决策应用日期）的实际行情数据，请确认:\n"
            f"  1. 执行日 {today} 已收盘\n"
            f"  2. 已运行数据更新（py00_fetch_stock_data.py --update）"
        )

    result_df = merge_decision_with_actual(top5, today_df)
    market = calc_market_stats(today_df)

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"{'代码':<14} {'名称':<8} {'预测':>8} {'今日涨跌':>9} {'状态'}")
    print("-" * 60)
    for _, r in result_df.iterrows():
        print(f"{r['代码']:<14} {r['名称']:<8} "
              f"{_pct_fmt(r['pred_return']):>8} "
              f"{_pct_fmt(r['actual_pctChg']):>9}  {r['status']}")
    print("-" * 60)
    weighted_total = result_df['weighted_return'].sum()
    print(f"组合加权收益（开→收）: {_pct_fmt(weighted_total)}")
    print(f"市场均值: {_pct_fmt(market['mean_pct'])}  |  上涨比例: {market['up_count']/market['total']:.1%}")
    print("=" * 60)

    out_path = os.path.join(OUTPUT_DIR, f"today_review_{today.strftime('%Y%m%d')}.png")
    plot_review(result_df, market, strategy_date, today, out_path)
    print("\n评估完成。")


if __name__ == '__main__':
    force = '--force' in sys.argv
    _eval_date = None
    if '--date' in sys.argv:
        _idx = sys.argv.index('--date')
        _eval_date = datetime.strptime(sys.argv[_idx + 1], '%Y-%m-%d').date()
    run_review(force, eval_date=_eval_date)
