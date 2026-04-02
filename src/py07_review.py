"""
py08_review.py — 盘后决策评估模块
====================================
职责：
1. 基于决策日(T)的预测结果，评估执行日(T+1)的实际表现
2. 读取执行日行情（需先运行 py00 --update 更新数据）
3. 评估买入推荐TOP5和潜力TOP20的预测准确度
4. 生成评估报告（Markdown）+ 可视化图表

日期定义（与 py05_today.py 一致）：
  T日   = 决策日（predictions.pkl 中的日期，盘后生成策略报告）
  T+1日 = 执行日/评估日（以T+1日开盘集合竞价买入，用T+1日收盘评估首日表现）

  这与 py05_today.py 的逻辑一致：
    py05 读取 predictions.pkl 最新日期 T → 推荐在 T+1 买入
    py08 评估 T+1 的表现：T+1 开盘买入 → T+1 收盘评估首日浮盈

评估口径：
  买入价 = T+1日开盘价
  首日浮盈 = (T+1日收盘 - T+1日开盘) / T+1日开盘
  完整收益 = 持有5天后开盘卖出（此处仅评估首日）

用法：
  python src/py08_review.py                        # 自动检测
  python src/py08_review.py --force                # 跳过时间检查
  python src/py08_review.py --date 2025-03-15      # 指定执行日(T+1日)
"""

import os
import sys
import glob as _glob
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['PingFang HK', 'Hiragino Sans GB', 'STHeiti',
                                       'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PREDICT_PKL = os.path.join(BASE_DIR, 'data', 'predictions.pkl')
FEATURE_PKL = os.path.join(BASE_DIR, 'data', 'features.pkl')

# 策略参数（与 py04 一致）
MIN_PRED_RETURN = 0.002
MIN_CONFIDENCE = 0.5

# ── 时间检查 ─────────────────────────────────────────────────────

def is_trading_day(check_date: date) -> bool:
    """判断是否为交易日（周一到周五）"""
    return check_date.weekday() < 5


def check_review_time(force: bool = False, exec_date_arg: date = None) -> None:
    """检查当前时间是否满足评估条件。

    允许条件：
    1. 指定 --force 标志
    2. 指定历史日期（exec_date_arg < 今日）
    3. 当前时间在第二个交易日开盘前都可以（即从今天 15:00 到明天 09:30 前都行，
       只要没有跨越到后一个交易日的 09:30 之后）
    """
    if force or (exec_date_arg is not None and exec_date_arg < datetime.now().date()):
        return

    now = datetime.now()

    # 计算第二个交易日开盘时间（09:30）
    check_date = now.date()
    days_offset = 1
    next_trading_day = check_date
    while days_offset <= 3:  # 最多向后查找3天
        next_trading_day = check_date + timedelta(days=days_offset)
        if is_trading_day(next_trading_day):
            break
        days_offset += 1

    # 第二个交易日开盘时间（09:30）
    next_open_time = datetime.combine(next_trading_day, datetime.min.time().replace(hour=9, minute=30))

    if now >= next_open_time:
        sys.exit(f"当前时间 {now.strftime('%Y-%m-%d %H:%M')} 已超过第二交易日开盘时间 {next_open_time.strftime('%Y-%m-%d %H:%M')}，"
                 f"无法评估。请在下个评估周期进行。")


# ── 数据加载 ─────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    """加载预测结果"""
    if not os.path.exists(PREDICT_PKL):
        sys.exit(f"找不到预测文件: {PREDICT_PKL}，请先运行 py03_model.py")
    return pd.read_pickle(PREDICT_PKL)


def load_price_info(target_date) -> pd.DataFrame:
    """从 features.pkl 加载指定日期的价格信息"""
    if not os.path.exists(FEATURE_PKL):
        sys.exit(f"找不到特征文件: {FEATURE_PKL}，请先运行 py02_features.py")
    df = pd.read_pickle(FEATURE_PKL)
    price_df = df[df['date'] == target_date][['代码', '名称', 'open', 'close']].copy()
    return price_df


def load_today_data(target_date: date) -> pd.DataFrame:
    """从月度CSV中读取指定日期的行情数据"""
    yyyymm = target_date.strftime('%Y%m')
    csv_path = os.path.join(DATA_DIR, f'Stock_dailyK_{yyyymm}.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到行情文件: {csv_path}，请先运行 py00 --update")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # 确保数值列正确
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = pd.to_datetime(df['date']).dt.date
    today_df = df[df['date'] == target_date].copy()

    if today_df.empty:
        raise ValueError(f"{target_date} 无行情数据，可能是非交易日或数据尚未更新。")

    return today_df


def determine_eval_date(pred_df: pd.DataFrame, exec_date_arg: date = None):
    """
    确定决策日(T)和执行日(T+1)。

    逻辑：
    - 决策日 T = predictions.pkl 中的日期（盘后选股）
    - 执行日 T+1 = T 的下一个交易日（开盘买入、收盘评估）

    参数：
    - exec_date_arg: 用户通过 --date 指定的执行日。
      - 若指定：找 predictions.pkl 中 < exec_date_arg 的最新日期作为决策日 T
      - 若不指定：T = predictions.pkl 最新日期，从CSV中找 T 之后最近的交易日作为执行日

    Returns:
        decision_date: T日（决策日，predictions.pkl 的日期）
        exec_date: T+1日（执行日/评估日）
    """
    all_pred_dates = sorted(pred_df['date'].unique())

    if exec_date_arg is not None:
        # 用户指定执行日 → 反推决策日
        ts = pd.Timestamp(exec_date_arg)
        candidates = [d for d in all_pred_dates if d < ts]
        if not candidates:
            sys.exit(f"predictions.pkl 中没有 {exec_date_arg} 之前的预测数据。\n"
                     f"可用日期范围: {all_pred_dates[0].date()} ~ {all_pred_dates[-1].date()}")
        decision_date = candidates[-1]
        return decision_date, exec_date_arg

    # 自动模式：决策日 = 最新预测日
    decision_date = all_pred_dates[-1]

    # 从CSV中查找决策日之后最近的交易日作为执行日
    csv_files = sorted(_glob.glob(os.path.join(DATA_DIR, 'Stock_dailyK_*.csv')))
    if csv_files:
        for f in reversed(csv_files[-3:]):
            tmp = pd.read_csv(f, encoding='utf-8-sig', usecols=['date'])
            tmp['date'] = pd.to_datetime(tmp['date'])
            future_dates = tmp[tmp['date'] > decision_date]['date'].unique()
            if len(future_dates) > 0:
                exec_dt = pd.Timestamp(min(future_dates)).date()
                return decision_date, exec_dt

    # Fallback：决策日后第一个工作日
    next_d = decision_date.date() if hasattr(decision_date, 'date') else decision_date
    next_d = next_d + timedelta(days=1)
    while next_d.weekday() >= 5:
        next_d += timedelta(days=1)
    return decision_date, next_d


# ── 决策生成 ─────────────────────────────────────────────────────

def generate_decisions(pred_df: pd.DataFrame, decision_date, price_df: pd.DataFrame):
    """
    基于决策日(T)的预测生成买入推荐。

    Args:
        pred_df: 完整预测数据
        decision_date: 决策日(T)
        price_df: 决策日的价格信息

    Returns:
        top5: 买入推荐TOP5 DataFrame
        top20: 潜力排名TOP20 DataFrame
        all_pred: 所有预测（合并了价格信息）
    """
    latest_pred = pred_df[pred_df['date'] == decision_date].copy()
    if latest_pred.empty:
        sys.exit(f"predictions.pkl 中无 {decision_date} 的预测数据")

    # 合并决策日价格信息（收盘价用于展示）
    if not price_df.empty:
        latest_pred = latest_pred.merge(
            price_df[['代码', '名称', 'close']].rename(columns={'close': 'prev_close'}),
            on='代码', how='left'
        )
    else:
        latest_pred['名称'] = ''
        latest_pred['prev_close'] = np.nan

    # 综合评分（与 py04/py05 一致）
    latest_pred['score'] = (latest_pred['pred_return'] * 0.6 +
                            latest_pred['confidence'] * latest_pred['pred_return'] * 0.4)

    # TOP5: 满足阈值条件，按score排序
    qualified = latest_pred[
        (latest_pred['pred_return'] > MIN_PRED_RETURN) &
        (latest_pred['confidence'] > MIN_CONFIDENCE)
    ].copy()
    top5 = qualified.sort_values('score', ascending=False).head(5).copy()
    n_buy = len(top5)
    if n_buy > 0:
        top5['weight'] = 1.0 / n_buy

    # TOP20: 全市场按pred_return排序
    top20 = latest_pred.sort_values('pred_return', ascending=False).head(20).copy()

    return top5, top20, latest_pred


# ── 评估计算 ─────────────────────────────────────────────────────

def evaluate_stocks(decision_df: pd.DataFrame, exec_day_df: pd.DataFrame) -> pd.DataFrame:
    """
    将决策与执行日(T+1)实际行情合并，计算评估指标。

    对每只推荐股票：
    - actual_open: T+1日开盘价（买入价）
    - actual_close: T+1日收盘价
    - intraday_return: (close - open) / open（开盘→收盘收益）
    - actual_pctChg: T+1日涨跌幅
    - status: 正常/涨停/跌停/停牌
    """
    rows = []
    for _, item in decision_df.iterrows():
        code = item['代码']
        actual = exec_day_df[exec_day_df['代码'] == code]
        row = item.to_dict()

        if actual.empty:
            row.update({
                'actual_open': np.nan, 'actual_close': np.nan,
                'actual_pctChg': np.nan, 'intraday_return': np.nan,
                'status': '停牌/无数据',
            })
        else:
            r = actual.iloc[0]
            open_p = float(r['open'])
            close_p = float(r['close'])
            pct_chg = float(r['pctChg']) if pd.notna(r['pctChg']) else 0.0

            if open_p > 0:
                intraday = (close_p - open_p) / open_p
            else:
                intraday = np.nan

            # 涨跌停判断
            if pct_chg >= 9.9:
                status = '涨停'
            elif pct_chg <= -9.9:
                status = '跌停'
            else:
                status = '正常'

            row.update({
                'actual_open': open_p,
                'actual_close': close_p,
                'actual_pctChg': pct_chg / 100,
                'intraday_return': intraday,
                'status': status,
            })
        rows.append(row)

    return pd.DataFrame(rows)


def calc_market_stats(today_df: pd.DataFrame) -> dict:
    """计算T日全市场统计"""
    df = today_df.dropna(subset=['pctChg']).copy()
    df['pctChg'] = pd.to_numeric(df['pctChg'], errors='coerce')
    df = df.dropna(subset=['pctChg'])

    total = len(df)
    if total == 0:
        return {'total': 0, 'up_count': 0, 'down_count': 0, 'flat_count': 0,
                'mean_pct': 0, 'median_pct': 0}

    return {
        'total': total,
        'up_count': int((df['pctChg'] > 0).sum()),
        'down_count': int((df['pctChg'] < 0).sum()),
        'flat_count': int((df['pctChg'] == 0).sum()),
        'mean_pct': df['pctChg'].mean() / 100,
        'median_pct': df['pctChg'].median() / 100,
    }


# ── Markdown 报告 ────────────────────────────────────────────────

def _pct_fmt(v, nan_str="N/A") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return nan_str
    return f"{v:+.2%}"


def generate_markdown_report(
    decision_date, exec_date,
    top5_eval: pd.DataFrame, top20_eval: pd.DataFrame,
    market: dict,
) -> str:
    """生成评估 Markdown 报告"""
    lines = []
    decision_date_str = decision_date.date() if hasattr(decision_date, 'date') else decision_date
    exec_date_str = exec_date

    lines.append(f"# 盘后决策评估 — {exec_date_str}\n")
    lines.append(f"> **决策日(T)**: {decision_date_str} | **执行日(T+1)**: {exec_date_str}\n")
    lines.append("---\n")

    # ── 一、买入推荐 TOP5 ──
    lines.append("## 一、买入推荐 (TOP 5)\n")
    lines.append(f"> 筛选条件: 预测收益率 > {MIN_PRED_RETURN:.1%}, 置信度 > {MIN_CONFIDENCE:.0%}\n")

    if len(top5_eval) == 0:
        lines.append("> **无股票满足买入条件**\n")
    else:
        lines.append("| # | 代码 | 名称 | 预测收益率 | 置信度 | T+1开盘 | T+1收盘 | 开→收 | 状态 |")
        lines.append("|:-:|------|------|----------:|-------:|--------:|--------:|------:|:----:|")
        for i, (_, r) in enumerate(top5_eval.iterrows()):
            name = r.get('名称', 'N/A')
            if pd.isna(name):
                name = 'N/A'
            lines.append(
                f"| {i+1} | {r['代码']} | {name}"
                f" | {_pct_fmt(r['pred_return'])}"
                f" | {r.get('confidence', 0):.2%}"
                f" | {r.get('actual_open', 0):.2f}"
                f" | {r.get('actual_close', 0):.2f}"
                f" | **{_pct_fmt(r.get('intraday_return', np.nan))}**"
                f" | {r.get('status', 'N/A')} |"
            )

        # 组合统计
        valid = top5_eval['intraday_return'].dropna()
        if len(valid) > 0:
            avg_intraday = valid.mean()
            lines.append(f"\n**组合等权平均收益（开→收）: {_pct_fmt(avg_intraday)}**")
            excess = avg_intraday - market['mean_pct']
            lines.append(f"**超额收益（vs 市场均值）: {_pct_fmt(excess)}**\n")

    lines.append("---\n")

    # ── 二、潜力 TOP20 评估 ──
    lines.append("## 二、潜力排名 TOP 20 评估\n")
    lines.append("> 全市场按预测收益率排名，不限制阈值\n")
    lines.append("| # | 代码 | 名称 | 预测收益率 | T+1开盘 | T+1收盘 | 开→收 | 命中 |")
    lines.append("|:-:|------|------|----------:|--------:|--------:|------:|:----:|")

    hit_count = 0
    total_count = 0
    for i, (_, r) in enumerate(top20_eval.iterrows()):
        name = r.get('名称', 'N/A')
        if pd.isna(name):
            name = 'N/A'
        intraday = r.get('intraday_return', np.nan)
        pred_ret = r.get('pred_return', 0)

        # 命中 = 预测方向与实际一致
        if pd.notna(intraday):
            total_count += 1
            hit = (pred_ret > 0 and intraday > 0) or (pred_ret < 0 and intraday < 0)
            if hit:
                hit_count += 1
            hit_str = 'O' if hit else 'X'
        else:
            hit_str = '-'

        lines.append(
            f"| {i+1} | {r['代码']} | {name}"
            f" | {_pct_fmt(pred_ret)}"
            f" | {r.get('actual_open', 0):.2f}"
            f" | {r.get('actual_close', 0):.2f}"
            f" | {_pct_fmt(intraday)}"
            f" | {hit_str} |"
        )

    # TOP20 统计
    if total_count > 0:
        accuracy = hit_count / total_count
        lines.append(f"\n**预测方向准确率: {hit_count}/{total_count} ({accuracy:.1%})**")

        # 相关系数
        valid_mask = top20_eval['intraday_return'].notna()
        if valid_mask.sum() >= 3:
            corr = top20_eval.loc[valid_mask, 'pred_return'].corr(
                top20_eval.loc[valid_mask, 'intraday_return']
            )
            if pd.notna(corr):
                lines.append(f"**预测-实际相关系数: {corr:.3f}**")

        # TOP20平均收益
        top20_avg = top20_eval['intraday_return'].dropna().mean()
        lines.append(f"**TOP20平均开→收: {_pct_fmt(top20_avg)}**\n")

    lines.append("---\n")

    # ── 三、市场概况 ──
    lines.append("## 三、T+1日市场概况\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|-----:|")
    lines.append(f"| 全市场股票数 | {market['total']:,} |")
    lines.append(f"| 上涨 | {market['up_count']:,} ({market['up_count']/max(market['total'],1):.1%}) |")
    lines.append(f"| 下跌 | {market['down_count']:,} ({market['down_count']/max(market['total'],1):.1%}) |")
    lines.append(f"| 平盘 | {market['flat_count']:,} |")
    lines.append(f"| 市场均值 | {_pct_fmt(market['mean_pct'])} |")
    lines.append(f"| 市场中位数 | {_pct_fmt(market['median_pct'])} |")

    lines.append("\n---\n")
    lines.append("*仅评估首日（T+1日开→收）表现，完整收益需持有5天后确认。*")

    return "\n".join(lines)


# ── 可视化 ────────────────────────────────────────────────────────

def _color(v: float, nan_color: str = '#888888') -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return nan_color
    return '#E8423E' if v >= 0 else '#1DBB50'  # A股红涨绿跌


def plot_review(top5_eval: pd.DataFrame, top20_eval: pd.DataFrame,
                market: dict, decision_date, exec_date, out_path: str) -> None:
    """生成盘后评估图"""
    decision_date_str = decision_date.date() if hasattr(decision_date, 'date') else decision_date

    fig = plt.figure(figsize=(16, 12), facecolor='#1C1C1E')
    fig.patch.set_facecolor('#1C1C1E')

    gs = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35,
                  left=0.07, right=0.97, top=0.90, bottom=0.05)
    ax1 = fig.add_subplot(gs[0, :])   # TOP5 柱状图
    ax2 = fig.add_subplot(gs[1, 0])   # 市场环境
    ax3 = fig.add_subplot(gs[1, 1])   # TOP5 数据表
    ax4 = fig.add_subplot(gs[2, :])   # TOP20 柱状图

    dark_bg = '#2C2C2E'
    text_color = '#F5F5F7'
    grid_color = '#3A3A3C'

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)

    # ── 图1：TOP 5 对比柱状图 ──
    if len(top5_eval) > 0:
        n = len(top5_eval)
        x = np.arange(n)
        w = 0.35

        pred_vals = top5_eval['pred_return'].values
        intraday_vals = top5_eval['intraday_return'].fillna(0).values

        ax1.bar(x - w/2, pred_vals, w, label='T日预测收益率',
                color='#5E81F4', alpha=0.9, zorder=3)
        bars2 = ax1.bar(x + w/2, intraday_vals, w, label='T+1日开→收',
                        color=[_color(v) for v in intraday_vals], alpha=0.9, zorder=3)

        # 标注数值
        for xi, (pv, iv) in enumerate(zip(pred_vals, intraday_vals)):
            for val, offset in [(pv, -w/2), (iv, w/2)]:
                if np.isnan(val):
                    continue
                y_pos = val + (0.0003 if val >= 0 else -0.0008)
                ax1.text(xi + offset, y_pos, f"{val:+.2%}",
                         ha='center', va='bottom' if val >= 0 else 'top',
                         fontsize=8, color=text_color, fontweight='bold')

        labels = []
        for _, r in top5_eval.iterrows():
            name = r.get('名称', '')
            if pd.isna(name):
                name = ''
            labels.append(f"{name}\n{r['代码'].split('.')[1]}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, color=text_color, fontsize=9)
    else:
        ax1.text(0.5, 0.5, '无满足条件的买入推荐', transform=ax1.transAxes,
                 ha='center', va='center', color=text_color, fontsize=12)

    ax1.axhline(0, color=grid_color, linewidth=0.8, zorder=2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax1.set_ylabel('收益率', color=text_color, fontsize=9)
    ax1.legend(fontsize=8, labelcolor=text_color, facecolor=dark_bg,
               edgecolor=grid_color, loc='upper right')
    ax1.set_title(f'买入推荐 TOP 5 — 预测 vs 实际  ({exec_date})',
                  color=text_color, fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', color=grid_color, linewidth=0.5, zorder=1)

    # ── 图2：市场环境 ──
    categories = ['上涨', '平盘', '下跌']
    counts = [market['up_count'], market['flat_count'], market['down_count']]
    colors_mkt = ['#E8423E', '#888888', '#1DBB50']
    bars_mkt = ax2.bar(categories, counts, color=colors_mkt, alpha=0.85, zorder=3)
    for bar, cnt in zip(bars_mkt, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(cnt), ha='center', va='bottom', color=text_color, fontsize=9)

    top5_avg = top5_eval['intraday_return'].dropna().mean() if len(top5_eval) > 0 else 0
    mkt_avg = market['mean_pct']
    note = (f"TOP5均值: {_pct_fmt(top5_avg)}\n"
            f"市场均值: {_pct_fmt(mkt_avg)}\n"
            f"上涨占比: {market['up_count']/max(market['total'],1):.1%}")
    ax2.text(0.97, 0.97, note, transform=ax2.transAxes,
             ha='right', va='top', color=text_color, fontsize=8,
             bbox=dict(facecolor='#3A3A3C', edgecolor='none', alpha=0.8, pad=5))
    ax2.set_title('T+1日市场概况', color=text_color, fontsize=10, fontweight='bold')
    ax2.set_ylabel('股票数量', color=text_color, fontsize=9)
    ax2.grid(axis='y', color=grid_color, linewidth=0.5, zorder=1)

    # ── 图3：TOP5 数据表 ──
    ax3.axis('off')
    if len(top5_eval) > 0:
        col_labels = ['代码', '名称', '预测', '开→收', '状态']
        table_data = []
        for _, r in top5_eval.iterrows():
            name = r.get('名称', 'N/A')
            if pd.isna(name):
                name = 'N/A'
            table_data.append([
                r['代码'].split('.')[1],
                name,
                _pct_fmt(r['pred_return']),
                _pct_fmt(r.get('intraday_return', np.nan)),
                r.get('status', 'N/A'),
            ])
        # 汇总行
        avg_intraday = top5_eval['intraday_return'].dropna().mean()
        table_data.append(['—', '均值', '—', _pct_fmt(avg_intraday), ''])

        tbl = ax3.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.92])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (row_idx, col_idx), cell in tbl.get_celld().items():
            cell.set_facecolor(dark_bg if row_idx > 0 else '#3A3A3C')
            cell.set_text_props(color=text_color)
            cell.set_edgecolor(grid_color)
            if row_idx == len(table_data):
                cell.set_facecolor('#3A3A3C')
    ax3.set_title('TOP 5 详细数据', color=text_color, fontsize=10, fontweight='bold', pad=12)

    # ── 图4：TOP 20 预测 vs 实际 ──
    if len(top20_eval) > 0:
        n20 = len(top20_eval)
        x20 = np.arange(n20)
        w20 = 0.35
        pred20 = top20_eval['pred_return'].values
        actual20 = top20_eval['intraday_return'].fillna(0).values

        ax4.bar(x20 - w20/2, pred20, w20, label='T日预测', color='#5E81F4', alpha=0.9, zorder=3)
        ax4.bar(x20 + w20/2, actual20, w20, label='T+1日开→收',
                color=[_color(v) for v in actual20], alpha=0.9, zorder=3)

        labels20 = []
        for _, r in top20_eval.iterrows():
            labels20.append(r['代码'].split('.')[1])
        ax4.set_xticks(x20)
        ax4.set_xticklabels(labels20, color=text_color, fontsize=7, rotation=45)
    ax4.axhline(0, color=grid_color, linewidth=0.8, zorder=2)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax4.set_ylabel('收益率', color=text_color, fontsize=9)
    ax4.legend(fontsize=8, labelcolor=text_color, facecolor=dark_bg,
               edgecolor=grid_color, loc='upper right')
    ax4.set_title(f'潜力 TOP 20 — 预测 vs 实际',
                  color=text_color, fontsize=12, fontweight='bold', pad=10)
    ax4.grid(axis='y', color=grid_color, linewidth=0.5, zorder=1)

    # ── 总标题 ──
    portfolio_avg = top5_eval['intraday_return'].dropna().mean() if len(top5_eval) > 0 else 0
    if np.isnan(portfolio_avg):
        portfolio_avg = 0
    verdict = "盈利" if portfolio_avg > 0 else ("持平" if portfolio_avg == 0 else "亏损")
    verdict_color = _color(portfolio_avg)
    title_color = verdict_color if portfolio_avg != 0 else text_color
    fig.suptitle(
        f"盘后决策评估  |  T: {decision_date_str}  →  T+1: {exec_date}"
        f"  |  TOP5均值: {_pct_fmt(portfolio_avg)}  [{verdict}]",
        color=title_color, fontsize=13, fontweight='bold', y=0.96
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"评估图已保存: {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────

def run_review(force: bool = False, exec_date_arg: date = None) -> None:
    check_review_time(force, exec_date_arg)

    # 1. 加载预测数据
    pred_df = load_predictions()

    # 2. 确定决策日(T)和执行日(T+1)
    decision_date, exec_date = determine_eval_date(pred_df, exec_date_arg)
    decision_date_str = decision_date.date() if hasattr(decision_date, 'date') else decision_date
    print(f"决策日(T):   {decision_date_str}")
    print(f"执行日(T+1): {exec_date}")

    # 3. 加载决策日的价格信息（收盘价用于展示）
    price_df = load_price_info(decision_date)

    # 4. 基于决策日预测生成推荐
    top5, top20, all_pred = generate_decisions(pred_df, decision_date, price_df)
    print(f"买入推荐: {len(top5)} 只, 潜力TOP20: {len(top20)} 只")
    if len(top5) > 0:
        print(f"  TOP5: {top5['代码'].tolist()}")

    # 5. 加载执行日(T+1)实际行情
    exec_date_val = exec_date if isinstance(exec_date, date) else exec_date.date()
    try:
        exec_day_df = load_today_data(exec_date_val)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(
            f"无法读取执行日 {exec_date} 的行情数据: {e}\n"
            f"请确认:\n"
            f"  1. 执行日 {exec_date} 已收盘\n"
            f"  2. 已运行数据更新（python src/py00_fetch_stock_data.py --update）"
        )

    # 6. 评估
    top5_eval = evaluate_stocks(top5, exec_day_df) if len(top5) > 0 else pd.DataFrame()
    top20_eval = evaluate_stocks(top20, exec_day_df)
    market = calc_market_stats(exec_day_df)

    # 7. 控制台摘要
    print(f"\n{'='*70}")
    print(f"  决策评估  |  T: {decision_date_str}  →  T+1: {exec_date}")
    print(f"{'='*70}")

    if len(top5_eval) > 0:
        print(f"\n{'─'*70}")
        print(f"  {'代码':<14} {'名称':<8} {'预测':>8} {'开→收':>9} {'状态'}")
        print(f"{'─'*70}")
        for _, r in top5_eval.iterrows():
            name = r.get('名称', 'N/A')
            if pd.isna(name):
                name = 'N/A'
            print(f"  {r['代码']:<14} {name:<8} "
                  f"{_pct_fmt(r['pred_return']):>8} "
                  f"{_pct_fmt(r.get('intraday_return', np.nan)):>9}  "
                  f"{r.get('status', 'N/A')}")
        avg_ret = top5_eval['intraday_return'].dropna().mean()
        print(f"{'─'*70}")
        print(f"  TOP5等权平均（开→收）: {_pct_fmt(avg_ret)}")
    else:
        print("\n  无满足条件的买入推荐")

    print(f"  市场均值: {_pct_fmt(market['mean_pct'])}  |  "
          f"上涨: {market['up_count']}/{market['total']} "
          f"({market['up_count']/max(market['total'],1):.1%})")

    # TOP20 方向准确率
    valid_top20 = top20_eval.dropna(subset=['intraday_return'])
    if len(valid_top20) > 0:
        hits = ((valid_top20['pred_return'] > 0) & (valid_top20['intraday_return'] > 0)).sum()
        print(f"  TOP20方向准确率: {hits}/{len(valid_top20)} "
              f"({hits/len(valid_top20):.1%})")
    print(f"{'='*70}")

    # 8. 生成 Markdown 报告
    md_report = generate_markdown_report(decision_date, exec_date, top5_eval, top20_eval, market)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    md_path = os.path.join(OUTPUT_DIR, f"today_review_{exec_date_val.strftime('%Y%m%d')}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"评估报告已保存: {md_path}")

    # 9. 生成 PNG 图表
    png_path = os.path.join(OUTPUT_DIR, f"today_review_{exec_date_val.strftime('%Y%m%d')}.png")
    plot_review(top5_eval, top20_eval, market, decision_date, exec_date, png_path)

    print("\n评估完成。")


if __name__ == '__main__':
    _force = '--force' in sys.argv
    _exec_date = None
    if '--date' in sys.argv:
        _idx = sys.argv.index('--date')
        _exec_date = datetime.strptime(sys.argv[_idx + 1], '%Y-%m-%d').date()
    run_review(_force, exec_date_arg=_exec_date)
