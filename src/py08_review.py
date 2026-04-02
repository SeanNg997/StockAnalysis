"""
py08_review.py — 盘后决策评估模块
====================================
职责：
1. 基于T-1日预测结果生成T日决策（买入推荐TOP5 + 潜力TOP20）
2. 读取T日实际行情（需先运行 py00 --update 更新数据）
3. 评估T日买入推荐和TOP20的预测准确度
4. 生成评估报告（Markdown）

日期定义：
  T-1日 = 决策基准日（预测数据来源日）
  T日   = 评估日（以T日开盘买入，用T日收盘评估首日表现）
  注意：T+1日策略报告由 py05_today.py 生成（today_strategy.md），
        本模块输出独立文件（today_review_*.md），不会覆盖。

评估口径：
  买入价 = T日开盘价
  首日浮盈 = (T日收盘 - T日开盘) / T日开盘
  完整收益 = 持有5天后开盘卖出（此处仅评估首日）

用法：
  python src/py08_review.py                        # 评估今日(T日)
  python src/py08_review.py --force                # 跳过时间检查
  python src/py08_review.py --date 2025-03-15      # 评估指定历史日期
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


def check_review_time(force: bool = False, eval_date: date = None) -> None:
    """检查当前时间是否满足评估条件。

    允许条件：
    1. 指定 --force 标志
    2. 指定历史日期（eval_date < 今日）
    3. 当前时间在第二个交易日开盘前都可以（即从今天 15:00 到明天 09:30 前都行，
       只要没有跨越到后一个交易日的 09:30 之后）
    """
    if force or (eval_date is not None and eval_date < datetime.now().date()):
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


def determine_eval_date(pred_df: pd.DataFrame, eval_date: date = None):
    """
    确定评估日(T日)和决策基准日(T-1日)。

    Returns:
        pred_date: T-1日（预测基准日）
        eval_date: T日（评估日）
    """
    if eval_date is not None:
        # 指定评估日：找 predictions.pkl 中 < eval_date 的最新预测日
        ts = pd.Timestamp(eval_date)
        candidates = pred_df[pred_df['date'] < ts]['date']
        if candidates.empty:
            sys.exit(f"predictions.pkl 中没有 {eval_date} 之前的预测数据。"
                     f"请确认已运行过 py03 且预测日期正确。")
        pred_date = candidates.max()
        return pred_date, eval_date

    # 自动检测：预测日 = predictions.pkl 中最新日期（T-1），T日 = 下一个交易日
    pred_date = pred_df['date'].max()

    # 从CSV中查找pred_date之后的最早日期作为T日
    csv_files = sorted(_glob.glob(os.path.join(DATA_DIR, 'Stock_dailyK_*.csv')))
    if csv_files:
        # 从最后几个月度文件中查找
        for f in reversed(csv_files[-3:]):
            tmp = pd.read_csv(f, encoding='utf-8-sig', usecols=['date'])
            tmp['date'] = pd.to_datetime(tmp['date'])
            future_dates = tmp[tmp['date'] > pred_date]['date'].unique()
            if len(future_dates) > 0:
                eval_dt = pd.Timestamp(min(future_dates)).date()
                return pred_date, eval_dt

    # Fallback：pred_date后第一个工作日
    next_d = pred_date.date() if hasattr(pred_date, 'date') else pred_date
    next_d = next_d + timedelta(days=1)
    while next_d.weekday() >= 5:
        next_d += timedelta(days=1)
    return pred_date, next_d


# ── 决策生成 ─────────────────────────────────────────────────────

def generate_decisions(pred_df: pd.DataFrame, pred_date, price_df: pd.DataFrame):
    """
    基于T-1日预测生成T日决策。

    Returns:
        top5: 买入推荐TOP5 DataFrame
        top20: 潜力排名TOP20 DataFrame
        all_pred: 所有预测（合并了价格信息）
    """
    latest_pred = pred_df[pred_df['date'] == pred_date].copy()
    if latest_pred.empty:
        sys.exit(f"predictions.pkl 中无 {pred_date} 的预测数据")

    # 合并T-1日价格信息（收盘价用于展示）
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

def evaluate_stocks(decision_df: pd.DataFrame, today_df: pd.DataFrame) -> pd.DataFrame:
    """
    将决策与T日实际行情合并，计算评估指标。

    对每只推荐股票：
    - actual_open: T日开盘价（买入价）
    - actual_close: T日收盘价
    - intraday_return: (close - open) / open（开盘→收盘收益）
    - actual_pctChg: T日涨跌幅
    - status: 正常/涨停/跌停/停牌
    """
    rows = []
    for _, item in decision_df.iterrows():
        code = item['代码']
        actual = today_df[today_df['代码'] == code]
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
    pred_date, eval_date,
    top5_eval: pd.DataFrame, top20_eval: pd.DataFrame,
    market: dict,
) -> str:
    """生成评估 Markdown 报告"""
    lines = []
    pred_date_str = pred_date.date() if hasattr(pred_date, 'date') else pred_date
    eval_date_str = eval_date

    lines.append(f"# 盘后决策评估 — {eval_date_str}\n")
    lines.append(f"> **决策基准日(T-1)**: {pred_date_str} | **评估日(T)**: {eval_date_str}\n")
    lines.append("---\n")

    # ── 一、买入推荐 TOP5 ──
    lines.append("## 一、T日买入推荐 (TOP 5)\n")
    lines.append(f"> 筛选条件: 预测收益率 > {MIN_PRED_RETURN:.1%}, 置信度 > {MIN_CONFIDENCE:.0%}\n")

    if len(top5_eval) == 0:
        lines.append("> **无股票满足买入条件**\n")
    else:
        lines.append("| # | 代码 | 名称 | 预测收益率 | 置信度 | T日开盘 | T日收盘 | 开→收 | 状态 |")
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
    lines.append("| # | 代码 | 名称 | 预测收益率 | T日开盘 | T日收盘 | 开→收 | 命中 |")
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
    lines.append("## 三、T日市场概况\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|-----:|")
    lines.append(f"| 全市场股票数 | {market['total']:,} |")
    lines.append(f"| 上涨 | {market['up_count']:,} ({market['up_count']/max(market['total'],1):.1%}) |")
    lines.append(f"| 下跌 | {market['down_count']:,} ({market['down_count']/max(market['total'],1):.1%}) |")
    lines.append(f"| 平盘 | {market['flat_count']:,} |")
    lines.append(f"| 市场均值 | {_pct_fmt(market['mean_pct'])} |")
    lines.append(f"| 市场中位数 | {_pct_fmt(market['median_pct'])} |")

    lines.append("\n---\n")
    lines.append("*仅评估首日（T日开→收）表现，完整收益需持有5天后确认。*")

    return "\n".join(lines)


# ── 可视化 ────────────────────────────────────────────────────────

def _color(v: float, nan_color: str = '#888888') -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return nan_color
    return '#E8423E' if v >= 0 else '#1DBB50'  # A股红涨绿跌


def plot_review(top5_eval: pd.DataFrame, top20_eval: pd.DataFrame,
                market: dict, pred_date, eval_date, out_path: str) -> None:
    """生成盘后评估图"""
    pred_date_str = pred_date.date() if hasattr(pred_date, 'date') else pred_date

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

        ax1.bar(x - w/2, pred_vals, w, label='T-1预测收益率',
                color='#5E81F4', alpha=0.9, zorder=3)
        bars2 = ax1.bar(x + w/2, intraday_vals, w, label='T日开→收',
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
    ax1.set_title(f'买入推荐 TOP 5 — 预测 vs 实际  ({eval_date})',
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
    ax2.set_title('T日市场概况', color=text_color, fontsize=10, fontweight='bold')
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

        ax4.bar(x20 - w20/2, pred20, w20, label='T-1预测', color='#5E81F4', alpha=0.9, zorder=3)
        ax4.bar(x20 + w20/2, actual20, w20, label='T日开→收',
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
        f"盘后决策评估  |  T-1: {pred_date_str}  →  T: {eval_date}"
        f"  |  TOP5均值: {_pct_fmt(portfolio_avg)}  [{verdict}]",
        color=title_color, fontsize=13, fontweight='bold', y=0.96
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"评估图已保存: {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────

def run_review(force: bool = False, eval_date: date = None) -> None:
    check_review_time(force, eval_date)

    # 1. 加载预测数据
    pred_df = load_predictions()

    # 2. 确定T-1日和T日
    pred_date, t_day = determine_eval_date(pred_df, eval_date)
    pred_date_str = pred_date.date() if hasattr(pred_date, 'date') else pred_date
    print(f"决策基准日(T-1): {pred_date_str}")
    print(f"评估日(T):       {t_day}")

    # 3. 加载T-1价格信息
    price_df = load_price_info(pred_date)

    # 4. 生成T日决策（基于T-1预测）
    top5, top20, all_pred = generate_decisions(pred_df, pred_date, price_df)
    print(f"买入推荐: {len(top5)} 只, 潜力TOP20: {len(top20)} 只")
    if len(top5) > 0:
        print(f"  TOP5: {top5['代码'].tolist()}")

    # 5. 加载T日实际行情
    t_day_date = t_day if isinstance(t_day, date) else t_day.date()
    try:
        today_df = load_today_data(t_day_date)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(
            f"无法读取T日 {t_day} 的行情数据: {e}\n"
            f"请确认:\n"
            f"  1. T日 {t_day} 已收盘\n"
            f"  2. 已运行数据更新（python src/py00_fetch_stock_data.py --update）"
        )

    # 6. 评估
    top5_eval = evaluate_stocks(top5, today_df) if len(top5) > 0 else pd.DataFrame()
    top20_eval = evaluate_stocks(top20, today_df)
    market = calc_market_stats(today_df)

    # 7. 控制台摘要
    print(f"\n{'='*70}")
    print(f"  T日决策评估  |  T-1: {pred_date_str}  →  T: {t_day}")
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
    md_report = generate_markdown_report(pred_date, t_day, top5_eval, top20_eval, market)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    md_path = os.path.join(OUTPUT_DIR, f"today_review_{t_day_date.strftime('%Y%m%d')}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"评估报告已保存: {md_path}")

    # 9. 生成 PNG 图表
    png_path = os.path.join(OUTPUT_DIR, f"today_review_{t_day_date.strftime('%Y%m%d')}.png")
    plot_review(top5_eval, top20_eval, market, pred_date, t_day, png_path)

    print("\n评估完成。")


if __name__ == '__main__':
    _force = '--force' in sys.argv
    _eval_date = None
    if '--date' in sys.argv:
        _idx = sys.argv.index('--date')
        _eval_date = datetime.strptime(sys.argv[_idx + 1], '%Y-%m-%d').date()
    run_review(_force, eval_date=_eval_date)
