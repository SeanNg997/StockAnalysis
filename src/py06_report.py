"""报告与可视化 — 净值曲线 / 回撤 / 月度热力图 / 特征重要性 / 交易分析"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import os
import warnings

from config import CONFIG

warnings.filterwarnings('ignore')

PREFERRED_CJK_FONTS = [
    'Hiragino Sans GB',
    'PingFang SC',
    'STHeiti',
    'Songti SC',
    'Arial Unicode MS',
    'WenQuanYi Micro Hei',
    'Noto Sans CJK SC',
    'Noto Sans Mono CJK SC',
    'SimHei',
    'Microsoft YaHei',
]


def _configure_matplotlib_fonts():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [font for font in PREFERRED_CJK_FONTS if font in available_fonts]

    # 动态选择本机存在的中文字体，避免反复触发 findfont 告警。
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = selected_fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if selected_fonts:
        print(f"使用中文字体: {selected_fonts[0]}")
    else:
        print("未找到本机中文字体，回退到 DejaVu Sans")


_configure_matplotlib_fonts()

BASE_DIR = CONFIG['paths']['BASE_DIR']
BACKTEST_OUTPUT_DIR = CONFIG['paths']['BACKTEST_OUTPUT_DIR']
FEATURE_PKL = CONFIG['paths']['FEATURE_PKL']
CLEAN_PKL = CONFIG['paths']['CLEAN_PKL']

INITIAL_CAPITAL = CONFIG['backtest']['INITIAL_CAPITAL']


def plot_equity_curve(daily_df: pd.DataFrame):
    """绘制累计收益率曲线"""
    print("绘制收益率曲线...")
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(daily_df['date'])
    strategy_pct = (daily_df['portfolio_value'] / INITIAL_CAPITAL - 1) * 100

    ax.plot(dates, strategy_pct, label='策略收益率', color='#e74c3c', linewidth=1.5)
    ax.fill_between(dates, strategy_pct, 0,
                    where=(strategy_pct >= 0), color='#e74c3c', alpha=0.1)
    ax.fill_between(dates, strategy_pct, 0,
                    where=(strategy_pct < 0), color='#2ecc71', alpha=0.1)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='基准(0%)')

    ax.set_title('A股量化策略 — 累计收益率曲线', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('累计收益率 (%)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(BACKTEST_OUTPUT_DIR, 'equity_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


def plot_drawdown(daily_df: pd.DataFrame):
    """绘制回撤曲线"""
    print("绘制回撤曲线...")
    fig, ax = plt.subplots(figsize=(14, 4))

    dates = pd.to_datetime(daily_df['date'])
    cummax = daily_df['portfolio_value'].cummax()
    drawdown = (daily_df['portfolio_value'] - cummax) / cummax * 100

    ax.fill_between(dates, drawdown, 0, color='#e74c3c', alpha=0.4)
    ax.plot(dates, drawdown, color='#e74c3c', linewidth=0.8)

    ax.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('回撤 (%)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(BACKTEST_OUTPUT_DIR, 'drawdown_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


MAX_POSITIONS = CONFIG['backtest']['MAX_POSITIONS']  # 最大持仓数量（对应100%仓位）


def plot_positions(daily_df: pd.DataFrame):
    """绘制每日仓位比例与交易次数"""
    print("绘制持仓数量曲线...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    dates = pd.to_datetime(daily_df['date'])
    position_pct = daily_df['n_positions'] / MAX_POSITIONS * 100

    # 仓位百分比
    axes[0].bar(dates, position_pct, color='#3498db', alpha=0.7, width=1)
    axes[0].set_ylabel('仓位比例 (%)')
    axes[0].set_title('每日仓位比例与交易次数', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 110)
    axes[0].axhline(y=100, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1, label='满仓(100%)')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 交易次数
    axes[1].bar(dates, daily_df['n_trades'], color='#e67e22', alpha=0.7, width=1)
    axes[1].set_ylabel('交易次数')
    axes[1].set_xlabel('日期')
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(BACKTEST_OUTPUT_DIR, 'daily_positions.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


def plot_monthly_returns(daily_df: pd.DataFrame):
    """绘制月度收益热力图"""
    print("绘制月度收益热力图...")
    # 避免不必要的复制
    daily = daily_df.copy()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['daily_return'] = daily['portfolio_value'].pct_change()

    monthly = daily.groupby([daily['date'].dt.year, daily['date'].dt.month])['daily_return'].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(monthly.values * 100, cmap='RdYlGn_r', aspect='auto', vmin=-10, vmax=10)

    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels([f'{m}月' for m in monthly.columns])
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index)

    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            val = monthly.values[i, j] * 100
            if not np.isnan(val) and val != 0:
                ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                        fontsize=9, color='white' if abs(val) >= 4 else 'black')

    plt.colorbar(im, ax=ax, label='月度收益率 (%)')
    ax.set_title('月度收益率热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(BACKTEST_OUTPUT_DIR, 'monthly_returns.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


def plot_feature_importance(model_path=None):
    """绘制特征重要性（快速训练一个小模型获取）"""
    print("计算特征重要性...")
    try:
        import lightgbm as lgb

        df = pd.read_pickle(FEATURE_PKL)
        recent = df[df['date'] >= '2024-01-01'].copy()

        exclude = {'code', 'name', 'date', 'open', 'high', 'low', 'close', 'preclose',
                    'volume', 'amount', 'turn', 'pctChg',
                    'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'label',
                    'isST', 'isTrading',
                    'ma_5', 'ma_10', 'ma_20', 'ma_60', 'pt_adjust_factor'}
        candidate_cols = [c for c in recent.columns if c not in exclude]
        numeric_cols = recent[candidate_cols].select_dtypes(include=[np.number, 'bool']).columns.tolist()
        skipped_cols = [c for c in candidate_cols if c not in numeric_cols]

        if skipped_cols:
            print(f"  跳过非数值特征: {', '.join(skipped_cols)}")

        if not numeric_cols:
            print("  无可用数值特征，跳过")
            return

        valid_rows = recent['label'].notna()
        X = recent.loc[valid_rows, numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = recent.loc[valid_rows, 'label']

        if X.empty or y.empty:
            print("  缺少可训练样本，跳过")
            return

        dtrain = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression', 'metric': 'mae',
            'num_leaves': 31, 'learning_rate': 0.1, 'verbose': -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=100)

        importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=True).tail(25)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance['feature'], importance['importance'], color='#3498db')
        ax.set_title('特征重要性 (Top 25 - Gain)', fontsize=14, fontweight='bold')
        ax.set_xlabel('重要性 (Gain)')
        plt.tight_layout()

        path = os.path.join(BACKTEST_OUTPUT_DIR, 'feature_importance.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  保存: {path}")
    except Exception as e:
        print(f"  特征重要性绘制失败: {e}")


def _load_trade_features():
    """加载卖出交易并关联买入日股票特征"""
    trade_path = os.path.join(BACKTEST_OUTPUT_DIR, 'trade_log.csv')
    trade_df = pd.read_csv(trade_path, parse_dates=['date'])
    trade_df = trade_df.reset_index(drop=False).rename(columns={'index': '_seq'})
    sells = trade_df[trade_df['action'] == 'SELL'].copy()
    if sells.empty:
        return None

    price_df = pd.read_pickle(CLEAN_PKL)
    price_df['date'] = pd.to_datetime(price_df['date'])

    # 按交易流水为每笔 SELL 绑定其对应 BUY（同 code 的先买先卖）。
    open_buys = {}
    sell_buy_pairs = []
    ordered_trades = trade_df.sort_values(['date', '_seq']).reset_index(drop=True)
    for _, row in ordered_trades.iterrows():
        code = row['code']
        action = row['action']
        if action == 'BUY':
            open_buys.setdefault(code, []).append(row['date'])
            continue
        if action != 'SELL':
            continue

        queue = open_buys.get(code, [])
        buy_date = queue.pop(0) if queue else pd.NaT
        if not queue and code in open_buys:
            del open_buys[code]
        sell_buy_pairs.append({'_seq': row['_seq'], 'buy_date': buy_date})

    buy_map = pd.DataFrame(sell_buy_pairs, columns=['_seq', 'buy_date'])
    sell_with_buy = sells.merge(buy_map, on='_seq', how='left')
    sell_with_buy['buy_date_approx'] = sell_with_buy['date'] - pd.to_timedelta(
        sell_with_buy['hold_days'].clip(lower=1), unit='D'
    )
    sell_with_buy['buy_date_final'] = sell_with_buy['buy_date'].fillna(
        sell_with_buy['buy_date_approx']
    )

    # 向量化：用 merge_asof 匹配买入日最近的交易日数据
    buy_day_info = sell_with_buy[['code', 'buy_date_final']].drop_duplicates()
    buy_day_info = buy_day_info.rename(columns={'buy_date_final': 'date'})

    # merge_asof 要求 on 列全局有序，按 code 分组处理
    price_subset = price_df[['code', 'date', 'turn', 'amount', 'peTTM', 'pbMRQ', 'close']].copy()
    merged_parts = []
    for code, grp in buy_day_info.groupby('code'):
        grp_sorted = grp.sort_values('date')
        price_code = price_subset[price_subset['code'] == code][['date', 'turn', 'amount', 'peTTM', 'pbMRQ', 'close']].sort_values('date')
        part = pd.merge_asof(grp_sorted, price_code, on='date', direction='backward')
        merged_parts.append(part)
    merged_buy = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()
    merged_buy = merged_buy.rename(columns={'date': 'buy_date_final'})

    # 计算20日波动率（向量化）
    vol_df = price_df.sort_values(['code', 'date']).copy()
    vol_df['vol_20'] = vol_df.groupby('code')['pctChg'].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    vol_subset = vol_df[['code', 'date', 'vol_20']].copy()
    vol_subset = vol_subset.rename(columns={'date': 'buy_date_final', 'vol_20': 'f_vol_20'})

    # merge_asof 要求 on 列全局有序，按 code 分组处理
    vol_parts = []
    for code, grp in merged_buy.groupby('code'):
        grp_sorted = grp.sort_values('buy_date_final')
        vol_code = vol_subset[vol_subset['code'] == code].sort_values('buy_date_final')
        part = pd.merge_asof(grp_sorted, vol_code[['buy_date_final', 'f_vol_20']], on='buy_date_final', direction='backward')
        vol_parts.append(part)
    merged_buy = pd.concat(vol_parts, ignore_index=True) if vol_parts else pd.DataFrame()

    merged_buy = merged_buy.rename(columns={
        'turn': 'f_turn', 'amount': 'f_amount',
        'peTTM': 'f_peTTM', 'pbMRQ': 'f_pbMRQ', 'close': 'f_close'
    })

    result = sell_with_buy.merge(
        merged_buy[['code', 'buy_date_final', 'f_turn', 'f_amount', 'f_peTTM', 'f_pbMRQ', 'f_close', 'f_vol_20']],
        on=['code', 'buy_date_final'], how='left'
    )
    return result


def _plot_group_analysis(data, col, bins, labels, title, filename):
    """通用分组分析绘图"""
    data = data.copy()
    data['group'] = pd.cut(data[col], bins=bins, labels=labels, include_lowest=True)
    data = data.dropna(subset=['group', 'profit_pct'])

    grouped = data.groupby('group', observed=True).agg(
        count=('profit_pct', 'size'),
        win_rate=('profit_pct', lambda x: (x > 0).mean()),
        avg_return=('profit_pct', 'mean'),
        avg_profit=('profit', 'mean'),
    ).reset_index()

    if grouped.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    bars = axes[0].bar(grouped['group'].astype(str), grouped['win_rate'] * 100,
                       color=colors[:len(grouped)])
    axes[0].set_ylabel('胜率 (%)')
    axes[0].set_title('胜率')
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, grouped['win_rate']):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{v:.0%}', ha='center', va='bottom', fontsize=9)

    bar_colors = ['#e74c3c' if v >= 0 else '#2ecc71' for v in grouped['avg_return']]
    bars = axes[1].bar(grouped['group'].astype(str), grouped['avg_return'] * 100,
                       color=bar_colors)
    axes[1].set_ylabel('平均收益率 (%)')
    axes[1].set_title('平均收益率')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, grouped['avg_return']):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.1 if v >= 0 else -0.3),
                     f'{v:+.2%}', ha='center', va='bottom', fontsize=9)

    bars = axes[2].bar(grouped['group'].astype(str), grouped['count'],
                       color=colors[:len(grouped)])
    axes[2].set_ylabel('交易笔数')
    axes[2].set_title('交易笔数')
    for bar, v in zip(bars, grouped['count']):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(int(v)), ha='center', va='bottom', fontsize=9)

    for ax in axes:
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(BACKTEST_OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


def plot_trade_analysis():
    """交易分析：按多维度分析模型表现"""
    print("生成交易分析图表...")
    data = _load_trade_features()
    if data is None or data.empty:
        print("  无卖出交易数据，跳过")
        return

    # 按成交额分组
    if 'f_amount' in data.columns and data['f_amount'].notna().sum() > 10:
        _plot_group_analysis(
            data, 'f_amount',
            bins=[0, 5e7, 1.5e8, 5e8, np.inf],
            labels=['<5千万', '5千万~1.5亿', '1.5~5亿', '>5亿'],
            title='按成交额分组 · 交易表现',
            filename='analysis_by_amount.png'
        )

    # 按换手率分组
    if 'f_turn' in data.columns and data['f_turn'].notna().sum() > 10:
        _plot_group_analysis(
            data, 'f_turn',
            bins=[0, 1, 3, 8, np.inf],
            labels=['<1%', '1~3%', '3~8%', '>8%'],
            title='按换手率分组 · 交易表现',
            filename='analysis_by_turnover.png'
        )

    # 按20日波动率分组
    if 'f_vol_20' in data.columns and data['f_vol_20'].notna().sum() > 10:
        _plot_group_analysis(
            data, 'f_vol_20',
            bins=[0, 1.5, 2.5, 4, np.inf],
            labels=['低波(<1.5%)', '中波(1.5~2.5%)', '中高波(2.5~4%)', '高波(>4%)'],
            title='按20日波动率分组 · 交易表现',
            filename='analysis_by_volatility.png'
        )

    # 按PE分组
    if 'f_peTTM' in data.columns:
        pe_valid = data[(data['f_peTTM'] > 0) & (data['f_peTTM'] < 200)].copy()
        if len(pe_valid) > 10:
            _plot_group_analysis(
                pe_valid, 'f_peTTM',
                bins=[0, 15, 30, 60, 200],
                labels=['低估值(<15)', '合理(15~30)', '偏高(30~60)', '高估值(>60)'],
                title='按PE(TTM)分组 · 交易表现',
                filename='analysis_by_pe.png'
            )

    # 按卖出原因
    _plot_sell_reason_analysis(data)

    # 综合统计
    _save_analysis_summary(data)
    print("  交易分析完成")


def _plot_sell_reason_analysis(data):
    """按卖出原因分析"""
    if 'reason' not in data.columns:
        return

    grouped = data.groupby('reason').agg(
        count=('profit_pct', 'size'),
        win_rate=('profit_pct', lambda x: (x > 0).mean()),
        avg_return=('profit_pct', 'mean'),
        total_profit=('profit', 'sum'),
    ).sort_values('count', ascending=False).reset_index()

    if grouped.empty:
        return

    REASON_CN = {
        'STOP_LOSS': '止损', 'TAKE_PROFIT': '止盈',
        'HOLD_EXPIRE': '到期', 'SIGNAL_REVERSE': '信号反转',
    }
    grouped['reason_cn'] = grouped['reason'].map(REASON_CN).fillna(grouped['reason'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('按卖出原因 · 交易表现', fontsize=14, fontweight='bold')

    colors = {'止损': '#2ecc71', '止盈': '#e74c3c', '到期': '#3498db', '信号反转': '#f39c12'}
    bar_colors = [colors.get(r, '#95a5a6') for r in grouped['reason_cn']]

    axes[0].pie(grouped['count'], labels=grouped['reason_cn'], autopct='%1.0f%%',
                colors=bar_colors, startangle=90)
    axes[0].set_title('卖出原因分布')

    bars = axes[1].bar(grouped['reason_cn'], grouped['win_rate'] * 100, color=bar_colors)
    axes[1].set_ylabel('胜率 (%)')
    axes[1].set_title('各原因胜率')
    axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, grouped['win_rate']):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{v:.0%}', ha='center', va='bottom', fontsize=9)

    profit_colors = ['#e74c3c' if v >= 0 else '#2ecc71' for v in grouped['total_profit']]
    bars = axes[2].bar(grouped['reason_cn'], grouped['total_profit'], color=profit_colors)
    axes[2].set_ylabel('总盈亏 (元)')
    axes[2].set_title('各原因总盈亏')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, grouped['total_profit']):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (50 if v >= 0 else -150),
                     f'{v:+,.0f}', ha='center', va='bottom', fontsize=9)

    for ax in axes[1:]:
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(BACKTEST_OUTPUT_DIR, 'analysis_by_sell_reason.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


def _save_analysis_summary(data):
    """保存交易分析汇总 Markdown"""
    lines = ["# 交易分析报告\n"]
    lines.append(f"> 分析样本：**{len(data)}** 笔卖出交易\n")
    lines.append("---\n")

    overall_wr = (data['profit_pct'] > 0).mean()
    overall_avg = data['profit_pct'].mean()
    lines.append("## 总体表现\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|-----:|")
    lines.append(f"| 总卖出笔数 | {len(data)} |")
    lines.append(f"| 总体胜率 | {overall_wr:.1%} |")
    lines.append(f"| 平均收益率 | {overall_avg:+.2%} |")
    lines.append(f"| 总盈亏 | ¥{data['profit'].sum():+,.0f} |\n")

    # 按维度输出最佳/最差组
    dims = [
        ('f_amount', [0, 5e7, 1.5e8, 5e8, np.inf],
         ['<5千万', '5千万~1.5亿', '1.5~5亿', '>5亿'], '成交额'),
        ('f_turn', [0, 1, 3, 8, np.inf],
         ['<1%', '1~3%', '3~8%', '>8%'], '换手率'),
        ('f_vol_20', [0, 1.5, 2.5, 4, np.inf],
         ['低波(<1.5%)', '中波(1.5~2.5%)', '中高波(2.5~4%)', '高波(>4%)'], '20日波动率'),
    ]

    lines.append("## 各维度最佳表现分组\n")
    lines.append("| 维度 | 最佳分组 | 胜率 | 平均收益率 | 笔数 |")
    lines.append("|------|:--------:|-----:|-----------:|-----:|")

    for col, bins, labels, dim_name in dims:
        if col not in data.columns or data[col].notna().sum() < 10:
            continue
        tmp = data.copy()
        tmp['group'] = pd.cut(tmp[col], bins=bins, labels=labels, include_lowest=True)
        tmp = tmp.dropna(subset=['group', 'profit_pct'])
        grp = tmp.groupby('group', observed=True).agg(
            wr=('profit_pct', lambda x: (x > 0).mean()),
            avg=('profit_pct', 'mean'),
            cnt=('profit_pct', 'size'),
        )
        grp = grp[grp['cnt'] >= 5]  # 至少5笔才有统计意义
        if grp.empty:
            continue
        best = grp.loc[grp['avg'].idxmax()]
        lines.append(f"| {dim_name} | {best.name} | {best['wr']:.0%}"
                     f" | {best['avg']:+.2%} | {int(best['cnt'])} |")

    lines.append("\n---\n")
    lines.append("> *样本量较小的分组结论可能不稳定，建议结合样本外数据验证。*\n")

    md_path = os.path.join(BACKTEST_OUTPUT_DIR, 'trade_analysis.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  保存: {md_path}")


def generate_all_reports():
    """生成所有报告和图表"""
    os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)

    daily_path = os.path.join(BACKTEST_OUTPUT_DIR, 'backtest_daily.csv')
    if not os.path.exists(daily_path):
        print("错误: 未找到回测结果文件，请先运行 py05_backtest.py")
        return

    daily_df = pd.read_csv(daily_path)
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    plot_equity_curve(daily_df)
    plot_drawdown(daily_df)
    plot_positions(daily_df)
    plot_monthly_returns(daily_df)
    plot_feature_importance()
    plot_trade_analysis()

    print(f"\n✅ 所有图表已保存至 {BACKTEST_OUTPUT_DIR}/")


if __name__ == '__main__':
    generate_all_reports()
