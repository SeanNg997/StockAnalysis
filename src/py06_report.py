"""
py06_report.py — 报告与可视化模块
====================================
职责：
1. 绘制累计收益率曲线（策略 vs 基准）
2. 回撤曲线
3. 每日持仓数量曲线
4. 月度收益热力图
5. 特征重要性
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings

warnings.filterwarnings('ignore')

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FEATURE_PKL = os.path.join(BASE_DIR, 'data', 'features.pkl')

INITIAL_CAPITAL = 100_000


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

    # 基准线（0%）
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

    path = os.path.join(OUTPUT_DIR, 'equity_curve.png')
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

    path = os.path.join(OUTPUT_DIR, 'drawdown_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


MAX_POSITIONS = 5  # 最大持仓数量（对应100%仓位）


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

    path = os.path.join(OUTPUT_DIR, 'daily_positions.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


def plot_monthly_returns(daily_df: pd.DataFrame):
    """绘制月度收益热力图"""
    print("绘制月度收益热力图...")
    daily = daily_df.copy()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['daily_return'] = daily['portfolio_value'].pct_change()
    daily['year'] = daily['date'].dt.year
    daily['month'] = daily['date'].dt.month

    # 月度收益率
    monthly = daily.groupby(['year', 'month'])['daily_return'].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    # A股惯例：红涨绿跌，使用反转的 RdYlGn_r（正收益=红，负收益=绿）
    im = ax.imshow(monthly.values * 100, cmap='RdYlGn_r', aspect='auto', vmin=-10, vmax=10)

    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels([f'{m}月' for m in monthly.columns])
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index)

    # 在每个格子中标注数值
    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            val = monthly.values[i, j] * 100
            if not np.isnan(val) and val != 0:
                ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                        fontsize=9, color='white' if abs(val) >= 4 else 'black')

    plt.colorbar(im, ax=ax, label='月度收益率 (%)')
    ax.set_title('月度收益率热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'monthly_returns.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")


def plot_feature_importance(model_path=None):
    """
    绘制特征重要性（如果模型可用）
    这里使用一个简化方法：读取最新模型的特征重要性
    """
    # 由于walk-forward中模型没有持久化到文件，
    # 我们用一个简化方式：训练一个小模型获取特征重要性
    print("计算特征重要性...")
    try:
        import lightgbm as lgb

        df = pd.read_pickle(FEATURE_PKL)

        # 使用最近2年数据
        recent = df[df['date'] >= '2024-01-01'].copy()

        exclude = {'代码', '名称', 'date', 'open', 'high', 'low', 'close',
                    'volume', 'amount', 'turn', 'pctChg',
                    'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'label',
                    'ma_5', 'ma_10', 'ma_20', 'ma_60'}
        feature_cols = [c for c in df.columns if c not in exclude]

        X = recent[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = recent['label'].dropna()
        valid = y.index.intersection(X.index)
        X = X.loc[valid]
        y = y.loc[valid]

        # 快速训练
        dtrain = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression', 'metric': 'mae',
            'num_leaves': 31, 'learning_rate': 0.1, 'verbose': -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=100)

        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=True).tail(25)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance['feature'], importance['importance'], color='#3498db')
        ax.set_title('特征重要性 (Top 25 - Gain)', fontsize=14, fontweight='bold')
        ax.set_xlabel('重要性 (Gain)')
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  保存: {path}")
    except Exception as e:
        print(f"  特征重要性绘制失败: {e}")


def generate_all_reports():
    """生成所有报告和图表"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    daily_path = os.path.join(OUTPUT_DIR, 'backtest_daily.csv')
    if not os.path.exists(daily_path):
        print("错误: 未找到回测结果文件，请先运行 py04_backtest.py")
        return

    daily_df = pd.read_csv(daily_path)
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    plot_equity_curve(daily_df)
    plot_drawdown(daily_df)
    plot_positions(daily_df)
    plot_monthly_returns(daily_df)
    plot_feature_importance()

    print(f"\n✅ 所有图表已保存至 {OUTPUT_DIR}/")


if __name__ == '__main__':
    generate_all_reports()
