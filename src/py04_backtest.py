"""
py04_backtest.py — 回测引擎模块
=================================
职责：
1. 基于模型预测结果执行完整回测
2. 严格模拟：集合竞价open价成交、T+1、交易成本、持仓≤5只
3. 涨跌停处理
4. 输出回测指标与交易日志
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PKL = os.path.join(BASE_DIR, 'data', 'features.pkl')
PREDICT_PKL = os.path.join(BASE_DIR, 'data', 'predictions.pkl')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ============ 交易成本 ============
BUY_COMMISSION = 0.000085    # 买入手续费 0.0085%
SELL_COMMISSION = 0.000085   # 卖出手续费 0.0085%
STAMP_TAX = 0.0005           # 印花税 0.05%（卖出时收取）

# ============ 策略参数 ============
MAX_POSITIONS = 5            # 最大持仓数
MIN_PRED_RETURN = 0.001      # 最低预测收益率阈值（0.1%）
INITIAL_CAPITAL = 1_000_000  # 初始资金100万
LIMIT_UP_PCT = 0.098         # 涨停判定阈值（接近10%）
LIMIT_DOWN_PCT = -0.098      # 跌停判定阈值


def load_data():
    """加载特征数据和预测结果"""
    print("加载数据...")
    df = pd.read_pickle(FEATURE_PKL)
    pred_df = pd.read_pickle(PREDICT_PKL)

    # 合并预测结果到主数据
    # 只保留原始价格数据和预测
    price_cols = ['代码', '名称', 'date', 'open', 'high', 'low', 'close',
                  'volume', 'amount', 'pctChg']
    price_df = df[price_cols].copy()

    # 合并预测
    pred_merge = pred_df[['date', '代码', 'pred_return', 'pred_std', 'confidence']].copy()
    merged = price_df.merge(pred_merge, on=['date', '代码'], how='inner')
    merged = merged.sort_values(['date', '代码']).reset_index(drop=True)

    print(f"合并后数据: {len(merged):,} 行, {merged['代码'].nunique()} 只股票")
    return merged


def check_limit(row_today, row_prev):
    """
    检查是否涨跌停
    涨停：当日open >= 前日close * 1.098
    跌停：当日open <= 前日close * 0.902
    """
    if row_prev is None:
        return False, False

    prev_close = row_prev['close']
    today_open = row_today['open']

    is_limit_up = today_open >= prev_close * (1 + LIMIT_UP_PCT)
    is_limit_down = today_open <= prev_close * (1 - abs(LIMIT_DOWN_PCT))

    return is_limit_up, is_limit_down


def run_backtest(merged: pd.DataFrame) -> dict:
    """
    执行回测

    核心逻辑：
    - 每个交易日T，在T-1日收盘后用模型打分
    - T日开盘集合竞价执行交易
    - 买入用T日open，卖出用T日open
    - T+1约束：当日买入的股票次日才可卖出
    """
    all_dates = sorted(merged['date'].unique())
    print(f"回测期间: {all_dates[0].date()} ~ {all_dates[-1].date()}, 共 {len(all_dates)} 个交易日")

    # 为了检查涨跌停，需要前日数据
    # 构建 (date, 代码) -> row 的映射
    print("构建数据索引...")
    date_stock_map = {}
    for _, row in merged.iterrows():
        date_stock_map[(row['date'], row['代码'])] = row

    # 构建日期到前一交易日的映射
    prev_date_map = {}
    for i in range(1, len(all_dates)):
        prev_date_map[all_dates[i]] = all_dates[i-1]

    # ============ 回测状态 ============
    cash = INITIAL_CAPITAL
    positions = {}  # {股票代码: {'shares': 股数, 'buy_price': 买入价, 'buy_date': 买入日期}}
    locked_stocks = set()  # 当日买入锁定（T+1）

    # 记录
    daily_records = []    # 每日资产记录
    trade_log = []        # 交易日志

    for day_idx, current_date in enumerate(all_dates):
        # 获取当日所有股票数据
        day_data = merged[merged['date'] == current_date]
        prev_date = prev_date_map.get(current_date)

        # ===== 步骤1：解锁昨日买入的股票 =====
        locked_stocks.clear()

        # ===== 步骤2：计算当前持仓市值 =====
        portfolio_value = cash
        positions_to_remove = []

        for code, pos in positions.items():
            key = (current_date, code)
            if key in date_stock_map:
                row = date_stock_map[key]
                pos['current_price'] = row['open']  # 以当日open估值
                portfolio_value += pos['shares'] * row['open']
            else:
                # 股票停牌，用上一个已知价格
                portfolio_value += pos['shares'] * pos.get('current_price', pos['buy_price'])

        # ===== 步骤3：决定卖出 =====
        # 获取当日持仓股的预测
        sell_list = []
        for code, pos in positions.items():
            day_pred = day_data[day_data['代码'] == code]
            if len(day_pred) == 0:
                continue  # 停牌，不能交易

            pred_ret = day_pred.iloc[0]['pred_return']
            row_today = day_pred.iloc[0]

            # 检查跌停（跌停不能卖出）
            prev_key = (prev_date, code) if prev_date else None
            row_prev = date_stock_map.get(prev_key) if prev_key else None
            _, is_limit_down = check_limit(row_today, row_prev)

            if is_limit_down:
                continue  # 跌停，不能卖出

            # 卖出条件：预测收益率 < 0 或 不在当日top5中
            if pred_ret < 0:
                sell_list.append(code)

        # 如果持仓股不在当日top选股名单中，也考虑卖出
        # 先获取当日选股排名
        day_ranked = day_data[day_data['pred_return'] > MIN_PRED_RETURN].sort_values(
            'pred_return', ascending=False
        )

        # 当日top候选（排除涨停）
        top_candidates = []
        for _, cand in day_ranked.iterrows():
            code = cand['代码']
            prev_key = (prev_date, code) if prev_date else None
            row_prev = date_stock_map.get(prev_key) if prev_key else None
            is_limit_up, _ = check_limit(cand, row_prev)
            if not is_limit_up:
                top_candidates.append(code)
            if len(top_candidates) >= MAX_POSITIONS * 2:
                break

        top_set = set(top_candidates[:MAX_POSITIONS])

        # 持仓中不在top名单的也卖出
        for code in list(positions.keys()):
            if code not in top_set and code not in sell_list:
                day_pred = day_data[day_data['代码'] == code]
                if len(day_pred) == 0:
                    continue
                row_today = day_pred.iloc[0]
                prev_key = (prev_date, code) if prev_date else None
                row_prev = date_stock_map.get(prev_key) if prev_key else None
                _, is_limit_down = check_limit(row_today, row_prev)
                if not is_limit_down:
                    sell_list.append(code)

        sell_list = list(set(sell_list))

        # 执行卖出
        for code in sell_list:
            if code not in positions:
                continue
            day_info = day_data[day_data['代码'] == code]
            if len(day_info) == 0:
                continue

            pos = positions[code]
            sell_price = day_info.iloc[0]['open']
            sell_amount = pos['shares'] * sell_price
            sell_cost = sell_amount * (SELL_COMMISSION + STAMP_TAX)
            cash += sell_amount - sell_cost

            trade_log.append({
                'date': current_date,
                '代码': code,
                '名称': day_info.iloc[0]['名称'],
                'action': 'SELL',
                'price': sell_price,
                'shares': pos['shares'],
                'amount': sell_amount,
                'cost': sell_cost,
                'profit': sell_amount - sell_cost - pos['shares'] * pos['buy_price'] * (1 + BUY_COMMISSION),
            })

            del positions[code]

        # ===== 步骤4：决定买入 =====
        n_empty = MAX_POSITIONS - len(positions)
        if n_empty > 0 and len(top_candidates) > 0:
            # 过滤掉已持仓的
            buy_candidates = [c for c in top_candidates if c not in positions][:n_empty]

            if buy_candidates:
                # 计算当前可用资金
                available_cash = cash

                # 基于预测收益率分配仓位
                buy_info = day_data[day_data['代码'].isin(buy_candidates)].copy()
                buy_info = buy_info.sort_values('pred_return', ascending=False)

                # 等权分配（简单且稳健）
                n_buy = min(len(buy_info), n_empty)
                per_stock_cash = available_cash / max(n_buy, 1) * 0.98  # 预留2%现金

                for _, cand in buy_info.head(n_buy).iterrows():
                    code = cand['代码']
                    buy_price = cand['open']

                    if buy_price <= 0:
                        continue

                    # 计算可买股数（100股为单位）
                    shares = int(per_stock_cash / buy_price / 100) * 100
                    if shares < 100:
                        continue

                    buy_amount = shares * buy_price
                    buy_cost = buy_amount * BUY_COMMISSION
                    cash -= buy_amount + buy_cost

                    positions[code] = {
                        'shares': shares,
                        'buy_price': buy_price,
                        'buy_date': current_date,
                        'current_price': buy_price,
                    }
                    locked_stocks.add(code)

                    trade_log.append({
                        'date': current_date,
                        '代码': code,
                        '名称': cand['名称'],
                        'action': 'BUY',
                        'price': buy_price,
                        'shares': shares,
                        'amount': buy_amount,
                        'cost': buy_cost,
                        'profit': 0,
                    })

        # ===== 步骤5：记录当日资产 =====
        total_value = cash
        for code, pos in positions.items():
            key = (current_date, code)
            if key in date_stock_map:
                current_price = date_stock_map[key]['close']  # 用收盘价估值
                pos['current_price'] = current_price
            else:
                current_price = pos['current_price']
            total_value += pos['shares'] * current_price

        daily_records.append({
            'date': current_date,
            'cash': cash,
            'portfolio_value': total_value,
            'n_positions': len(positions),
            'n_trades': sum(1 for t in trade_log if t['date'] == current_date),
        })

        # 定期打印进度
        if day_idx % 100 == 0:
            ret = (total_value / INITIAL_CAPITAL - 1) * 100
            print(f"  [{current_date.date()}] 资产: {total_value:,.0f}, "
                  f"收益: {ret:+.2f}%, 持仓: {len(positions)}只")

    # ===== 转换为DataFrame =====
    daily_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_log)

    return {
        'daily': daily_df,
        'trades': trade_df,
        'final_value': daily_df.iloc[-1]['portfolio_value'],
        'positions': positions,
    }


def compute_metrics(daily_df: pd.DataFrame) -> dict:
    """计算回测指标"""
    daily_df = daily_df.copy()
    daily_df['daily_return'] = daily_df['portfolio_value'].pct_change()

    total_days = len(daily_df)
    total_return = daily_df.iloc[-1]['portfolio_value'] / INITIAL_CAPITAL - 1
    annual_return = (1 + total_return) ** (252 / total_days) - 1

    # 夏普比率（无风险利率2.5%）
    rf_daily = 0.025 / 252
    excess_returns = daily_df['daily_return'].dropna() - rf_daily
    sharpe = excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(252)

    # 最大回撤
    cummax = daily_df['portfolio_value'].cummax()
    drawdown = (daily_df['portfolio_value'] - cummax) / cummax
    max_drawdown = drawdown.min()

    # 年化波动率
    annual_vol = daily_df['daily_return'].dropna().std() * np.sqrt(252)

    # Calmar比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        '初始资金': f"{INITIAL_CAPITAL:,.0f}",
        '期末资产': f"{daily_df.iloc[-1]['portfolio_value']:,.0f}",
        '总回报率': f"{total_return:.2%}",
        '年化收益率': f"{annual_return:.2%}",
        '夏普比率': f"{sharpe:.3f}",
        '最大回撤': f"{max_drawdown:.2%}",
        '年化波动率': f"{annual_vol:.2%}",
        'Calmar比率': f"{calmar:.3f}",
        '交易天数': total_days,
    }


def compute_trade_metrics(trade_df: pd.DataFrame) -> dict:
    """计算交易统计"""
    if len(trade_df) == 0:
        return {}

    sells = trade_df[trade_df['action'] == 'SELL']
    if len(sells) == 0:
        return {'总交易笔数': len(trade_df)}

    wins = sells[sells['profit'] > 0]
    losses = sells[sells['profit'] <= 0]

    win_rate = len(wins) / len(sells) if len(sells) > 0 else 0
    avg_win = wins['profit'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['profit'].mean()) if len(losses) > 0 else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    total_days = (trade_df['date'].max() - trade_df['date'].min()).days
    buys = trade_df[trade_df['action'] == 'BUY']

    return {
        '总交易笔数(买入)': len(buys),
        '总交易笔数(卖出)': len(sells),
        '胜率': f"{win_rate:.2%}",
        '盈亏比': f"{profit_loss_ratio:.3f}",
        '平均盈利': f"{avg_win:,.0f}",
        '平均亏损': f"{-losses['profit'].mean() if len(losses) > 0 else 0:,.0f}",
        '日均交易次数': f"{len(trade_df) / max(total_days, 1) * 365 / 252:.2f}",
    }


def run_pipeline():
    """执行回测流水线"""
    merged = load_data()
    results = run_backtest(merged)

    daily_df = results['daily']
    trade_df = results['trades']

    # 计算指标
    metrics = compute_metrics(daily_df)
    trade_metrics = compute_trade_metrics(trade_df)

    # 打印指标
    print("\n" + "=" * 50)
    print("回测结果汇总")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("-" * 50)
    for k, v in trade_metrics.items():
        print(f"  {k}: {v}")

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    daily_df.to_csv(os.path.join(OUTPUT_DIR, 'backtest_daily.csv'), index=False)
    trade_df.to_csv(os.path.join(OUTPUT_DIR, 'trade_log.csv'), index=False)

    # 保存指标
    with open(os.path.join(OUTPUT_DIR, 'backtest_metrics.txt'), 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("A股量化策略回测报告\n")
        f.write(f"回测期间: {daily_df.iloc[0]['date'].date()} ~ {daily_df.iloc[-1]['date'].date()}\n")
        f.write("=" * 50 + "\n\n")
        f.write("【资产表现】\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n【交易统计】\n")
        for k, v in trade_metrics.items():
            f.write(f"  {k}: {v}\n")

    print(f"\n✅ 结果已保存至 {OUTPUT_DIR}/")

    return results, metrics, trade_metrics


if __name__ == '__main__':
    run_pipeline()
