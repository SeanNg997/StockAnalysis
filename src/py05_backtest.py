"""回测引擎 — 基于预测结果的完整回测模拟"""

import pandas as pd
import numpy as np
import os
import warnings

from config import CONFIG
import trading_rules as rules

warnings.filterwarnings('ignore')

CLEAN_PKL = CONFIG['paths']['CLEAN_PKL']
PREDICT_PKL = CONFIG['paths']['PREDICT_PKL']
MARKET_PKL = CONFIG['paths']['MARKET_PKL']
OUTPUT_DIR = CONFIG['paths']['BACKTEST_OUTPUT_DIR']
INITIAL_CAPITAL = CONFIG['backtest']['INITIAL_CAPITAL']


def load_data():
    """加载回测所需数据（价格、预测、市场状态）"""
    print("加载数据...")

    price_cols = ['code', 'name', 'date', 'open', 'high', 'low', 'close',
                  'volume', 'amount', 'pctChg']
    df = pd.read_pickle(CLEAN_PKL)
    price_df = df[price_cols].copy()
    del df

    pred_df = pd.read_pickle(PREDICT_PKL)
    pred_merge = pred_df[['date', 'code', 'pred_return', 'pred_std', 'confidence']].copy()

    merged = price_df.merge(pred_merge, on=['date', 'code'], how='inner')
    merged = merged.sort_values(['date', 'code']).reset_index(drop=True)

    market_df = pd.read_pickle(MARKET_PKL)

    print(f"合并后数据: {len(merged):,} 行, {merged['code'].nunique()} 只股票")
    print(f"市场状态: {len(market_df):,} 行")
    return merged, market_df


def run_backtest(merged: pd.DataFrame, market_df: pd.DataFrame,
                 scoring_method='return_only') -> dict:
    """执行回测：T日决策 → T+1日集合竞价成交"""
    all_dates = sorted(merged['date'].unique())
    print(f"回测期间: {all_dates[0].date()} ~ {all_dates[-1].date()}, 共 {len(all_dates)} 个交易日")

    print("构建数据索引...")
    indexed = merged.set_index(['date', 'code'])
    date_grouped = merged.groupby('date')

    market_indexed = market_df.set_index(['date', 'code'])

    market_grouped = market_df.groupby('date')

    next_date_map = {}
    for i in range(len(all_dates) - 1):
        next_date_map[all_dates[i]] = all_dates[i + 1]

    daily_mkt_ret = merged.groupby('date')['pctChg'].mean().sort_index()

    cash = INITIAL_CAPITAL
    positions = {}
    daily_records = []
    trade_log = []
    position_log = []
    n_trades_today = 0

    daily_records.append({
        'date': all_dates[0],
        'cash': INITIAL_CAPITAL,
        'portfolio_value': INITIAL_CAPITAL,
        'n_positions': 0,
        'n_trades': 0,
    })

    for day_idx in range(len(all_dates)):
        decision_date = all_dates[day_idx]
        exec_date = next_date_map.get(decision_date)

        try:
            decision_data = date_grouped.get_group(decision_date)
        except KeyError:
            decision_data = pd.DataFrame()

        if exec_date is None:
            total_value = cash
            for code, pos in positions.items():
                try:
                    current_price = indexed.loc[(decision_date, code), 'close']
                    pos['current_price'] = current_price
                except KeyError:
                    current_price = pos['current_price']
                total_value += pos['shares'] * current_price

                position_log.append(_build_position_record(
                    decision_date, code, pos, indexed, decision_date
                ))

            if len(daily_records) == 0 or daily_records[-1]['date'] != decision_date:
                daily_records.append({
                    'date': decision_date,
                    'cash': cash,
                    'portfolio_value': total_value,
                    'n_positions': len(positions),
                    'n_trades': 0,
                })
            continue

        # 步骤1: 更新持有天数
        for code in positions:
            positions[code]['hold_days'] += 1
        n_trades_today = 0

        # 步骤2: 卖出决策（T日盘后）
        sell_list = rules.decide_sells(positions, decision_data)

        # 步骤3: 执行卖出（T+1日验证）
        actually_sold = set()
        for code, sell_reason in sell_list:
            if code not in positions:
                continue
            pos = positions[code]

            try:
                prev_close = indexed.loc[(decision_date, code), 'close']
            except KeyError:
                # DELIST_FORCE_SELL 时 decision_date 可能无数据，用买入价兜底
                if sell_reason == 'DELIST_FORCE_SELL':
                    prev_close = pos['buy_price']
                else:
                    continue

            can_sell, fail_reason, t1_open = rules.validate_sell_execution(
                code, market_indexed, exec_date, prev_close, sell_reason
            )

            stock_name = _get_stock_name(indexed, market_indexed, exec_date, decision_date, code)

            if not can_sell:
                trade_log.append({
                    'date': exec_date, 'code': code, 'name': stock_name,
                    'action': f'SELL_FAILED_{fail_reason}',
                    'price': 0, 'shares': pos['shares'],
                    'amount': 0, 'cost': 0, 'profit': 0,
                    'profit_pct': np.nan,
                    'reason': f'{sell_reason}_BLOCKED_{fail_reason}',
                    'hold_days': pos['hold_days'],
                })
                continue

            sell_amount = pos['shares'] * t1_open
            sell_cost = rules.calc_sell_cost(sell_amount) if t1_open > 0 else 0
            cash += sell_amount - sell_cost

            buy_cost_total = pos['shares'] * pos['buy_price'] + pos['buy_cost']
            profit = sell_amount - sell_cost - buy_cost_total
            profit_pct_val = profit / buy_cost_total if buy_cost_total > 0 else 0

            trade_log.append({
                'date': exec_date, 'code': code, 'name': stock_name,
                'action': 'SELL', 'price': t1_open,
                'shares': pos['shares'], 'amount': sell_amount,
                'cost': sell_cost, 'profit': profit,
                'profit_pct': profit_pct_val,
                'reason': sell_reason, 'hold_days': pos['hold_days'],
            })
            n_trades_today += 1
            actually_sold.add(code)
            del positions[code]

        # 步骤4: 买入决策（T日盘后）
        mkt_factor = rules.check_market_regime(daily_mkt_ret, decision_date)
        n_slots = rules.compute_buy_slots(len(positions), mkt_factor)

        if n_slots > 0 and len(decision_data) > 0:
            try:
                market_day = market_grouped.get_group(decision_date)
            except KeyError:
                market_day = pd.DataFrame()

            if len(market_day) > 0:
                pool = rules.filter_stock_pool(decision_data, market_day, market_history=market_df)
                buy_codes = rules.select_buys(
                    pool, set(positions.keys()), actually_sold,
                    n_slots, scoring_method
                )

                # 步骤5: 执行买入（T+1日验证）
                if buy_codes:
                    remaining_slots = len(buy_codes)
                    available_cash = cash * 0.95

                    for code in buy_codes:
                        if remaining_slots <= 0:
                            break

                        try:
                            prev_close = indexed.loc[(decision_date, code), 'close']
                        except KeyError:
                            prev_close = None

                        can_buy, fail_reason, t1_open = rules.validate_buy_execution(
                            code, market_indexed, exec_date, prev_close
                        )

                        stock_name = _get_stock_name(
                            indexed, market_indexed, exec_date, decision_date, code
                        )

                        if not can_buy:
                            trade_log.append({
                                'date': exec_date, 'code': code, 'name': stock_name,
                                'action': f'BUY_FAILED_{fail_reason}',
                                'price': t1_open or 0, 'shares': 0,
                                'amount': 0, 'cost': 0, 'profit': 0,
                                'profit_pct': np.nan,
                                'reason': f'{fail_reason}_BLOCKED',
                                'hold_days': 0,
                            })
                            continue

                        per_stock_cash = available_cash / remaining_slots
                        shares = int(per_stock_cash / t1_open / 100) * 100
                        if shares < 100:
                            continue

                        buy_amount = shares * t1_open
                        buy_cost = rules.calc_buy_cost(buy_amount)
                        total_cost = buy_amount + buy_cost
                        if total_cost > cash:
                            continue

                        cash -= total_cost
                        available_cash -= total_cost
                        remaining_slots -= 1

                        positions[code] = {
                            'shares': shares,
                            'buy_price': t1_open,
                            'buy_cost': buy_cost,
                            'buy_date': exec_date,
                            'current_price': t1_open,
                            'hold_days': 0,
                        }

                        trade_log.append({
                            'date': exec_date, 'code': code, 'name': stock_name,
                            'action': 'BUY', 'price': t1_open,
                            'shares': shares, 'amount': buy_amount,
                            'cost': buy_cost, 'profit': 0,
                            'profit_pct': np.nan,
                            'reason': 'SIGNAL', 'hold_days': 0,
                        })
                        n_trades_today += 1

        # 步骤6: 记录资产与持仓快照
        total_value = cash
        for code, pos in positions.items():
            try:
                current_price = indexed.loc[(exec_date, code), 'close']
                pos['current_price'] = current_price
            except KeyError:
                try:
                    current_price = indexed.loc[(decision_date, code), 'close']
                    pos['current_price'] = current_price
                except KeyError:
                    current_price = pos['current_price']
            total_value += pos['shares'] * current_price

            position_log.append(_build_position_record(
                exec_date, code, pos, indexed, exec_date, fallback_date=decision_date
            ))

        daily_records.append({
            'date': exec_date,
            'cash': cash,
            'portfolio_value': total_value,
            'n_positions': len(positions),
            'n_trades': n_trades_today,
        })

        if day_idx % 100 == 0:
            ret = (total_value / INITIAL_CAPITAL - 1) * 100
            print(f"  [{decision_date.date()}] 资产: {total_value:,.0f}, "
                  f"收益: {ret:+.2f}%, 持仓: {len(positions)}只")

    daily_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_log)
    position_df = pd.DataFrame(position_log)

    return {
        'daily': daily_df,
        'trades': trade_df,
        'positions_log': position_df,
        'final_value': daily_df.iloc[-1]['portfolio_value'],
        'positions': positions,
    }


def _get_stock_name(indexed, market_indexed, exec_date, decision_date, code):
    """从多数据源获取股票名称"""
    for src, dt in [(indexed, exec_date), (indexed, decision_date)]:
        try:
            row = src.loc[(dt, code)]
            name = row['name'] if isinstance(row, pd.Series) else row.iloc[0]['name']
            if name:
                return name
        except (KeyError, IndexError):
            continue
    return ''


def _build_position_record(date, code, pos, indexed, primary_date, fallback_date=None):
    """构建持仓快照记录"""
    name = ''
    for dt in [primary_date, fallback_date]:
        if dt is None:
            continue
        try:
            row = indexed.loc[(dt, code)]
            name = row['name'] if isinstance(row, pd.Series) else row.iloc[0]['name']
            break
        except KeyError:
            continue

    return {
        'date': date,
        'code': code,
        'name': name,
        'buy_price': pos['buy_price'],
        'buy_date': pos['buy_date'],
        'hold_days': pos['hold_days'],
        'current_price': pos['current_price'],
        'shares': pos['shares'],
        'market_value': pos['shares'] * pos['current_price'],
        'float_profit': pos['shares'] * (pos['current_price'] - pos['buy_price']),
        'float_profit_pct': (pos['current_price'] - pos['buy_price']) / pos['buy_price'],
    }


def compute_metrics(daily_df: pd.DataFrame) -> dict:
    """计算回测绩效指标"""
    daily_df = daily_df.copy()
    daily_df['daily_return'] = daily_df['portfolio_value'].pct_change()

    total_days = len(daily_df)
    total_trading_days = daily_df['date'].nunique()
    total_return = daily_df.iloc[-1]['portfolio_value'] / INITIAL_CAPITAL - 1
    annual_return = (1 + total_return) ** (252 / max(total_trading_days, 1)) - 1

    rf_daily = 0.025 / 252
    daily_returns = daily_df['daily_return'].dropna()
    excess_returns = daily_returns - rf_daily
    sharpe = excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(252)

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-10
    sortino = excess_returns.mean() / (downside_std + 1e-10) * np.sqrt(252)

    cummax = daily_df['portfolio_value'].cummax()
    drawdown = (daily_df['portfolio_value'] - cummax) / cummax
    max_drawdown = drawdown.min()

    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    annual_vol = daily_returns.std() * np.sqrt(252)
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        '初始资金': f"{INITIAL_CAPITAL:,.0f}",
        '期末资产': f"{daily_df.iloc[-1]['portfolio_value']:,.0f}",
        '总回报率': f"{total_return:.2%}",
        '年化收益率': f"{annual_return:.2%}",
        '夏普比率': f"{sharpe:.3f}",
        'Sortino比率': f"{sortino:.3f}",
        '最大回撤': f"{max_drawdown:.2%}",
        '最大回撤持续天数': max_dd_duration,
        '年化波动率': f"{annual_vol:.2%}",
        'Calmar比率': f"{calmar:.3f}",
        '交易天数': total_days,
    }


def compute_trade_metrics(trade_df: pd.DataFrame) -> dict:
    """计算交易统计"""
    if len(trade_df) == 0:
        return {}

    buys = trade_df[trade_df['action'] == 'BUY']
    sells = trade_df[trade_df['action'] == 'SELL']

    # 统计各类失败
    failed = trade_df[trade_df['action'].str.startswith(('BUY_FAILED', 'SELL_FAILED'))]
    fail_counts = failed['action'].value_counts().to_dict() if len(failed) > 0 else {}

    if len(sells) == 0:
        result = {'总交易笔数(买入)': len(buys)}
        result.update(fail_counts)
        return result

    wins = sells[sells['profit'] > 0]
    losses = sells[sells['profit'] <= 0]

    win_rate = len(wins) / len(sells)
    avg_win = wins['profit'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['profit'].mean()) if len(losses) > 0 else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    total_days = (trade_df['date'].max() - trade_df['date'].min()).days
    sell_holds = sells['hold_days']
    avg_hold = sell_holds.mean() if len(sell_holds) > 0 else 0

    reason_stats = ''
    if 'reason' in sells.columns:
        reasons = sells['reason'].value_counts()
        reason_stats = ', '.join([f"{k}:{v}" for k, v in reasons.items()])

    result = {
        '总交易笔数(买入)': len(buys),
        '总交易笔数(卖出)': len(sells),
        '胜率': f"{win_rate:.2%}",
        '盈亏比': f"{profit_loss_ratio:.3f}",
        '平均盈利': f"{avg_win:,.0f}",
        '平均亏损': f"{-losses['profit'].mean() if len(losses) > 0 else 0:,.0f}",
        '平均持有天数': f"{avg_hold:.1f}",
        '日均交易次数': f"{len(trade_df) / max(total_days, 1) * 365 / 252:.2f}",
        '卖出原因': reason_stats,
    }

    if fail_counts:
        result['执行失败统计'] = ', '.join(f"{k}:{v}" for k, v in fail_counts.items())

    return result


def run_pipeline(scoring_method='return_only'):
    """执行回测流水线"""
    merged, market_df = load_data()
    results = run_backtest(merged, market_df, scoring_method=scoring_method)

    daily_df = results['daily']
    trade_df = results['trades']
    position_df = results['positions_log']

    metrics = compute_metrics(daily_df)
    trade_metrics = compute_trade_metrics(trade_df)

    print("\n" + "=" * 50)
    print("回测结果汇总")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("-" * 50)
    for k, v in trade_metrics.items():
        print(f"  {k}: {v}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    daily_df.to_csv(os.path.join(OUTPUT_DIR, 'backtest_daily.csv'), index=False)
    trade_df.to_csv(os.path.join(OUTPUT_DIR, 'trade_log.csv'), index=False)
    position_df.to_csv(os.path.join(OUTPUT_DIR, 'position_log.csv'), index=False)

    md_path = os.path.join(OUTPUT_DIR, 'backtest_metrics.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        start_date = daily_df.iloc[0]['date'].date()
        end_date = daily_df.iloc[-1]['date'].date()
        f.write(f"# A股量化策略回测报告\n\n")
        f.write(f"> 回测期间：**{start_date}** ~ **{end_date}**\n\n")
        f.write("---\n\n")

        f.write("## 资产表现\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|------|-----:|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | **{v}** |\n")
        f.write("\n")

        if trade_metrics:
            f.write("## 交易统计\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|-----:|\n")
            for k, v in trade_metrics.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("*以上为历史回测结果，不代表未来收益，仅供参考。*\n")

    print(f"\n回测结果已保存至 {OUTPUT_DIR}/")
    print(f"   - backtest_daily.csv  (每日资产)")
    print(f"   - trade_log.csv       (交易日志)")
    print(f"   - position_log.csv    (每日持仓快照)")
    print(f"   - backtest_metrics.md (回测指标)")

    return results, metrics, trade_metrics


if __name__ == '__main__':
    run_pipeline()
