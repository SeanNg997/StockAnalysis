"""回测引擎 — 极速优化版（内存换速度）"""

import json
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
BACKTEST_MARKET_PKL = CONFIG['paths'].get('BACKTEST_MARKET_PKL', MARKET_PKL)
OUTPUT_DIR = CONFIG['paths']['BACKTEST_OUTPUT_DIR']
INITIAL_CAPITAL = CONFIG['backtest']['INITIAL_CAPITAL']
LIVE_PROGRESS_FILE = os.environ.get('STOCK_ANALYSIS_PROGRESS_FILE')
MAX_OPEN_TRADE_AMOUNT_RATIO = float(CONFIG['backtest'].get('MAX_OPEN_TRADE_AMOUNT_RATIO', 0.02))
SPECIAL_LIMIT_GAP_TOL = float(CONFIG['backtest'].get('SPECIAL_LIMIT_GAP_TOL', 0.03))
LOT_SIZE = 100


def _json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _reset_live_progress():
    if not LIVE_PROGRESS_FILE:
        return
    os.makedirs(os.path.dirname(LIVE_PROGRESS_FILE), exist_ok=True)
    with open(LIVE_PROGRESS_FILE, 'w', encoding='utf-8'):
        pass


def _emit_live_event(event_type: str, payload: dict):
    if not LIVE_PROGRESS_FILE:
        return
    event = {'type': event_type, 'payload': payload}
    with open(LIVE_PROGRESS_FILE, 'a', encoding='utf-8') as fp:
        fp.write(json.dumps(event, ensure_ascii=False, default=_json_default) + '\n')
        fp.flush()


def _build_live_snapshot(record: dict, progress_pct: float, total_days: int, current_idx: int) -> dict:
    portfolio_value = float(record['portfolio_value'])
    return {
        'date': pd.Timestamp(record['date']).date().isoformat(),
        'cash': float(record['cash']),
        'portfolio_value': portfolio_value,
        'n_positions': int(record['n_positions']),
        'n_trades': int(record['n_trades']),
        'progress_pct': round(float(progress_pct), 4),
        'total_days': int(total_days),
        'current_index': int(current_idx),
        'return_pct': round((portfolio_value / INITIAL_CAPITAL - 1) * 100, 4),
    }


def _max_fill_shares(exec_row: dict, exec_price: float) -> int:
    if exec_price <= 0:
        return 0
    amount = float(exec_row.get('amount', 0) or 0)
    if amount <= 0:
        return 0
    ratio = min(max(float(MAX_OPEN_TRADE_AMOUNT_RATIO), 0.0), 1.0)
    max_amount = amount * ratio
    return int(max_amount / exec_price / LOT_SIZE) * LOT_SIZE


def _is_special_limit_context(prev_close: float, t1_open: float, code: str, is_st: bool) -> bool:
    """识别可能的特殊涨跌停规则日（如新股/复牌），避免误用常规涨跌停限制。"""
    if prev_close is None or prev_close <= 0 or t1_open <= 0:
        return False
    limit_up, _ = rules.get_limit_price(prev_close, code, is_st=is_st)
    normal_limit_pct = abs(limit_up / prev_close - 1.0)
    open_move_pct = abs(t1_open / prev_close - 1.0)
    return open_move_pct > normal_limit_pct + max(SPECIAL_LIMIT_GAP_TOL, 0.0)


def load_data():
    """加载回测所需数据"""
    print("加载数据...")

    price_cols = ['code', 'name', 'date', 'open', 'close', 'pctChg']
    df = pd.read_pickle(CLEAN_PKL)
    price_df = df[price_cols].copy()
    del df

    pred_df = pd.read_pickle(PREDICT_PKL)
    pred_merge = pred_df[['date', 'code', 'pred_return', 'pred_std', 'confidence']].copy()
    del pred_df

    price_df['code'] = price_df['code'].astype('category')
    pred_merge['code'] = pred_merge['code'].astype('category')
    all_codes = price_df['code'].cat.categories.union(pred_merge['code'].cat.categories)
    price_df['code'] = price_df['code'].cat.set_categories(all_codes)
    pred_merge['code'] = pred_merge['code'].cat.set_categories(all_codes)

    merged = price_df.merge(pred_merge, on=['date', 'code'], how='inner')
    merged = merged.sort_values(['date', 'code']).reset_index(drop=True)
    del price_df, pred_merge

    market_cols = ['code', 'date', 'isST', 'isTrading', 'open', 'close', 'amount']
    market_source = BACKTEST_MARKET_PKL if BACKTEST_MARKET_PKL and os.path.exists(BACKTEST_MARKET_PKL) else MARKET_PKL
    market_df = pd.read_pickle(market_source)[market_cols].copy()

    merged_min_date = merged['date'].min()
    merged_max_date = merged['date'].max()
    if (
        market_source != MARKET_PKL and
        (market_df['date'].min() > merged_min_date or market_df['date'].max() < merged_max_date)
    ):
        market_source = MARKET_PKL
        market_df = pd.read_pickle(MARKET_PKL)[market_cols].copy()

    market_df['code'] = market_df['code'].astype('category')
    market_df['isST'] = market_df['isST'].fillna(0).astype('int8')
    market_df['isTrading'] = market_df['isTrading'].fillna(0).astype('int8')

    print(f"合并后数据: {len(merged):,} 行, {merged['code'].nunique()} 只股票")
    print(f"市场状态: {len(market_df):,} 行 (来源: {os.path.basename(market_source)})")
    return merged, market_df


def _prepare_market_for_backtest(market_df: pd.DataFrame, all_dates: list) -> pd.DataFrame:
    """仅保留回测需要的市场日期，加上历史缓冲"""
    if len(all_dates) == 0:
        return market_df.iloc[0:0].copy()

    needed_dates = set(all_dates)
    buffer_days = max(int(rules.MIN_PRICE_DAYS) - 1, 0)
    if buffer_days > 0:
        first_date = all_dates[0]
        market_dates = np.sort(market_df['date'].unique())
        history_dates = market_dates[market_dates < first_date]
        if len(history_dates) > 0:
            needed_dates.update(history_dates[-buffer_days:])

    return market_df[market_df['date'].isin(needed_dates)].copy()


def run_backtest(merged: pd.DataFrame, market_df: pd.DataFrame,
                 scoring_method='confidence_weighted') -> dict:
    """执行回测：T日决策 → T+1日集合竞价成交"""
    all_dates = sorted(merged['date'].unique())
    print(f"回测期间: {all_dates[0].date()} ~ {all_dates[-1].date()}, 共 {len(all_dates)} 个交易日")

    market_df = _prepare_market_for_backtest(market_df, all_dates)
    print(f"市场状态裁剪后: {len(market_df):,} 行, {market_df['date'].nunique()} 个交易日")

    print("构建数据索引...")
    date_grouped = merged.groupby('date')
    market_grouped = market_df.groupby('date')

    next_date_map = {all_dates[i]: all_dates[i + 1] for i in range(len(all_dates) - 1)}
    daily_mkt_ret = merged.groupby('date')['pctChg'].mean().sort_index()

    # 批量构建价格缓存
    print("预处理价格数据...")
    price_cache = {}
    for date, group in merged.groupby('date', sort=False):
        price_cache[date] = group.set_index('code')[['close', 'open', 'name']].to_dict('index')

    # 批量构建市场缓存（用于执行日验证）
    print("预处理市场状态...")
    market_cache = {}
    for date, group in market_df.groupby('date', sort=False):
        market_cache[date] = group.set_index('code')[['open', 'isTrading', 'isST', 'amount']].to_dict('index')

    cash = INITIAL_CAPITAL
    positions = {}
    daily_records = []
    trade_log = []
    position_log = []

    daily_records.append({
        'date': all_dates[0],
        'cash': INITIAL_CAPITAL,
        'portfolio_value': INITIAL_CAPITAL,
        'n_positions': 0,
        'n_trades': 0,
    })
    _emit_live_event('equity', _build_live_snapshot(daily_records[-1], 0.0, len(all_dates), 0))

    print("开始回测模拟...")
    progress_interval = max(1, len(all_dates) // 20)
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
                current_price = price_cache.get(decision_date, {}).get(code, {}).get('close', pos['current_price'])
                pos['current_price'] = current_price
                total_value += pos['shares'] * current_price
                position_log.append(_build_position_record(
                    decision_date, code, pos, price_cache, decision_date
                ))

            if len(daily_records) == 0 or daily_records[-1]['date'] != decision_date:
                daily_records.append({
                    'date': decision_date, 'cash': cash, 'portfolio_value': total_value,
                    'n_positions': len(positions), 'n_trades': 0,
                })
                progress_pct = (day_idx + 1) / len(all_dates) * 100
                _emit_live_event(
                    'equity',
                    _build_live_snapshot(daily_records[-1], progress_pct, len(all_dates), day_idx + 1),
                )
            continue

        for code in positions:
            positions[code]['hold_days'] += 1
        n_trades_today = 0

        # 卖出决策与执行
        sell_list = rules.decide_sells(positions, decision_data)
        actually_sold = set()
        if sell_list:
            exec_market = market_cache.get(exec_date, {})
            decision_prices = price_cache.get(decision_date, {})

            for code, sell_reason in sell_list:
                if code not in positions:
                    continue
                pos = positions[code]

                prev_close = decision_prices.get(code, {}).get('close')
                if prev_close is None:
                    if sell_reason == 'DELIST_FORCE_SELL':
                        prev_close = pos['buy_price']
                    else:
                        continue

                stock_name = decision_prices.get(code, {}).get('name', '')
                if sell_reason == 'DELIST_FORCE_SELL':
                    exec_row = exec_market.get(code, {})
                    t1_open = float(exec_row.get('open', 0) or 0)
                    can_trade = exec_row.get('isTrading') == 1 and t1_open > 0
                    exec_price = t1_open if can_trade else 0.0
                    shares = pos['shares']
                    sell_amount = shares * exec_price
                    sell_cost = rules.calc_sell_cost(sell_amount) if exec_price > 0 else 0.0
                    cash += sell_amount - sell_cost

                    buy_cost_total = shares * pos['buy_price'] + pos['buy_cost']
                    profit = sell_amount - sell_cost - buy_cost_total
                    profit_pct_val = profit / buy_cost_total if buy_cost_total > 0 else 0.0
                    trade_log.append({
                        'date': exec_date, 'code': code, 'name': stock_name,
                        'action': 'SELL', 'price': exec_price, 'shares': shares,
                        'amount': sell_amount, 'cost': sell_cost, 'profit': profit,
                        'profit_pct': profit_pct_val,
                        'reason': 'DELIST_FORCE_SELL' if can_trade else 'DELIST_WRITE_OFF',
                        'hold_days': pos['hold_days'],
                    })
                    n_trades_today += 1
                    actually_sold.add(code)
                    del positions[code]
                    continue

                can_sell, fail_reason, t1_open = _validate_sell(
                    code, exec_market, prev_close, sell_reason
                )

                if not can_sell:
                    trade_log.append({
                        'date': exec_date, 'code': code, 'name': stock_name,
                        'action': f'SELL_FAILED_{fail_reason}', 'price': 0,
                        'shares': pos['shares'], 'amount': 0, 'cost': 0,
                        'profit': 0, 'profit_pct': np.nan,
                        'reason': f'{sell_reason}_BLOCKED_{fail_reason}',
                        'hold_days': pos['hold_days'],
                    })
                    continue

                exec_row = exec_market.get(code, {})
                exec_price = t1_open
                max_fill_shares = _max_fill_shares(exec_row, exec_price)
                filled_shares = min(pos['shares'], max_fill_shares)
                if filled_shares < LOT_SIZE:
                    trade_log.append({
                        'date': exec_date, 'code': code, 'name': stock_name,
                        'action': 'SELL_FAILED_CAPACITY', 'price': exec_price,
                        'shares': pos['shares'], 'amount': 0, 'cost': 0,
                        'profit': 0, 'profit_pct': np.nan,
                        'reason': f'{sell_reason}_BLOCKED_CAPACITY',
                        'hold_days': pos['hold_days'],
                    })
                    continue

                original_shares = pos['shares']
                buy_cost_alloc = pos['buy_cost'] * (filled_shares / original_shares)
                sell_amount = filled_shares * exec_price
                sell_cost = rules.calc_sell_cost(sell_amount) if exec_price > 0 else 0
                cash += sell_amount - sell_cost

                buy_cost_total = filled_shares * pos['buy_price'] + buy_cost_alloc
                profit = sell_amount - sell_cost - buy_cost_total
                profit_pct_val = profit / buy_cost_total if buy_cost_total > 0 else 0
                is_partial = filled_shares < original_shares

                trade_log.append({
                    'date': exec_date, 'code': code, 'name': stock_name,
                    'action': 'SELL', 'price': exec_price, 'shares': filled_shares,
                    'amount': sell_amount, 'cost': sell_cost, 'profit': profit,
                    'profit_pct': profit_pct_val,
                    'reason': f'{sell_reason}_PARTIAL' if is_partial else sell_reason,
                    'hold_days': pos['hold_days'],
                })
                n_trades_today += 1
                if is_partial:
                    pos['shares'] = original_shares - filled_shares
                    pos['buy_cost'] = max(pos['buy_cost'] - buy_cost_alloc, 0.0)
                else:
                    actually_sold.add(code)
                    del positions[code]

        # 买入决策与执行
        mkt_factor = rules.check_market_regime(daily_mkt_ret, decision_date)
        n_slots = rules.compute_buy_slots(len(positions), mkt_factor)

        if n_slots > 0 and len(decision_data) > 0:
            try:
                market_day = market_grouped.get_group(decision_date)
            except KeyError:
                market_day = pd.DataFrame()

            if len(market_day) > 0:
                pool = rules.filter_stock_pool(
                    decision_data, market_day, market_history=market_df
                )
                buy_codes = rules.select_buys(
                    pool, set(positions.keys()), actually_sold,
                    n_slots, scoring_method
                )

                if buy_codes:
                    buy_pool = pool[pool['code'].isin(buy_codes)].copy()
                    allocation = rules.compute_weighted_allocation(cash, buy_pool)

                    exec_market = market_cache.get(exec_date, {})
                    decision_prices = price_cache.get(decision_date, {})

                    for code in buy_codes:
                        if code not in allocation:
                            continue

                        prev_close = decision_prices.get(code, {}).get('close')
                        can_buy, fail_reason, t1_open = _validate_buy(
                            code, exec_market, prev_close
                        )
                        stock_name = decision_prices.get(code, {}).get('name', '')

                        if not can_buy:
                            trade_log.append({
                                'date': exec_date, 'code': code, 'name': stock_name,
                                'action': f'BUY_FAILED_{fail_reason}',
                                'price': t1_open or 0, 'shares': 0,
                                'amount': 0, 'cost': 0, 'profit': 0,
                                'profit_pct': np.nan,
                                'reason': f'{fail_reason}_BLOCKED', 'hold_days': 0,
                            })
                            continue

                        allocated_cash = allocation[code]
                        exec_row = exec_market.get(code, {})
                        exec_price = t1_open
                        max_shares_cash = int(allocated_cash / exec_price / LOT_SIZE) * LOT_SIZE
                        max_shares_capacity = _max_fill_shares(exec_row, exec_price)
                        shares = min(max_shares_cash, max_shares_capacity)
                        if shares < LOT_SIZE:
                            trade_log.append({
                                'date': exec_date, 'code': code, 'name': stock_name,
                                'action': 'BUY_FAILED_CAPACITY',
                                'price': exec_price, 'shares': 0,
                                'amount': 0, 'cost': 0, 'profit': 0,
                                'profit_pct': np.nan,
                                'reason': 'CAPACITY_BLOCKED', 'hold_days': 0,
                            })
                            continue

                        buy_amount = shares * exec_price
                        buy_cost = rules.calc_buy_cost(buy_amount)
                        total_cost = buy_amount + buy_cost
                        if total_cost > cash:
                            continue

                        cash -= total_cost
                        is_partial_fill = shares < max_shares_cash
                        positions[code] = {
                            'shares': shares, 'buy_price': exec_price,
                            'buy_cost': buy_cost, 'buy_date': exec_date,
                            'current_price': exec_price, 'hold_days': 0,
                        }

                        trade_log.append({
                            'date': exec_date, 'code': code, 'name': stock_name,
                            'action': 'BUY', 'price': exec_price, 'shares': shares,
                            'amount': buy_amount, 'cost': buy_cost, 'profit': 0,
                            'profit_pct': np.nan,
                            'reason': 'SIGNAL_PARTIAL' if is_partial_fill else 'SIGNAL',
                            'hold_days': 0,
                        })
                        n_trades_today += 1

        # 记录资产与持仓快照
        total_value = cash
        exec_prices = price_cache.get(exec_date, {})
        decision_prices = price_cache.get(decision_date, {})

        for code, pos in positions.items():
            current_price = exec_prices.get(code, {}).get('close')
            if current_price is None:
                current_price = decision_prices.get(code, {}).get('close', pos['current_price'])
            pos['current_price'] = current_price
            total_value += pos['shares'] * current_price

            position_log.append(_build_position_record(
                exec_date, code, pos, price_cache, exec_date, fallback_date=decision_date
            ))

        daily_records.append({
            'date': exec_date, 'cash': cash, 'portfolio_value': total_value,
            'n_positions': len(positions), 'n_trades': n_trades_today,
        })
        progress_pct = (day_idx + 1) / len(all_dates) * 100
        _emit_live_event(
            'equity',
            _build_live_snapshot(daily_records[-1], progress_pct, len(all_dates), day_idx + 1),
        )

        if day_idx % progress_interval == 0 or day_idx == len(all_dates) - 1:
            ret = (total_value / INITIAL_CAPITAL - 1) * 100
            print(f"  进度: {progress_pct:.1f}% ({day_idx+1}/{len(all_dates)}) | "
                  f"日期: {decision_date.date()} | 收益: {ret:+.2f}% | 持仓: {len(positions)}只")

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


def _validate_buy(code: str, market_cache: dict, prev_close: float) -> tuple:
    """快速买入验证"""
    if code not in market_cache:
        return False, 'NO_DATA', None

    row = market_cache[code]
    if row.get('isTrading') != 1:
        return False, 'SUSPENDED', None
    if row.get('isST', 0) == 1 and not bool(rules.ALLOW_ST_BUY):
        return False, 'ST_DISABLED', None

    t1_open = row.get('open', 0)
    if t1_open <= 0:
        return False, 'INVALID_PRICE', None
    if row.get('amount', 0) < rules.MIN_EXEC_AMOUNT:
        return False, 'LOW_LIQUIDITY', None

    if prev_close is not None:
        is_st = row.get('isST', 0) == 1
        if _is_special_limit_context(prev_close, t1_open, code, is_st):
            return True, 'OK', t1_open
        limit_up, _ = rules.get_limit_price(prev_close, code, is_st=is_st)
        if t1_open >= limit_up - 0.001:
            return False, 'LIMIT_UP', None

    return True, 'OK', t1_open


def _validate_sell(code: str, market_cache: dict, prev_close: float, sell_reason: str = '') -> tuple:
    """快速卖出验证"""
    if code not in market_cache:
        if sell_reason == 'DELIST_FORCE_SELL':
            return True, 'OK', 0.0
        return False, 'NO_DATA', None

    row = market_cache[code]
    if row.get('isTrading') != 1:
        return False, 'SUSPENDED', None

    t1_open = row.get('open', 0)
    if t1_open <= 0:
        return False, 'INVALID_PRICE', None

    if sell_reason != 'DELIST_FORCE_SELL':
        is_st = row.get('isST', 0) == 1
        if _is_special_limit_context(prev_close, t1_open, code, is_st):
            return True, 'OK', t1_open
        _, limit_down = rules.get_limit_price(prev_close, code, is_st=is_st)
        if t1_open <= limit_down + 0.001:
            return False, 'LIMIT_DOWN', None

    return True, 'OK', t1_open


def _build_position_record(date, code, pos, price_cache: dict, primary_date, fallback_date=None):
    """构建持仓快照记录"""
    name = ''
    for dt in [primary_date, fallback_date]:
        if dt is None:
            continue
        cache = price_cache.get(dt, {})
        if code in cache:
            name = cache[code].get('name', '')
            if name:
                break

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


def run_pipeline(scoring_method='confidence_weighted'):
    """执行回测流水线"""
    _reset_live_progress()
    _emit_live_event('status', {'stage': 'load_data', 'message': '开始加载回测数据'})
    merged, market_df = load_data()
    _emit_live_event('status', {'stage': 'backtest', 'message': '开始执行回测模拟'})
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
        f.write("# A股量化策略回测报告\n\n")
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
    print("   - backtest_daily.csv  (每日资产)")
    print("   - trade_log.csv       (交易日志)")
    print("   - position_log.csv    (每日持仓快照)")
    print("   - backtest_metrics.md (回测指标)")
    _emit_live_event(
        'summary',
        {
            'metrics': metrics,
            'trade_metrics': trade_metrics,
            'output_dir': OUTPUT_DIR,
        },
    )

    return results, metrics, trade_metrics


if __name__ == '__main__':
    run_pipeline()
