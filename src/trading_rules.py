"""交易规则引擎 — 票池筛选 / 执行日验证 / 卖出决策 / 买入选股 / 市场择时"""

import numpy as np
import pandas as pd

from config import CONFIG

_BT = CONFIG['backtest']
COMMISSION_RATE = _BT['COMMISSION_RATE']
MIN_COMMISSION = _BT['MIN_COMMISSION']
STAMP_TAX = _BT['STAMP_TAX']
MAX_POSITIONS = _BT['MAX_POSITIONS']
MIN_PRED_RETURN = _BT['MIN_PRED_RETURN']
MIN_CONFIDENCE = _BT['MIN_CONFIDENCE']
HOLD_DAYS = _BT['HOLD_DAYS']
STOP_LOSS = _BT['STOP_LOSS']
TAKE_PROFIT = _BT['TAKE_PROFIT']
MAX_DAILY_BUY = _BT['MAX_DAILY_BUY']
MIN_EXEC_AMOUNT = _BT['MIN_EXEC_AMOUNT']
MIN_STOCK_PRICE = _BT['MIN_STOCK_PRICE']
MAX_DELIST_HOLD_DAYS = _BT['MAX_DELIST_HOLD_DAYS']
MARKET_REGIME_LOOKBACK = _BT['MARKET_REGIME_LOOKBACK']
MIN_PRICE_DAYS = _BT.get('MIN_PRICE_DAYS', 5)  # 连续低于MIN_STOCK_PRICE的天数阈值
MIN_PRICE_CONSECUTIVE = _BT.get('MIN_PRICE_CONSECUTIVE', True)  # 是否要求连续


def calc_buy_cost(amount: float) -> float:
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def calc_sell_cost(amount: float) -> float:
    return max(amount * COMMISSION_RATE, MIN_COMMISSION) + amount * STAMP_TAX


def get_limit_price(prev_close: float, code: str) -> tuple:
    """根据T日收盘价计算T+1日涨停/跌停价"""
    code_lower = code.lower()
    if code_lower.startswith('sh.688') or code_lower.startswith('sz.300'):
        pct = 0.20
    elif code_lower.startswith('bj.'):
        pct = 0.30
    else:
        pct = 0.10
    limit_up = round(prev_close * (1 + pct), 2)
    limit_down = round(prev_close * (1 - pct), 2)
    return limit_up, limit_down


def filter_stock_pool(candidates: pd.DataFrame,
                      market_day: pd.DataFrame,
                      return_stats: bool = False,
                      market_history: pd.DataFrame = None) -> pd.DataFrame | tuple:
    """决策日票池筛选

    Args:
        candidates: 候选股票（含预测数据）
        market_day: 决策日市场状态
        return_stats: 是否返回筛选统计
        market_history: 历史市场数据（用于检查连续低价），需要包含['code','date','close']
    """
    total = len(candidates)
    stats = {'总候选': total}

    # 合并市场状态
    merged = candidates.merge(
        market_day[['code', 'isST', 'isTrading', 'amount', 'close']].rename(
            columns={'amount': 'mkt_amount', 'close': 'mkt_close'}
        ),
        on='code', how='left'
    )

    # 1. 不再排除 ST 股票
    # st_mask = merged['isST'] == 1
    # n_st = st_mask.sum()
    # merged = merged[~st_mask]
    # stats['排除ST'] = int(n_st)

    # 2. 不再单独排除停牌（已被最近5天无停牌规则包含）
    # suspended_mask = merged['isTrading'] != 1
    # n_suspended = suspended_mask.sum()
    # merged = merged[~suspended_mask]
    # stats['排除停牌'] = int(n_suspended)

    # 3. 排除流动性不足
    low_liq_mask = merged['mkt_amount'] < MIN_EXEC_AMOUNT
    n_low_liq = low_liq_mask.sum()
    merged = merged[~low_liq_mask]
    stats['排除流动性不足'] = int(n_low_liq)

    # 4. 排除极低价股（股价 < MIN_STOCK_PRICE）
    # 注意：不使用name含"退"过滤，因为name是当前名称，历史时点不知道会改名
    low_price_mask = merged['mkt_close'].fillna(0) < MIN_STOCK_PRICE
    n_delist = low_price_mask.sum()
    merged = merged[~low_price_mask]
    stats['排除极低价股'] = int(n_delist)

    # 5. 排除连续低价股（连续N日低于阈值，避免买到即将退市的票）
    if market_history is not None and len(merged) > 0:
        current_date = market_day['date'].iloc[0] if 'date' in market_day.columns else None
        if current_date is not None:
            # 获取最近MIN_PRICE_DAYS天的历史数据
            recent_history = market_history[
                market_history['date'] <= current_date
            ].sort_values(['code', 'date'])

            # 计算每只股票最近MIN_PRICE_DAYS天的收盘价
            codes = merged['code'].unique()
            low_consecutive_codes = set()

            for code in codes:
                code_hist = recent_history[recent_history['code'] == code].tail(MIN_PRICE_DAYS)
                if len(code_hist) >= MIN_PRICE_DAYS:
                    if MIN_PRICE_CONSECUTIVE:
                        # 检查是否连续N天都低于阈值
                        if (code_hist['close'] < MIN_STOCK_PRICE).all():
                            low_consecutive_codes.add(code)
                    else:
                        # 检查最近N天内是否有任意N天低于阈值
                        if (code_hist['close'] < MIN_STOCK_PRICE).sum() >= MIN_PRICE_DAYS:
                            low_consecutive_codes.add(code)

            low_consecutive_mask = merged['code'].isin(low_consecutive_codes)
            n_low_consecutive = low_consecutive_mask.sum()
            merged = merged[~low_consecutive_mask]
            stats['排除连续低价股'] = int(n_low_consecutive)

    # 6. 排除特征期内有连续2天及以上停牌的股票
    if 'consecutive_suspend' in merged.columns:
        suspend_mask = merged['consecutive_suspend'] > 2
        n_suspend = suspend_mask.sum()
        merged = merged[~suspend_mask]
        stats['排除连续停牌'] = int(n_suspend)
    
    # 7. 排除最近5天有停牌的股票
    if 'recent_5d_suspend' in merged.columns:
        recent_suspend_mask = merged['recent_5d_suspend'] == 1
        n_recent_suspend = recent_suspend_mask.sum()
        merged = merged[~recent_suspend_mask]
        stats['排除最近5天停牌'] = int(n_recent_suspend)
    
    # 8. 排除上市时间不足60天的股票
    if 'isNew' in merged.columns:
        new_mask = merged['isNew'] == 1
        n_new = new_mask.sum()
        merged = merged[~new_mask]
        stats['排除新股'] = int(n_new)

    # 7. 排除低预测收益
    low_ret_mask = merged['pred_return'] <= MIN_PRED_RETURN
    n_low_ret = low_ret_mask.sum()
    merged = merged[~low_ret_mask]
    stats['低于收益阈值'] = int(n_low_ret)

    # 8. 排除低置信度
    low_conf_mask = merged['confidence'] <= MIN_CONFIDENCE
    n_low_conf = low_conf_mask.sum()
    merged = merged[~low_conf_mask]
    stats['低于置信度阈值'] = int(n_low_conf)

    stats['最终候选'] = len(merged)

    # 清理临时列
    drop_cols = [c for c in ['isST', 'isTrading', 'mkt_amount', 'mkt_close', 'consecutive_suspend'] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    if return_stats:
        return merged, stats
    return merged


def validate_buy_execution(code: str, market_indexed, exec_date,
                           prev_close: float = None) -> tuple:
    """执行日买入验证，返回 (can_buy, reason, exec_open)"""
    try:
        row = market_indexed.loc[(exec_date, code)]
    except KeyError:
        return False, 'NO_DATA', None

    if row['isTrading'] != 1:
        return False, 'SUSPENDED', None

    if row['isST'] == 1:
        return False, 'ST', None

    t1_open = row['open']
    if t1_open <= 0:
        return False, 'INVALID_PRICE', None

    if row['amount'] < MIN_EXEC_AMOUNT:
        return False, 'LOW_LIQUIDITY', None

    # 涨停检查
    if prev_close is not None:
        limit_up, _ = get_limit_price(prev_close, code)
        if t1_open >= limit_up - 0.001:
            return False, 'LIMIT_UP', None

    return True, 'OK', t1_open


def validate_sell_execution(code: str, market_indexed, exec_date,
                            prev_close: float, sell_reason: str = '') -> tuple:
    """执行日卖出验证，返回 (can_sell, reason, exec_open)"""
    try:
        row = market_indexed.loc[(exec_date, code)]
    except KeyError:
        # exec_date 在 market_status 里完全找不到该股票 → 真正退市，以0价格强制出仓
        if sell_reason == 'DELIST_FORCE_SELL':
            return True, 'OK', 0.0
        return False, 'NO_DATA', None

    if row['isTrading'] != 1:
        # 停牌：区分是临时停牌还是长期退市
        if sell_reason == 'DELIST_FORCE_SELL':
            # 停牌超过 MAX_DELIST_HOLD_DAYS 天（远超正常持有期）才归零，否则等待复牌
            return False, 'SUSPENDED', None
        return False, 'SUSPENDED', None

    t1_open = row['open']
    if t1_open <= 0:
        return False, 'INVALID_PRICE', None

    # 跌停检查（强制清仓不受跌停限制）
    if sell_reason != 'DELIST_FORCE_SELL':
        _, limit_down = get_limit_price(prev_close, code)
        if t1_open <= limit_down + 0.001:
            return False, 'LIMIT_DOWN', None

    return True, 'OK', t1_open


def decide_sells(positions: dict, decision_data: pd.DataFrame) -> list:
    """T日盘后卖出决策，返回 [(code, sell_reason), ...]"""
    sell_list = []
    dec_by_code = decision_data.set_index('code') if len(decision_data) > 0 else pd.DataFrame()

    for code, pos in positions.items():
        if code not in dec_by_code.index:
            # 持仓股当日无数据（停牌/退市/被过滤）：
            # 超过正常持有天数后强制清仓，避免仓位永久死锁
            if pos['hold_days'] >= HOLD_DAYS:
                sell_list.append((code, 'DELIST_FORCE_SELL'))
            continue

        row = dec_by_code.loc[code]
        current_price = row['close']
        profit_pct = (current_price - pos['buy_price']) / pos['buy_price']

        # 优先级：止损 > 止盈 > 到期 > 信号反转
        if profit_pct <= STOP_LOSS:
            sell_list.append((code, 'STOP_LOSS'))
        elif profit_pct >= TAKE_PROFIT:
            sell_list.append((code, 'TAKE_PROFIT'))
        elif pos['hold_days'] >= HOLD_DAYS:
            sell_list.append((code, 'HOLD_EXPIRE'))
        elif pos['hold_days'] >= 3 and row.get('pred_return', 0) < -0.001:
            sell_list.append((code, 'SIGNAL_REVERSE'))

    return sell_list


def score_candidates(candidates: pd.DataFrame, method: str = 'return_only') -> pd.DataFrame:
    """计算综合评分并按score降序排列"""
    df = candidates.copy()
    if method == 'default':
        df['score'] = df['pred_return'] * (0.6 + 0.4 * df['confidence'])
    elif method == 'return_only':
        df['score'] = df['pred_return']
    elif method == 'confidence_weighted':
        df['score'] = df['pred_return'] * df['confidence']
    elif method == 'sharpe_like':
        pred_std = (1.0 / (df['confidence'] + 1e-10) - 1.0) / 100.0
        df['score'] = df['pred_return'] / (pred_std + 1e-10)
    else:
        df['score'] = df['pred_return']
    return df.sort_values('score', ascending=False)


def select_buys(candidates: pd.DataFrame, held_codes: set, sold_today: set,
                n_slots: int, method: str = 'return_only') -> list:
    """选出要买入的股票代码列表"""
    if n_slots <= 0 or len(candidates) == 0:
        return []

    scored = score_candidates(candidates, method)
    available = scored[~scored['code'].isin(held_codes | sold_today)]
    return available['code'].head(n_slots).tolist()


def compute_buy_allocation(cash: float, n_buy: int) -> float:
    """计算每只股票可分配的资金"""
    if n_buy <= 0:
        return 0.0
    available = cash * 0.95
    return available / n_buy


def check_market_regime(daily_mkt_ret: pd.Series, current_date) -> float:
    """根据近期市场环境返回仓位系数 0.0~1.0"""
    recent = daily_mkt_ret[daily_mkt_ret.index <= current_date].tail(MARKET_REGIME_LOOKBACK)
    if len(recent) < 5:
        return 1.0

    mkt_ret = recent.mean()
    down_ratio = (recent < 0).mean()

    if mkt_ret < -0.01 and down_ratio > 0.7:
        return 0.4
    if mkt_ret < -0.005 and down_ratio > 0.6:
        return 0.6
    return 1.0


def compute_buy_slots(n_current: int, mkt_factor: float) -> int:
    """计算当日可买入的槽位数"""
    n_empty = MAX_POSITIONS - n_current
    if n_empty <= 0:
        return 0
    adjusted_max = max(1, int(MAX_POSITIONS * mkt_factor))
    n_empty = min(n_empty, adjusted_max - n_current)
    return min(max(n_empty, 0), MAX_DAILY_BUY)
