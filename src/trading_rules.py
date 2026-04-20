"""交易规则引擎 — 票池筛选 / 执行日验证 / 卖出决策 / 买入选股 / 市场择时"""

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
ALLOW_ST_BUY = _BT.get('ALLOW_ST_BUY', True)


def calc_buy_cost(amount: float) -> float:
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def calc_sell_cost(amount: float) -> float:
    return max(amount * COMMISSION_RATE, MIN_COMMISSION) + amount * STAMP_TAX


def get_limit_price(prev_close: float, code: str, is_st: bool = False) -> tuple:
    """根据T日收盘价与证券属性计算T+1日涨停/跌停价"""
    code_lower = code.lower()
    if is_st:
        pct = 0.05
    elif code_lower.startswith('sh.688') or code_lower.startswith('sz.300'):
        pct = 0.20
    elif code_lower.startswith('bj.'):
        pct = 0.30
    else:
        pct = 0.10
    limit_up = round(prev_close * (1 + pct), 2)
    limit_down = round(prev_close * (1 - pct), 2)
    return limit_up, limit_down


def compute_dynamic_threshold(candidates: pd.DataFrame, base_threshold: float = MIN_PRED_RETURN) -> float:
    """根据市场环境动态调整预测收益阈值

    Args:
        candidates: 候选股票（含预测数据）
        base_threshold: 基础阈值

    Returns:
        动态调整后的阈值
    """
    if len(candidates) == 0:
        return base_threshold

    median_return = candidates['pred_return'].median()
    pct_75 = candidates['pred_return'].quantile(0.75)
    pct_25 = candidates['pred_return'].quantile(0.25)

    # 市场整体预测收益高时，提高阈值
    if median_return > 0.005:  # 中位数>0.5%，牛市
        return max(base_threshold, pct_25)  # 至少25分位数
    elif median_return < -0.002:  # 中位数<-0.2%，熊市
        return max(base_threshold * 0.5, pct_75)  # 降低阈值但取75分位数
    else:  # 震荡市
        return base_threshold


def _compute_low_price_risk_codes(market_history: pd.DataFrame,
                                  current_date,
                                  candidate_codes=None) -> set:
    """向量化判断最近 N 个有效交易日是否持续低于最低股价阈值。"""
    if market_history is None or len(market_history) == 0 or current_date is None:
        return set()

    history = market_history.loc[
        market_history['date'] <= current_date, ['code', 'date', 'close']
    ].copy()
    if history.empty:
        return set()

    if candidate_codes is not None:
        history = history[history['code'].isin(candidate_codes)]
        if history.empty:
            return set()

    history = history.sort_values(['code', 'date'])
    recent = history.groupby('code', group_keys=False).tail(MIN_PRICE_DAYS)
    if recent.empty:
        return set()

    counts = recent.groupby('code')['close'].size()
    low_counts = recent['close'].lt(MIN_STOCK_PRICE).groupby(recent['code']).sum()
    enough_history = counts >= MIN_PRICE_DAYS

    if MIN_PRICE_CONSECUTIVE:
        flagged = enough_history & (low_counts == MIN_PRICE_DAYS)
    else:
        flagged = enough_history & (low_counts >= MIN_PRICE_DAYS)

    return set(flagged[flagged].index)


def filter_stock_pool(candidates: pd.DataFrame,
                      market_day: pd.DataFrame,
                      return_stats: bool = False,
                      market_history: pd.DataFrame = None,
                      use_dynamic_threshold: bool = True) -> pd.DataFrame | tuple:
    """决策日票池筛选

    Args:
        candidates: 候选股票（含预测数据）
        market_day: 决策日市场状态
        return_stats: 是否返回筛选统计
        market_history: 历史市场数据（用于检查连续低价），需要包含['code','date','close']
        use_dynamic_threshold: 是否使用动态预测收益阈值
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

    # 1. ST 过滤：可配置（实盘可按券商权限关闭）
    if not ALLOW_ST_BUY:
        st_mask = merged['isST'] == 1
        n_st = st_mask.sum()
        merged = merged[~st_mask]
        stats['排除ST'] = int(n_st)

    # 2. 排除当日停牌，减少执行层 BUY_FAILED_SUSPENDED
    suspended_mask = merged['isTrading'] != 1
    n_suspended = suspended_mask.sum()
    merged = merged[~suspended_mask]
    stats['排除停牌'] = int(n_suspended)

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
            low_consecutive_codes = _compute_low_price_risk_codes(
                market_history, current_date, candidate_codes=merged['code'].unique()
            )

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

    # 9. 动态调整预测收益阈值（高优先级优化 #9）
    if use_dynamic_threshold:
        dynamic_threshold = compute_dynamic_threshold(merged, MIN_PRED_RETURN)
        stats['动态收益阈值'] = f"{dynamic_threshold:.4f}"
    else:
        dynamic_threshold = MIN_PRED_RETURN

    # 10. 排除低预测收益
    low_ret_mask = merged['pred_return'] <= dynamic_threshold
    n_low_ret = low_ret_mask.sum()
    merged = merged[~low_ret_mask]
    stats['低于收益阈值'] = int(n_low_ret)

    # 11. 排除低置信度
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


def score_candidates(candidates: pd.DataFrame, method: str = 'confidence_weighted') -> pd.DataFrame:
    """计算综合评分并按score降序排列

    高优先级优化 #6: 默认使用置信度加权评分
    """
    df = candidates.copy()
    if method == 'default':
        df['score'] = df['pred_return'] * (0.6 + 0.4 * df['confidence'])
    elif method == 'return_only':
        df['score'] = df['pred_return']
    elif method == 'confidence_weighted':
        # 置信度加权：基础权重0.5 + 置信度贡献0.5
        df['score'] = df['pred_return'] * (0.5 + 0.5 * df['confidence'])
    elif method == 'sharpe_like':
        pred_std = (1.0 / (df['confidence'] + 1e-10) - 1.0) / 100.0
        df['score'] = df['pred_return'] / (pred_std + 1e-10)
    else:
        df['score'] = df['pred_return']
    return df.sort_values('score', ascending=False)


def select_buys(candidates: pd.DataFrame, held_codes: set, sold_today: set,
                n_slots: int, method: str = 'confidence_weighted') -> list:
    """选出要买入的股票代码列表

    高优先级优化 #6: 默认使用置信度加权评分
    """
    if n_slots <= 0 or len(candidates) == 0:
        return []

    scored = score_candidates(candidates, method)
    available = scored[~scored['code'].isin(held_codes | sold_today)]
    return available['code'].head(n_slots).tolist()


def compute_weighted_allocation(cash: float, candidates: pd.DataFrame) -> dict:
    """按质量加权分配资金

    高优先级优化 #13: 根据预测收益和置信度加权分配资金

    Args:
        cash: 可用现金
        candidates: 候选股票（必须包含 pred_return 和 confidence 列）

    Returns:
        {code: allocation_amount} 字典
    """
    if len(candidates) == 0:
        return {}

    # 计算每只股票的权重分数
    candidates = candidates.copy()
    candidates['weight_score'] = (
        candidates['pred_return'] * (0.5 + 0.5 * candidates['confidence'])
    )

    # 归一化权重
    total_score = candidates['weight_score'].sum()
    if total_score <= 0:
        # 如果所有分数都<=0，回退到等权分配
        n = len(candidates)
        equal_weight = 1.0 / n
        candidates['weight'] = equal_weight
    else:
        candidates['weight'] = candidates['weight_score'] / total_score

    # 分配资金（保留5%现金）
    available = cash * 0.95
    allocation = {}
    for _, row in candidates.iterrows():
        allocation[row['code']] = available * row['weight']

    return allocation


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
