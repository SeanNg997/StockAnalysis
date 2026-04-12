# py05_backtest.py 代码解释

## 文件概览

本文件是回测引擎模块（优化版），负责基于模型预测结果执行完整回测，严格模拟真实交易环境，包括集合竞价open价成交、T+1、交易成本、持仓限制等，并输出回测指标、交易日志和每日持仓日志。

## 逐行代码解释

### 第1-18行：文件头部注释

```python
"""
py05_backtest.py — 回测引擎模块（优化版）
===========================================
职责：
1. 基于模型预测结果执行完整回测
2. 严格模拟：集合竞价open价成交、T+1、交易成本、持仓≤5只
3. 涨跌停处理（T+1日执行时判断，不影响T日决策）
4. 持有期5天策略 + 止损止盈 + 市场择时
5. 输出回测指标、交易日志、每日持仓日志

日期定义：
  T日    = 决策日（盘后），可看到当天open/close
  T+1日  = 执行日，以开盘集合竞价价格成交
  涨跌停 = 根据T日收盘价计算T+1日的涨跌停价格，
           检查T+1日开盘价是否触及涨跌停：
           - 买入涨停封板 → 买入失败，记录 BUY_FAILED_LIMIT_UP，不重新选股
           - 卖出跌停封板 → 卖出失败，记录 SELL_FAILED_LIMIT_DOWN，继续持仓
"""
```
- **第1-18行**：文件头部的文档字符串，说明该文件的职责、功能和日期定义。

### 第20-32行：导入模块和设置路径

```python
import pandas as pd
import numpy as np
import os
import warnings

from config import CONFIG

warnings.filterwarnings('ignore')

BASE_DIR = CONFIG['paths']['BASE_DIR']
FEATURE_PKL = CONFIG['paths']['FEATURE_PKL']
PREDICT_PKL = CONFIG['paths']['PREDICT_PKL']
OUTPUT_DIR = CONFIG['paths']['BACKTEST_OUTPUT_DIR']
```
- **第20行**：导入pandas库，用于数据处理。
- **第21行**：导入numpy库，用于数值计算。
- **第22行**：导入os模块，用于文件和路径操作。
- **第23行**：导入warnings模块，用于处理警告。
- **第25行**：从config模块导入CONFIG配置。
- **第27行**：忽略所有警告。
- **第29行**：获取基础目录路径。
- **第30行**：获取特征数据文件路径。
- **第31行**：获取预测结果文件路径。
- **第32行**：获取回测输出目录路径。

### 第35-37行：交易成本配置

```python
# ============ 交易成本 ============
COMMISSION_RATE = CONFIG['backtest']['COMMISSION_RATE']   # 手续费费率
MIN_COMMISSION = CONFIG['backtest']['MIN_COMMISSION']         # 每笔最低佣金
STAMP_TAX = CONFIG['backtest']['STAMP_TAX']           # 印花税（卖出时收取）
```
- **第35行**：获取手续费费率。
- **第36行**：获取每笔最低佣金。
- **第37行**：获取印花税（卖出时收取）。

### 第40-48行：交易成本计算函数

```python
def calc_buy_cost(amount: float) -> float:
    """计算买入交易成本（佣金，有最低限额）"""
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def calc_sell_cost(amount: float) -> float:
    """计算卖出交易成本（佣金 + 印花税，佣金有最低限额）"""
    return max(amount * COMMISSION_RATE, MIN_COMMISSION) + amount * STAMP_TAX
```
- **第40-42行**：定义calc_buy_cost函数，计算买入交易成本（佣金，有最低限额）。
- **第45-47行**：定义calc_sell_cost函数，计算卖出交易成本（佣金 + 印花税，佣金有最低限额）。

### 第50-57行：策略参数

```python
# ============ 策略参数 ============
MAX_POSITIONS = CONFIG['backtest']['MAX_POSITIONS']            # 最大持仓数
MIN_PRED_RETURN = CONFIG['backtest']['MIN_PRED_RETURN']      # 最低预测收益率阈值
MIN_CONFIDENCE = CONFIG['backtest']['MIN_CONFIDENCE']         # 最低置信度阈值
HOLD_DAYS = CONFIG['backtest']['HOLD_DAYS']                # 目标持有天数
STOP_LOSS = CONFIG['backtest']['STOP_LOSS']            # 止损线
TAKE_PROFIT = CONFIG['backtest']['TAKE_PROFIT']           # 止盈线
INITIAL_CAPITAL = CONFIG['backtest']['INITIAL_CAPITAL']  # 初始资金
MAX_DAILY_BUY = CONFIG['backtest']['MAX_DAILY_BUY']            # 每天最多买入数量
```
- **第50行**：获取最大持仓数。
- **第51行**：获取最低预测收益率阈值。
- **第52行**：获取最低置信度阈值。
- **第53行**：获取目标持有天数。
- **第54行**：获取止损线。
- **第55行**：获取止盈线。
- **第56行**：获取初始资金。
- **第57行**：获取每天最多买入数量。

### 第60-77行：load_data函数

```python
def load_data():
    """加载特征数据和预测结果"""
    print("加载数据...")
    # 只加载需要的列
    df = pd.read_pickle(FEATURE_PKL)
    price_cols = ['代码', '名称', 'date', 'open', 'high', 'low', 'close',
                  'volume', 'amount', 'pctChg']
    price_df = df[price_cols].copy()
    del df  # 释放内存
    
    pred_df = pd.read_pickle(PREDICT_PKL)
    # 合并预测
    pred_merge = pred_df[['date', '代码', 'pred_return', 'pred_std', 'confidence']].copy()
    merged = price_df.merge(pred_merge, on=['date', '代码'], how='inner')
    merged = merged.sort_values(['date', '代码']).reset_index(drop=True)

    print(f"合并后数据: {len(merged):,} 行, {merged['代码'].nunique()} 只股票")
    return merged
```
- **第60-77行**：定义load_data函数，加载特征数据和预测结果。
  - **第61行**：函数文档字符串，说明函数的作用。
  - **第63-68行**：加载特征数据，只保留需要的列。
  - **第69行**：删除df，释放内存。
  - **第71-74行**：加载预测结果，合并到价格数据中。
  - **第76-77行**：打印合并后的数据信息并返回。

### 第80-92行：get_limit_price函数

```python
def get_limit_price(prev_close: float, code: str) -> tuple:
    """根据T日收盘价和股票类型计算T+1日涨停价和跌停价"""
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
```
- **第80-92行**：定义get_limit_price函数，根据T日收盘价和股票类型计算T+1日涨停价和跌停价。
  - **第81行**：函数文档字符串，说明函数的作用。
  - **第82-88行**：根据股票代码判断股票类型，设置不同的涨跌幅限制。
  - **第90-92行**：计算涨停价和跌停价并返回。

### 第95-122行：compute_composite_score函数

```python
def compute_composite_score(pred_return, confidence, method='return_only'):
    """计算综合评分
    
    Args:
        pred_return: 预测收益率
        confidence: 置信度
        method: 权重方法，可选值：
            'default': 原始方法，pred_return * (0.6 + 0.4 * confidence)
            'return_only': 仅按预测收益率排序
            'confidence_weighted': 预测收益率 * 置信度
            'volatility_adjusted': 预测收益率 / (1 + pred_std)
            'sharpe_like': 预测收益率 / (pred_std + 1e-10)
    """
    if method == 'default':
        return pred_return * (0.6 + 0.4 * confidence)
    elif method == 'return_only':
        return pred_return
    elif method == 'confidence_weighted':
        return pred_return * confidence
    elif method == 'volatility_adjusted':
        # 假设pred_std可以通过confidence计算：confidence = 1/(1+std*100) → std = (1/confidence - 1)/100
        pred_std = (1.0 / (confidence + 1e-10) - 1.0) / 100.0
        return pred_return / (1.0 + pred_std)
    elif method == 'sharpe_like':
        pred_std = (1.0 / (confidence + 1e-10) - 1.0) / 100.0
        return pred_return / (pred_std + 1e-10)
    else:
        return pred_return * (0.6 + 0.4 * confidence)
```
- **第95-122行**：定义compute_composite_score函数，计算综合评分。
  - **第96-107行**：函数文档字符串，说明函数的作用、参数和方法选项。
  - **第108-122行**：根据不同的方法计算综合评分。

### 第125-149行：check_market_regime函数

```python
def check_market_regime(daily_mkt_ret: pd.Series, current_date):
    """
    市场择时：检查近期市场环境（使用预计算的每日市场平均收益率）
    返回仓位系数 0.0~1.0
    """
    lookback = CONFIG['backtest']['MARKET_REGIME_LOOKBACK']
    recent = daily_mkt_ret[daily_mkt_ret.index <= current_date].tail(lookback)

    if len(recent) < 5:
        return 1.0

    # 近期市场平均涨跌幅
    mkt_ret = recent.mean()
    # 近期下跌天数占比
    down_ratio = (recent < 0).mean()

    # 市场极度弱势（均值 < -1% 且下跌天数 > 70%）→ 半仓
    if mkt_ret < -0.01 and down_ratio > 0.7:
        return 0.4
    # 市场偏弱 → 减仓
    elif mkt_ret < -0.005 and down_ratio > 0.6:
        return 0.6
    # 正常
    else:
        return 1.0
```
- **第125-149行**：定义check_market_regime函数，检查近期市场环境，返回仓位系数。
  - **第126-129行**：函数文档字符串，说明函数的作用。
  - **第130-131行**：获取市场择时回看天数，获取近期的市场平均收益率。
  - **第133-134行**：如果数据不足，返回1.0（满仓）。
  - **第137-139行**：计算近期市场平均涨跌幅和下跌天数占比。
  - **第142-148行**：根据市场情况返回不同的仓位系数。

### 第152-568行：run_backtest函数

```python
def run_backtest(merged: pd.DataFrame, scoring_method='return_only') -> dict:
    """
    执行回测（优化版）

    核心逻辑：
    - 决策日T：T日收盘后选股，不受T日涨跌停影响
    - 执行日T+1：T+1日开盘集合竞价成交
      * 买入时：若T+1日开盘价触及涨停 → 买入失败，记录 BUY_FAILED_LIMIT_UP
      * 卖出时：若T+1日开盘价触及跌停 → 卖出失败，记录 SELL_FAILED_LIMIT_DOWN，继续持仓
      * 失败后不重新选股补位
    - 持有5个交易日后卖出（或触发止损/止盈提前卖出）
    - 每天最多买入2只，分批建仓
    """
    all_dates = sorted(merged['date'].unique())
    print(f"回测期间: {all_dates[0].date()} ~ {all_dates[-1].date()}, 共 {len(all_dates)} 个交易日")

    # 构建 MultiIndex 以便快速查询
    print("构建数据索引...")
    indexed = merged.set_index(['date', '代码'])

    # 预分组：每日数据（避免循环内重复过滤）
    date_grouped = merged.groupby('date')

    # 构建日期索引映射
    next_date_map = {}
    for i in range(len(all_dates) - 1):
        next_date_map[all_dates[i]] = all_dates[i+1]

    # 预计算每日市场平均涨跌幅（用于市场择时，避免循环内全表扫描）
    daily_mkt_ret = merged.groupby('date')['pctChg'].mean().sort_index()

    # ============ 回测状态 ============
    cash = INITIAL_CAPITAL
    positions = {}  # {股票代码: {'shares', 'buy_price', 'buy_cost', 'buy_date', 'current_price', 'hold_days'}}

    # 记录（不插入初始记录，避免重复）
    daily_records = []
    trade_log = []
    position_log = []  # 每日持仓快照
    n_trades_today = 0  # 当日交易计数

    # 先添加第一天的初始记录
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

        # 获取决策日数据（使用预分组）
        try:
            decision_data = date_grouped.get_group(decision_date)
        except KeyError:
            decision_data = pd.DataFrame()

        if exec_date is None:
            # 最后一个交易日：无后续执行，仅记录资产和持仓快照
            total_value = cash
            for code, pos in positions.items():
                try:
                    current_price = indexed.loc[(decision_date, code), 'close']
                    pos['current_price'] = current_price
                except KeyError:
                    current_price = pos['current_price']
                total_value += pos['shares'] * current_price

                # 持仓快照
                try:
                    row_info = indexed.loc[(decision_date, code)]
                except KeyError:
                    row_info = None
                position_log.append({
                    'date': decision_date,
                    '代码': code,
                    '名称': row_info['名称'] if row_info is not None else '',
                    'buy_price': pos['buy_price'],
                    'buy_date': pos['buy_date'],
                    'hold_days': pos['hold_days'],
                    'current_price': pos['current_price'],
                    'shares': pos['shares'],
                    'market_value': pos['shares'] * pos['current_price'],
                    'float_profit': pos['shares'] * (pos['current_price'] - pos['buy_price']),
                    'float_profit_pct': (pos['current_price'] - pos['buy_price']) / pos['buy_price'],
                })

            # 注意：第一天的初始记录已经添加，最后一天只需要追加一次
            # 检查是否已经是最后一天且不是第一天（避免重复）
            if len(daily_records) == 0 or daily_records[-1]['date'] != decision_date:
                daily_records.append({
                    'date': decision_date,
                    'cash': cash,
                    'portfolio_value': total_value,
                    'n_positions': len(positions),
                    'n_trades': 0,
                })
            continue

        try:
            exec_data = date_grouped.get_group(exec_date)
        except KeyError:
            exec_data = pd.DataFrame()

        # ===== 步骤1：更新持有天数 =====
        for code in positions:
            positions[code]['hold_days'] += 1
        n_trades_today = 0

        # ===== 步骤2：卖出决策（T日，不受T日涨跌停影响） =====
        sell_list = []
        for code, pos in list(positions.items()):

            dec_pred = decision_data[decision_data['代码'] == code]
            if len(dec_pred) == 0:
                continue

            row_decision = dec_pred.iloc[0]

            # 计算浮动盈亏（使用T日close估算）
            current_price = row_decision['close']
            profit_pct = (current_price - pos['buy_price']) / pos['buy_price']

            # 卖出条件
            should_sell = False
            sell_reason = ''

            # 1. 止损
            if profit_pct <= STOP_LOSS:
                should_sell = True
                sell_reason = 'STOP_LOSS'

            # 2. 止盈
            elif profit_pct >= TAKE_PROFIT:
                should_sell = True
                sell_reason = 'TAKE_PROFIT'

            # 3. 持有到期
            # T+1买入, 持有HOLD_DAYS天, T+1+HOLD_DAYS卖出
            elif pos['hold_days'] >= HOLD_DAYS:
                should_sell = True
                sell_reason = 'HOLD_EXPIRE'

            # 4. 持有超过3天且预测转负
            elif pos['hold_days'] >= 3:
                pred_ret = row_decision['pred_return']
                if pred_ret < -0.001:
                    should_sell = True
                    sell_reason = 'SIGNAL_REVERSE'

            if should_sell:
                sell_list.append((code, sell_reason))

        # ===== 步骤3：执行卖出（T+1日，检查T+1日跌停） =====
        # 注意：sold_today 在执行后构建，只包含成功卖出的股票
        actually_sold = set()

        for code, sell_reason in sell_list:
            if code not in positions:
                continue
            exec_info = exec_data[exec_data['代码'] == code]
            if len(exec_info) == 0:
                continue  # 停牌，无法执行

            pos = positions[code]
            t1_open = exec_info.iloc[0]['open']
            stock_name = exec_info.iloc[0]['名称']

            # 用T日收盘价计算T+1日跌停价，判断是否跌停封板
            try:
                t_day_row = indexed.loc[(decision_date, code)]
                _, limit_down_price = get_limit_price(t_day_row['close'], code)
                is_limit_down = t1_open <= limit_down_price + 0.001
            except KeyError:
                is_limit_down = False

            if is_limit_down:
                # 跌停封板，卖出失败，继续持仓
                trade_log.append({
                    'date': exec_date,
                    '代码': code,
                    '名称': stock_name,
                    'action': 'SELL_FAILED_LIMIT_DOWN',
                    'price': t1_open,
                    'shares': pos['shares'],
                    'amount': 0,
                    'cost': 0,
                    'profit': 0,
                    'profit_pct': np.nan,
                    'reason': sell_reason + '_BLOCKED_LIMIT_DOWN',
                    'hold_days': pos['hold_days'],
                })
                continue  # 继续持仓，不重新选股补位

            # 正常卖出
            sell_amount = pos['shares'] * t1_open
            sell_cost = calc_sell_cost(sell_amount)
            cash += sell_amount - sell_cost

            buy_cost_total = pos['shares'] * pos['buy_price'] + pos['buy_cost']
            profit = sell_amount - sell_cost - buy_cost_total
            profit_pct_val = (profit / buy_cost_total) if buy_cost_total > 0 else 0

            trade_log.append({
                'date': exec_date,
                '代码': code,
                '名称': stock_name,
                'action': 'SELL',
                'price': t1_open,
                'shares': pos['shares'],
                'amount': sell_amount,
                'cost': sell_cost,
                'profit': profit,
                'profit_pct': profit_pct_val,
                'reason': sell_reason,
                'hold_days': pos['hold_days'],
            })
            n_trades_today += 1

            actually_sold.add(code)
            del positions[code]

        # ===== 步骤4：买入决策（T日，不受T日涨跌停影响） =====
        n_empty = MAX_POSITIONS - len(positions)
        if n_empty > 0:
            # 市场择时系数
            mkt_factor = check_market_regime(daily_mkt_ret, decision_date)

            # 根据市场状态调整最大仓位
            adjusted_max = max(1, int(MAX_POSITIONS * mkt_factor))
            n_empty = min(n_empty, adjusted_max - len(positions))
            n_empty = min(n_empty, MAX_DAILY_BUY)  # 每天最多买入2只

            if n_empty > 0:
                # 综合评分选股
                candidates = decision_data[
                    (decision_data['pred_return'] > MIN_PRED_RETURN) &
                    (decision_data['confidence'] > MIN_CONFIDENCE)
                ].copy()

                if len(candidates) > 0:
                    # 向量化计算综合评分
                    if scoring_method == 'default':
                        candidates['score'] = candidates['pred_return'] * (0.6 + 0.4 * candidates['confidence'])
                    elif scoring_method == 'return_only':
                        candidates['score'] = candidates['pred_return']
                    elif scoring_method == 'confidence_weighted':
                        candidates['score'] = candidates['pred_return'] * candidates['confidence']
                    elif scoring_method == 'volatility_adjusted':
                        # 从confidence计算pred_std
                        pred_std = (1.0 / (candidates['confidence'] + 1e-10) - 1.0) / 100.0
                        candidates['score'] = candidates['pred_return'] / (1.0 + pred_std)
                    elif scoring_method == 'sharpe_like':
                        pred_std = (1.0 / (candidates['confidence'] + 1e-10) - 1.0) / 100.0
                        candidates['score'] = candidates['pred_return'] / (pred_std + 1e-10)
                    else:
                        candidates['score'] = candidates['pred_return'] * (0.6 + 0.4 * candidates['confidence'])
                    candidates = candidates.sort_values('score', ascending=False)

                    # 过滤已持仓和当日成功卖出的
                    available_candidates = candidates[~candidates['代码'].isin(positions.keys()) & ~candidates['代码'].isin(actually_sold)]
                    buy_candidates = available_candidates['代码'].head(n_empty).tolist()

                    # ===== 步骤5：执行买入（T+1日，检查T+1日涨停） =====
                    if buy_candidates:
                        available_cash = cash * 0.95  # 保留5%现金
                        remaining_buy_slots = len(buy_candidates)  # 剩余可买入的股票数量
                        
                        for code in buy_candidates:
                            if remaining_buy_slots <= 0:
                                break
                                
                            exec_info = exec_data[exec_data['代码'] == code]
                            if len(exec_info) == 0:
                                continue

                            t1_open = exec_info.iloc[0]['open']
                            stock_name = exec_info.iloc[0]['名称']

                            if t1_open <= 0:
                                continue

                            # 用T日收盘价计算T+1日涨停价，判断是否涨停封板
                            try:
                                t_day_row = indexed.loc[(decision_date, code)]
                                limit_up_price, _ = get_limit_price(t_day_row['close'], code)
                                is_limit_up = t1_open >= limit_up_price - 0.001
                            except KeyError:
                                is_limit_up = False

                            if is_limit_up:
                                # 涨停封板，买入失败，不算买入
                                trade_log.append({
                                    'date': exec_date,
                                    '代码': code,
                                    '名称': stock_name,
                                    'action': 'BUY_FAILED_LIMIT_UP',
                                    'price': t1_open,
                                    'shares': 0,
                                    'amount': 0,
                                    'cost': 0,
                                    'profit': 0,
                                    'profit_pct': np.nan,
                                    'reason': 'LIMIT_UP_BLOCKED',
                                    'hold_days': 0,
                                })
                                continue  # 不重新选股补位

                            # 计算这只股票可以使用的现金：剩余可用现金 / 剩余买入槽位
                            per_stock_cash = available_cash / remaining_buy_slots
                            
                            shares = int(per_stock_cash / t1_open / 100) * 100
                            if shares < 100:
                                continue

                            buy_amount = shares * t1_open
                            buy_cost = calc_buy_cost(buy_amount)
                            total_cost = buy_amount + buy_cost

                            if total_cost > cash:
                                continue

                            cash -= total_cost
                            available_cash -= total_cost  # 从可用现金中扣除已花费的部分
                            remaining_buy_slots -= 1  # 减少剩余买入槽位

                            positions[code] = {
                                'shares': shares,
                                'buy_price': t1_open,
                                'buy_cost': buy_cost,
                                'buy_date': exec_date,
                                'current_price': t1_open,
                                'hold_days': 0,
                            }

                            trade_log.append({
                                'date': exec_date,
                                '代码': code,
                                '名称': stock_name,
                                'action': 'BUY',
                                'price': t1_open,
                                'shares': shares,
                                'amount': buy_amount,
                                'cost': buy_cost,
                                'profit': 0,
                                'profit_pct': np.nan,
                                'reason': 'SIGNAL',
                                'hold_days': 0,
                            })
                            n_trades_today += 1

        # ===== 步骤6：记录资产与持仓快照 =====
        # 买卖操作发生在 exec_date，所以应使用 exec_date 的 close 做估值
        total_value = cash
        for code, pos in positions.items():
            # 优先用执行日 close 估值（操作已在 exec_date 发生）
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

            # 持仓快照
            try:
                row_info = indexed.loc[(exec_date, code)]
            except KeyError:
                try:
                    row_info = indexed.loc[(decision_date, code)]
                except KeyError:
                    row_info = None
            position_log.append({
                'date': exec_date,
                '代码': code,
                '名称': row_info['名称'] if row_info is not None else '',
                'buy_price': pos['buy_price'],
                'buy_date': pos['buy_date'],
                'hold_days': pos['hold_days'],
                'current_price': pos['current_price'],
                'shares': pos['shares'],
                'market_value': pos['shares'] * pos['current_price'],
                'float_profit': pos['shares'] * (pos['current_price'] - pos['buy_price']),
                'float_profit_pct': (pos['current_price'] - pos['buy_price']) / pos['buy_price'],
            })

        daily_records.append({
            'date': exec_date,
            'cash': cash,
            'portfolio_value': total_value,
            'n_positions': len(positions),
            'n_trades': n_trades_today,
        })

        # 定期打印进度
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
```
- **第152-568行**：定义run_backtest函数，执行回测（优化版）。
  - **第153-164行**：函数文档字符串，说明函数的作用和核心逻辑。
  - **第165-166行**：获取所有交易日并打印回测期间信息。
  - **第169-170行**：构建MultiIndex以便快速查询。
  - **第173行**：预分组每日数据，避免循环内重复过滤。
  - **第176-178行**：构建日期索引映射，用于快速查找下一个交易日。
  - **第181行**：预计算每日市场平均涨跌幅，用于市场择时。
  - **第184-191行**：初始化回测状态，包括现金、持仓和记录。
  - **第194-201行**：添加第一天的初始记录。
  - **第202-550行**：遍历每个交易日，执行回测逻辑。
  - **第552-556行**：定期打印回测进度。
  - **第558-560行**：将记录转换为DataFrame。
  - **第562-568行**：返回回测结果。

### 第571-621行：compute_metrics函数

```python
def compute_metrics(daily_df: pd.DataFrame) -> dict:
    """计算回测指标"""
    daily_df = daily_df.copy()
    daily_df['daily_return'] = daily_df['portfolio_value'].pct_change()

    total_days = len(daily_df)
    total_trading_days = daily_df['date'].nunique()  # 去重，避免末日重复记录影响年化
    total_return = daily_df.iloc[-1]['portfolio_value'] / INITIAL_CAPITAL - 1
    annual_return = (1 + total_return) ** (252 / max(total_trading_days, 1)) - 1

    # 夏普比率（无风险利率2.5%）
    rf_daily = 0.025 / 252
    daily_returns = daily_df['daily_return'].dropna()
    excess_returns = daily_returns - rf_daily
    sharpe = excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(252)

    # Sortino比率（仅考虑下行风险）
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-10
    sortino = excess_returns.mean() / (downside_std + 1e-10) * np.sqrt(252)

    # 最大回撤
    cummax = daily_df['portfolio_value'].cummax()
    drawdown = (daily_df['portfolio_value'] - cummax) / cummax
    max_drawdown = drawdown.min()

    # 最大回撤持续天数
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # 年化波动率
    annual_vol = daily_returns.std() * np.sqrt(252)

    # Calmar比率
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
```
- **第571-621行**：定义compute_metrics函数，计算回测指标。
  - **第572行**：函数文档字符串，说明函数的作用。
  - **第573-574行**：计算每日收益率。
  - **第576-579行**：计算总交易天数、总回报率和年化收益率。
  - **第582-585行**：计算夏普比率。
  - **第588-590行**：计算Sortino比率。
  - **第593-595行**：计算最大回撤。
  - **第598-601行**：计算最大回撤持续天数。
  - **第604行**：计算年化波动率。
  - **第607行**：计算Calmar比率。
  - **第609-621行**：返回计算的指标。

### 第624-674行：compute_trade_metrics函数

```python
def compute_trade_metrics(trade_df: pd.DataFrame) -> dict:
    """计算交易统计"""
    if len(trade_df) == 0:
        return {}

    # 仅统计正常买卖（不含失败记录）
    sells = trade_df[trade_df['action'] == 'SELL']
    buys = trade_df[trade_df['action'] == 'BUY']
    buy_failed = trade_df[trade_df['action'] == 'BUY_FAILED_LIMIT_UP']
    sell_failed = trade_df[trade_df['action'] == 'SELL_FAILED_LIMIT_DOWN']

    if len(sells) == 0:
        return {
            '总交易笔数(买入)': len(buys),
            '买入涨停失败': len(buy_failed),
            '卖出跌停失败': len(sell_failed),
        }

    wins = sells[sells['profit'] > 0]
    losses = sells[sells['profit'] <= 0]

    win_rate = len(wins) / len(sells) if len(sells) > 0 else 0
    avg_win = wins['profit'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['profit'].mean()) if len(losses) > 0 else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    total_days = (trade_df['date'].max() - trade_df['date'].min()).days

    # 平均持有天数
    sell_holds = sells['hold_days'] if 'hold_days' in sells.columns else pd.Series([0])
    avg_hold = sell_holds.mean() if len(sell_holds) > 0 else 0

    # 卖出原因统计
    reason_stats = ''
    if 'reason' in sells.columns:
        reasons = sells['reason'].value_counts()
        reason_stats = ', '.join([f"{k}:{v}" for k, v in reasons.items()])

    return {
        '总交易笔数(买入)': len(buys),
        '总交易笔数(卖出)': len(sells),
        '买入涨停失败': len(buy_failed),
        '卖出跌停失败': len(sell_failed),
        '胜率': f"{win_rate:.2%}",
        '盈亏比': f"{profit_loss_ratio:.3f}",
        '平均盈利': f"{avg_win:,.0f}",
        '平均亏损': f"{-losses['profit'].mean() if len(losses) > 0 else 0:,.0f}",
        '平均持有天数': f"{avg_hold:.1f}",
        '日均交易次数': f"{len(trade_df) / max(total_days, 1) * 365 / 252:.2f}",
        '卖出原因': reason_stats,
    }
```
- **第624-674行**：定义compute_trade_metrics函数，计算交易统计。
  - **第625行**：函数文档字符串，说明函数的作用。
  - **第626-627行**：如果没有交易记录，返回空字典。
  - **第630-633行**：分别统计正常买卖和失败记录。
  - **第635-640行**：如果没有卖出记录，返回基本统计信息。
  - **第642-643行**：统计盈利和亏损的交易。
  - **第645-648行**：计算胜率、平均盈利、平均亏损和盈亏比。
  - **第650行**：计算总天数。
  - **第653-654行**：计算平均持有天数。
  - **第657-660行**：统计卖出原因。
  - **第662-674行**：返回交易统计信息。

### 第677-749行：run_pipeline函数

```python
def run_pipeline(scoring_method='return_only'):
    """执行回测流水线
    
    Args:
        scoring_method: 选股加权方法，可选值：
            'default': 原始方法，pred_return * (0.6 + 0.4 * confidence)
            'return_only': 仅按预测收益率排序
            'confidence_weighted': 预测收益率 * 置信度
            'volatility_adjusted': 预测收益率 / (1 + pred_std)
            'sharpe_like': 预测收益率 / (pred_std + 1e-10)
    """
    merged = load_data()
    results = run_backtest(merged, scoring_method=scoring_method)

    daily_df = results['daily']
    trade_df = results['trades']
    position_df = results['positions_log']

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
    position_df.to_csv(os.path.join(OUTPUT_DIR, 'position_log.csv'), index=False)

    # 保存指标（Markdown格式）
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

    print(f"\n✅ 结果已保存至 {OUTPUT_DIR}/")
    print(f"   - backtest_daily.csv  (每日资产)")
    print(f"   - trade_log.csv       (交易日志，含涨跌停失败记录)")
    print(f"   - position_log.csv    (每日持仓快照)")
    print(f"   - backtest_metrics.md (回测指标，Markdown格式)")

    return results, metrics, trade_metrics
```
- **第677-749行**：定义run_pipeline函数，执行回测流水线。
  - **第678-687行**：函数文档字符串，说明函数的作用和参数。
  - **第688-689行**：加载数据并执行回测。
  - **第691-693行**：获取回测结果。
  - **第696-697行**：计算回测指标和交易统计。
  - **第700-707行**：打印回测结果。
  - **第710-714行**：保存回测结果到文件。
  - **第717-742行**：保存回测指标到Markdown文件。
  - **第744-747行**：打印保存信息。
  - **第749行**：返回回测结果。

### 第752-753行：主程序入口

```python
if __name__ == '__main__':
    run_pipeline()
```
- **第752-753行**：主程序入口，调用run_pipeline函数执行回测。