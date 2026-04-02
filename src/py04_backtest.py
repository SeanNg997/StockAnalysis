"""
py04_backtest.py — 回测引擎模块（优化版）
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
MIN_PRED_RETURN = 0.002      # 最低预测收益率阈值（0.2%，提高门槛）
MIN_CONFIDENCE = 0.5         # 最低置信度阈值
HOLD_DAYS = 5                # 目标持有天数
STOP_LOSS = -0.05            # 止损线 -5%
TAKE_PROFIT = 0.08           # 止盈线 +8%
INITIAL_CAPITAL = 1_000_000  # 初始资金100万
MAX_DAILY_BUY = 2            # 每天最多买入2只（分批建仓）


def load_data():
    """加载特征数据和预测结果"""
    print("加载数据...")
    df = pd.read_pickle(FEATURE_PKL)
    pred_df = pd.read_pickle(PREDICT_PKL)

    # 合并预测结果到主数据
    price_cols = ['代码', '名称', 'date', 'open', 'high', 'low', 'close',
                  'volume', 'amount', 'pctChg']
    price_df = df[price_cols].copy()

    # 合并预测
    pred_merge = pred_df[['date', '代码', 'pred_return', 'pred_std', 'confidence']].copy()
    merged = price_df.merge(pred_merge, on=['date', '代码'], how='inner')
    merged = merged.sort_values(['date', '代码']).reset_index(drop=True)

    print(f"合并后数据: {len(merged):,} 行, {merged['代码'].nunique()} 只股票")
    return merged


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


def compute_composite_score(pred_return, confidence):
    """综合评分 = 预测收益率权重 * 0.6 + 置信度权重 * 0.4"""
    return pred_return * 0.6 + confidence * pred_return * 0.4


def check_market_regime(merged, current_date, lookback=10):
    """
    市场择时：检查近期市场环境
    返回仓位系数 0.0~1.0
    """
    recent = merged[merged['date'] <= current_date].groupby('date')['pctChg'].mean()
    recent = recent.sort_index().tail(lookback)

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


def run_backtest(merged: pd.DataFrame) -> dict:
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

    # 构建 (date, 代码) -> row 的映射
    print("构建数据索引...")
    date_stock_map = {}
    for _, row in merged.iterrows():
        date_stock_map[(row['date'], row['代码'])] = row

    # 构建日期索引映射
    next_date_map = {}
    for i in range(len(all_dates) - 1):
        next_date_map[all_dates[i]] = all_dates[i+1]

    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # ============ 回测状态 ============
    cash = INITIAL_CAPITAL
    positions = {}  # {股票代码: {'shares', 'buy_price', 'buy_date', 'current_price', 'hold_days'}}
    locked_stocks = set()  # 当日买入锁定（T+1）

    # 记录
    daily_records = []
    trade_log = []
    position_log = []  # 每日持仓快照

    for day_idx in range(len(all_dates)):
        decision_date = all_dates[day_idx]
        exec_date = next_date_map.get(decision_date)

        # 获取决策日数据
        decision_data = merged[merged['date'] == decision_date]

        if exec_date is None:
            # 最后一个交易日：无后续执行，仅记录资产和持仓快照
            total_value = cash
            for code, pos in positions.items():
                key = (decision_date, code)
                if key in date_stock_map:
                    current_price = date_stock_map[key]['close']
                    pos['current_price'] = current_price
                else:
                    current_price = pos['current_price']
                total_value += pos['shares'] * current_price

                # 持仓快照
                row_info = date_stock_map.get((decision_date, code))
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

            daily_records.append({
                'date': decision_date,
                'cash': cash,
                'portfolio_value': total_value,
                'n_positions': len(positions),
                'n_trades': 0,
            })
            continue

        exec_data = merged[merged['date'] == exec_date]

        # ===== 步骤1：解锁昨日买入的股票，更新持有天数 =====
        locked_stocks.clear()
        for code in positions:
            positions[code]['hold_days'] += 1

        # ===== 步骤2：卖出决策（T日，不受T日涨跌停影响） =====
        sell_list = []
        for code, pos in list(positions.items()):
            if code in locked_stocks:
                continue

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
            t_day_row = date_stock_map.get((decision_date, code))
            if t_day_row is not None:
                _, limit_down_price = get_limit_price(t_day_row['close'], code)
                is_limit_down = t1_open <= limit_down_price + 0.001
            else:
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
            sell_cost = sell_amount * (SELL_COMMISSION + STAMP_TAX)
            cash += sell_amount - sell_cost

            buy_cost_total = pos['shares'] * pos['buy_price'] * (1 + BUY_COMMISSION)
            profit = sell_amount - sell_cost - buy_cost_total
            profit_pct_val = (profit / buy_cost_total * 100) if buy_cost_total > 0 else 0

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

            actually_sold.add(code)
            del positions[code]

        # ===== 步骤4：买入决策（T日，不受T日涨跌停影响） =====
        n_empty = MAX_POSITIONS - len(positions)
        if n_empty > 0:
            # 市场择时系数
            mkt_factor = check_market_regime(merged, decision_date)

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
                    candidates['score'] = candidates.apply(
                        lambda r: compute_composite_score(r['pred_return'], r['confidence']),
                        axis=1
                    )
                    candidates = candidates.sort_values('score', ascending=False)

                    # 过滤已持仓和当日成功卖出的
                    buy_candidates = []
                    for _, cand in candidates.iterrows():
                        code = cand['代码']
                        if code in positions or code in actually_sold:
                            continue
                        buy_candidates.append(code)
                        if len(buy_candidates) >= n_empty:
                            break

                    # ===== 步骤5：执行买入（T+1日，检查T+1日涨停） =====
                    if buy_candidates:
                        available_cash = cash * 0.95  # 保留5%现金
                        per_stock_cash = available_cash / max(n_empty, 1)

                        for code in buy_candidates:
                            exec_info = exec_data[exec_data['代码'] == code]
                            if len(exec_info) == 0:
                                continue

                            t1_open = exec_info.iloc[0]['open']
                            stock_name = exec_info.iloc[0]['名称']

                            if t1_open <= 0:
                                continue

                            # 用T日收盘价计算T+1日涨停价，判断是否涨停封板
                            t_day_row = date_stock_map.get((decision_date, code))
                            if t_day_row is not None:
                                limit_up_price, _ = get_limit_price(t_day_row['close'], code)
                                is_limit_up = t1_open >= limit_up_price - 0.001
                            else:
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

                            shares = int(per_stock_cash / t1_open / 100) * 100
                            if shares < 100:
                                continue

                            buy_amount = shares * t1_open
                            buy_cost = buy_amount * BUY_COMMISSION

                            if buy_amount + buy_cost > cash:
                                continue

                            cash -= buy_amount + buy_cost

                            positions[code] = {
                                'shares': shares,
                                'buy_price': t1_open,
                                'buy_date': exec_date,
                                'current_price': t1_open,
                                'hold_days': 0,
                            }
                            locked_stocks.add(code)

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

        # ===== 步骤6：记录资产与持仓快照 =====
        # 买卖操作发生在 exec_date，所以应使用 exec_date 的 close 做估值
        total_value = cash
        for code, pos in positions.items():
            # 优先用执行日 close 估值（操作已在 exec_date 发生）
            key_exec = (exec_date, code)
            key_dec = (decision_date, code)
            if key_exec in date_stock_map:
                current_price = date_stock_map[key_exec]['close']
                pos['current_price'] = current_price
            elif key_dec in date_stock_map:
                current_price = date_stock_map[key_dec]['close']
                pos['current_price'] = current_price
            else:
                current_price = pos['current_price']
            total_value += pos['shares'] * current_price

            # 持仓快照
            _r = date_stock_map.get(key_exec)
            row_info = _r if _r is not None else date_stock_map.get(key_dec)
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
            'n_trades': sum(1 for t in trade_log if t['date'] == exec_date),
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


def run_pipeline():
    """执行回测流水线"""
    merged = load_data()
    results = run_backtest(merged)

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


if __name__ == '__main__':
    run_pipeline()
