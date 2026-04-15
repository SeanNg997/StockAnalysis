"""今日实盘决策 — 全市场/单股/持仓监控报告"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
from datetime import date, timedelta

from config import CONFIG
import trading_rules as rules

warnings.filterwarnings('ignore')

CLEAN_PKL = CONFIG['paths']['CLEAN_PKL']
PREDICT_PKL = CONFIG['paths']['PREDICT_PKL']
MARKET_PKL = CONFIG['paths']['MARKET_PKL']
OUTPUT_DIR = CONFIG['paths']['OUTPUT_DIR']


def next_trading_day(d: date, known_trading_days=None) -> date:
    """返回d之后的下一个交易日"""
    next_d = d + timedelta(days=1)
    if known_trading_days is not None:
        for _ in range(30):
            if next_d in known_trading_days:
                return next_d
            next_d += timedelta(days=1)
        next_d = d + timedelta(days=1)
    while next_d.weekday() >= 5:
        next_d += timedelta(days=1)
    return next_d


def normalize_stock_code(code):
    """6位数字代码自动补全为带前缀的完整代码"""
    if '.' in code:
        return code
    if len(code) == 6 and code.isdigit():
        if code.startswith('60'):
            return f'sh.{code}'
        elif code.startswith('00'):
            return f'sz.{code}'
    raise ValueError(f"无法识别股票代码: {code}，请输入6位主板代码（如 600000 或 002202）")


def _load_latest_data(target_date=None):
    """加载预测结果、价格数据和市场状态"""
    pred_df = pd.read_pickle(PREDICT_PKL)

    if target_date is not None:
        ts = pd.Timestamp(target_date)
        available = pred_df[pred_df['date'] <= ts]['date']
        if available.empty:
            raise ValueError(f"predictions.pkl 中没有 <= {target_date} 的数据")
        latest_date = available.max()
        if latest_date != ts:
            print(f"  [target_date={target_date} 非交易日，取最近交易日 {latest_date.date()}]")
    else:
        latest_date = pred_df['date'].max()

    print(f"最新数据日期: {latest_date.date()}")

    latest_pred = pred_df[pred_df['date'] == latest_date].copy()

    df = pd.read_pickle(CLEAN_PKL)
    latest_price = df[df['date'] == latest_date][
        ['code', 'name', 'open', 'close', 'volume', 'amount']
    ].copy()
    del df

    latest = latest_pred.merge(latest_price, on='code', how='left')
    latest = latest.sort_values('pred_return', ascending=False)

    market_df = pd.read_pickle(MARKET_PKL)
    market_day = market_df[market_df['date'] == latest_date].copy()

    return latest, latest_date, pred_df, market_day, market_df


def generate_stock_report(stock_code, target_date=None):
    """生成单只股票预测报告"""
    full_code = normalize_stock_code(stock_code)
    short_code = full_code.split('.')[1]
    print(f"生成 {full_code} 的交易决策...")
    latest, latest_date, pred_df, market_day, market_df = _load_latest_data(target_date)

    stock = latest[latest['code'] == full_code].copy()
    if stock.empty:
        print(f"未找到股票 {full_code} 的预测数据")
        sys.exit(1)

    row = stock.iloc[0]
    stock_name = row.get('name', 'N/A')
    close_price = row.get('close', 0)
    open_price = row.get('open', 0)
    volume = row.get('volume', 0)
    amount = row.get('amount', 0)
    pred_return = row['pred_return']
    confidence = row['confidence']
    hold_days = CONFIG['backtest']['HOLD_DAYS']

    mkt_row = market_day[market_day['code'] == full_code]
    warnings_list = []
    if len(mkt_row) > 0:
        mr = mkt_row.iloc[0]
        if mr['isST'] == 1:
            warnings_list.append("该股当前处于 ST 状态，模型预测可能不准确")
        if mr['isTrading'] != 1:
            warnings_list.append("该股当日停牌，无法交易")

    total = len(latest)
    rank = (latest['pred_return'] > pred_return).sum() + 1
    pct = 1 - (rank - 1) / total

    min_pred_return = CONFIG['backtest']['MIN_PRED_RETURN']
    min_confidence = CONFIG['backtest']['MIN_CONFIDENCE']

    if pred_return > 0.005 and confidence > 0.6:
        advice = "强烈推荐买入"
        advice_detail = f"预测收益率较高且置信度强，建议积极关注（持有{hold_days}个交易日）"
    elif pred_return > min_pred_return and confidence > min_confidence:
        advice = "建议买入"
        advice_detail = f"预测有正向收益且置信度高于阈值（持有{hold_days}个交易日）"
    elif pred_return > 0:
        advice = "谨慎关注"
        advice_detail = "预测有正向收益但信号较弱，可少量仓位试探"
    elif pred_return > -0.002:
        advice = "观望"
        advice_detail = "预测收益接近零，建议暂不操作"
    else:
        advice = "不建议买入"
        advice_detail = "预测收益为负，建议回避或减仓"

    stock_history = pred_df[pred_df['code'] == full_code].sort_values('date').tail(5)
    known_days = set(pred_df['date'].dt.date.unique())
    exec_date = next_trading_day(latest_date.date(), known_trading_days=known_days)

    lines = []
    lines.append(f"# {stock_name}（{full_code}）· 今日决策\n")
    lines.append(f"> **决策基准日**：{latest_date.date()}  \n")
    lines.append(f"> **决策应用日期**：{exec_date}（盘前集合竞价）\n")

    if warnings_list:
        lines.append("\n> **风险提示**：" + "；".join(warnings_list) + "\n")

    lines.append("---\n")
    lines.append("## 预测摘要\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|:----:|")
    lines.append(f"| 收盘价 | ¥{close_price:.2f} |")
    lines.append(f"| 开盘价 | ¥{open_price:.2f} |")
    lines.append(f"| 预测收益率（T+1开盘买入→T+{hold_days+1}开盘卖出，持有{hold_days}天） | **{pred_return:+.4%}** |")
    lines.append(f"| 置信度 | {confidence:.2%} |")
    lines.append(f"| 全市场排名 | {rank} / {total}（前 {pct:.1%}） |")
    lines.append(f"| 成交量 | {volume:,.0f} |")
    lines.append(f"| 成交额 | {amount:,.0f} |")

    lines.append(f"\n## 操作建议：{advice}\n")
    lines.append(f"> {advice_detail}\n")

    lines.append("## 近5日预测趋势\n")
    lines.append("| 日期 | 预测收益率 | 置信度 |")
    lines.append("|:----:|----------:|-------:|")
    for _, h in stock_history.iterrows():
        lines.append(f"| {h['date'].date()} | {h['pred_return']:+.4%} | {h['confidence']:.2%} |")

    lines.append("\n## 市场对比\n")
    rel_strength = pred_return - latest['pred_return'].mean()
    lines.append("| 指标 | 数值 |")
    lines.append("|------|-----:|")
    lines.append(f"| 全市场预测均值 | {latest['pred_return'].mean():.4%} |")
    lines.append(f"| 全市场预测中位数 | {latest['pred_return'].median():.4%} |")
    lines.append(f"| 相对强度（vs均值） | {rel_strength:+.4%} |")
    lines.append(f"| 置信度排名 | 前 {(latest['confidence'] < confidence).sum() + 1} / {total} |")

    lines.append("\n---\n")
    lines.append("*以上为模型预测结果，仅供参考，不构成投资建议。*")

    report = "\n".join(lines)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, f'trading_strategy_{short_code}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n策略已保存至 {report_path}")
    return report


def generate_today_strategy(target_date=None):
    """生成全市场实盘交易决策"""
    print("生成今日交易决策...")
    latest, latest_date, pred_df, market_day, market_df = _load_latest_data(target_date)

    hold_days = CONFIG['backtest']['HOLD_DAYS']
    min_pred_return = CONFIG['backtest']['MIN_PRED_RETURN']
    min_confidence = CONFIG['backtest']['MIN_CONFIDENCE']

    # 票池筛选（通过规则引擎）
    n_total = len(latest)
    if len(market_day) > 0:
        qualified, pool_stats = rules.filter_stock_pool(
            latest, market_day, return_stats=True, market_history=market_df
        )
    else:
        qualified = latest[
            (latest['pred_return'] > min_pred_return) &
            (latest['confidence'] > min_confidence)
        ].copy()
        pool_stats = {'总候选': n_total, '最终候选': len(qualified)}

    # 评分排序
    if len(qualified) > 0:
        qualified = rules.score_candidates(qualified)
        qualified = qualified.head(20)

    exec_date = next_trading_day(latest_date.date(), known_trading_days=set(
        pred_df['date'].dt.date.unique()
    ))

    # Markdown 报告
    lines = []
    lines.append(f"# A股量化策略 · 今日决策\n")
    lines.append(f"| 决策基准日 | 执行日期 |")
    lines.append(f"|:----------:|:--------:|")
    lines.append(f"| {latest_date.date()} | **{exec_date}**（盘前集合竞价） |\n")
    lines.append("---\n")

    # 市场概况
    n_positive = (latest['pred_return'] > 0).sum()
    mkt_mean = latest['pred_return'].mean()
    mkt_median = latest['pred_return'].median()
    bullish_pct = n_positive / n_total if n_total > 0 else 0
    if bullish_pct >= 0.65:
        sentiment = "偏多"
    elif bullish_pct >= 0.45:
        sentiment = "中性"
    else:
        sentiment = "偏空"

    lines.append("## 市场概况\n")
    lines.append("| 指标 | 数值 | 指标 | 数值 |")
    lines.append("|------|-----:|------|-----:|")
    lines.append(f"| 可预测股票数 | {n_total:,} 只 | 市场情绪 | {sentiment} |")
    lines.append(f"| 预测收益率 > 0 | {n_positive:,} 只（{bullish_pct:.0%}） | 全市场预测均值 | {mkt_mean:+.4%} |")
    lines.append(f"| 全市场预测中位数 | {mkt_median:+.4%} | | |\n")
    lines.append("---\n")

    # 票池筛选统计
    lines.append("## 票池筛选\n")
    lines.append("| 环节 | 数量 |")
    lines.append("|------|-----:|")
    for k, v in pool_stats.items():
        prefix = "" if k in ('总候选', '最终候选') else "-"
        bold = "**" if k == '最终候选' else ""
        lines.append(f"| {bold}{k}{bold} | {bold}{prefix}{v} 只{bold} |")
    lines.append("")
    lines.append("---\n")

    # 推荐买入 TOP5
    lines.append("## 今日推荐买入\n")
    top5 = qualified.head(5)
    n_buy = len(top5)

    if n_buy == 0:
        lines.append(f"> **建议空仓**：当前无股票满足买入条件"
                      f"（预测收益率 > {min_pred_return:.2%} 且置信度 > {min_confidence:.0%}）\n")
    else:
        lines.append(f"> 共 **{n_buy}** 只股票推荐买入，建议各 **{1/n_buy:.0%}** 仓位，"
                      f"持有 {hold_days} 个交易日\n")
        lines.append("| # | 名称 | 代码 | 收盘价 | 预测收益率 | 置信度 | 建议仓位 |")
        lines.append("|:-:|:----:|:----:|-------:|-----------:|-------:|:--------:|")
        for i, (_, row) in enumerate(top5.iterrows()):
            weight = 1.0 / max(n_buy, 1)
            lines.append(
                f"| **{i+1}** | {row.get('name', 'N/A')} | `{row['code']}`"
                f" | ¥{row.get('close', 0):.2f}"
                f" | **{row['pred_return']:+.4%}**"
                f" | {row['confidence']:.1%}"
                f" | {weight:.0%} |"
            )
    lines.append("\n---\n")

    # TOP20 详细列表
    lines.append("## 全市场潜力排名 TOP 20\n")
    lines.append("> 通过票池筛选后的排名\n")
    lines.append("| # | 名称 | 代码 | 收盘价 | 预测收益率 | 置信度 |")
    lines.append("|:-:|:----:|:----:|-------:|-----------:|-------:|")
    for i, (_, row) in enumerate(qualified.head(20).iterrows()):
        arrow = "+" if row['pred_return'] > 0 else ""
        bold = "**" if i < n_buy else ""
        lines.append(
            f"| {bold}{i+1}{bold} | {bold}{row.get('name', 'N/A')}{bold}"
            f" | `{row['code']}`"
            f" | ¥{row.get('close', 0):.2f}"
            f" | {arrow}{row['pred_return']:.4%}"
            f" | {row['confidence']:.1%} |"
        )

    lines.append("\n---\n")
    lines.append("> *以上为模型预测结果，仅供参考，不构成投资建议。严格遵守 T+1 规则，以集合竞价开盘价成交。*")

    report = "\n".join(lines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'trading_strategy.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    history_dir = os.path.join(OUTPUT_DIR, 'history')
    os.makedirs(history_dir, exist_ok=True)
    archive_path = os.path.join(history_dir, f'strategy_{latest_date.date()}.md')
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n策略已保存至 {report_path}")
    print(f"历史归档: {archive_path}")

    return report


def _load_portfolio(portfolio_path):
    """加载持仓文件（JSON格式）"""
    with open(portfolio_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    positions = {}
    for item in raw:
        code = normalize_stock_code(str(item['code']))
        buy_date = pd.Timestamp(item['buy_date'])
        positions[code] = {
            'buy_price': float(item['buy_price']),
            'buy_date': buy_date,
            'shares': int(item['shares']),
            'current_price': float(item['buy_price']),
            'hold_days': 0,  # 稍后根据交易日历计算
        }
    return positions


def _calc_hold_days(buy_date, latest_date, known_trading_days):
    """计算持有交易日数"""
    count = 0
    for d in known_trading_days:
        if d > buy_date.date() and d <= latest_date.date():
            count += 1
    return count


def generate_portfolio_advice(portfolio_path, target_date=None):
    """生成持仓监控报告"""
    print("生成持仓操作建议...")
    positions = _load_portfolio(portfolio_path)
    if not positions:
        print("持仓为空，无需生成报告")
        return

    latest, latest_date, pred_df, market_day, market_df = _load_latest_data(target_date)
    known_days = sorted(pred_df['date'].dt.date.unique())
    known_days_set = set(known_days)

    # 计算每只持仓的持有天数和当前价格
    for code, pos in positions.items():
        pos['hold_days'] = _calc_hold_days(pos['buy_date'], latest_date, known_days)
        row = latest[latest['code'] == code]
        if not row.empty:
            pos['current_price'] = row.iloc[0].get('close', pos['buy_price'])

    # 构建决策数据（需要 code, close, pred_return 列）
    decision_data = latest[latest['code'].isin(positions.keys())].copy()

    # 卖出
    sell_list = rules.decide_sells(positions, decision_data)
    sell_codes = {code for code, _ in sell_list}
    sell_reasons = {code: reason for code, reason in sell_list}

    # 买入建议
    n_after_sell = len(positions) - len(sell_codes)
    n_empty = rules.MAX_POSITIONS - n_after_sell
    buy_candidates = []
    if n_empty > 0 and len(market_day) > 0:
        qualified = rules.filter_stock_pool(latest, market_day, market_history=market_df)
        if len(qualified) > 0:
            qualified = rules.score_candidates(qualified)
            available = qualified[~qualified['code'].isin(set(positions.keys()))]
            buy_candidates = available.head(n_empty)

    hold_days_cfg = CONFIG['backtest']['HOLD_DAYS']
    exec_date = next_trading_day(latest_date.date(), known_trading_days=known_days_set)

    # 生成报告
    lines = []
    lines.append("# 持仓监控 · 今日操作建议\n")
    lines.append(f"| 决策基准日 | 执行日期 |")
    lines.append(f"|:----------:|:--------:|")
    lines.append(f"| {latest_date.date()} | **{exec_date}**（盘前集合竞价） |\n")
    lines.append("---\n")

    # 持仓概览
    total_cost = sum(p['shares'] * p['buy_price'] for p in positions.values())
    total_value = sum(p['shares'] * p['current_price'] for p in positions.values())
    total_profit = total_value - total_cost
    total_profit_pct = total_profit / total_cost if total_cost > 0 else 0

    lines.append("## 持仓概览\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|-----:|")
    lines.append(f"| 持仓数 | {len(positions)} / {rules.MAX_POSITIONS} |")
    lines.append(f"| 持仓市值 | ¥{total_value:,.0f} |")
    lines.append(f"| 总成本 | ¥{total_cost:,.0f} |")
    lines.append(f"| 浮动盈亏 | **{total_profit:+,.0f}**（{total_profit_pct:+.2%}） |\n")
    lines.append("---\n")

    REASON_CN = {
        'STOP_LOSS': '触发止损',
        'TAKE_PROFIT': '触发止盈',
        'HOLD_EXPIRE': '持有到期',
        'SIGNAL_REVERSE': '信号反转',
    }

    # 逐只持仓详情
    lines.append("## 持仓明细与操作建议\n")
    for code, pos in positions.items():
        row = latest[latest['code'] == code]
        stock_name = row.iloc[0].get('name', 'N/A') if not row.empty else 'N/A'
        pred_return = row.iloc[0]['pred_return'] if not row.empty else np.nan
        confidence = row.iloc[0]['confidence'] if not row.empty else np.nan
        profit_pct = (pos['current_price'] - pos['buy_price']) / pos['buy_price']

        if code in sell_codes:
            action = f"**明日卖出**（{REASON_CN.get(sell_reasons[code], sell_reasons[code])}）"
            action_icon = "SELL"
        else:
            remaining = hold_days_cfg - pos['hold_days']
            action = f"继续持有（剩余 {max(remaining, 0)} 个交易日）"
            action_icon = "HOLD"

        lines.append(f"### {'[卖]' if action_icon == 'SELL' else '[持]'} {stock_name}（`{code}`）\n")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|-----:|")
        lines.append(f"| 买入价 | ¥{pos['buy_price']:.2f} |")
        lines.append(f"| 当前价 | ¥{pos['current_price']:.2f} |")
        lines.append(f"| 浮盈/亏 | **{profit_pct:+.2%}** |")
        lines.append(f"| 已持有 | {pos['hold_days']} 个交易日 |")
        if not np.isnan(pred_return):
            lines.append(f"| 最新预测收益率 | {pred_return:+.4%} |")
            lines.append(f"| 置信度 | {confidence:.2%} |")
        lines.append(f"| **操作建议** | {action} |")
        lines.append("")

    lines.append("---\n")

    # 买入建议
    lines.append("## 新买入建议\n")
    if len(buy_candidates) == 0:
        if n_empty <= 0:
            lines.append("> 仓位已满，无需新买入\n")
        else:
            lines.append("> 当前无股票满足买入条件\n")
    else:
        n_buy = len(buy_candidates)
        lines.append(f"> 卖出后空余 **{n_empty}** 个仓位，推荐以下股票：\n")
        lines.append("| # | 名称 | 代码 | 收盘价 | 预测收益率 | 置信度 |")
        lines.append("|:-:|:----:|:----:|-------:|-----------:|-------:|")
        for i, (_, row) in enumerate(buy_candidates.iterrows()):
            lines.append(
                f"| {i+1} | {row.get('name', 'N/A')} | `{row['code']}`"
                f" | ¥{row.get('close', 0):.2f}"
                f" | **{row['pred_return']:+.4%}**"
                f" | {row['confidence']:.1%} |"
            )

    lines.append("\n---\n")

    # 操作清单
    lines.append("## 操作清单（集合竞价执行）\n")
    has_action = False
    seq = 0
    for code, reason in sell_list:
        pos = positions[code]
        row = latest[latest['code'] == code]
        name = row.iloc[0].get('name', '') if not row.empty else ''
        seq += 1
        lines.append(f"{seq}. **卖出** {name}（`{code}`）{pos['shares']} 股"
                      f" — {REASON_CN.get(reason, reason)}")
        has_action = True
    for _, row in buy_candidates.iterrows():
        seq += 1
        lines.append(f"{seq}. **买入** {row.get('name', '')}（`{row['code']}`）"
                      f" — 预测收益 {row['pred_return']:+.4%}")
        has_action = True
    if not has_action:
        lines.append("> 今日无需操作，继续持有\n")

    lines.append("\n---\n")
    lines.append("> *以上为模型预测结果，仅供参考，不构成投资建议。*")

    report = "\n".join(lines)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'portfolio_advice.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n操作建议已保存至 {report_path}")
    return report


if __name__ == '__main__':
    _target_date = None
    _stock_code = None
    _portfolio_path = None
    _args = sys.argv[1:]

    if '--date' in _args:
        _idx = _args.index('--date')
        _target_date = _args[_idx + 1]
        _args = [a for i, a in enumerate(_args) if i != _idx and i != _idx + 1]

    if '--portfolio' in _args:
        _idx = _args.index('--portfolio')
        _portfolio_path = _args[_idx + 1]
        _args = [a for i, a in enumerate(_args) if i != _idx and i != _idx + 1]

    if _args:
        _stock_code = _args[0]

    if _portfolio_path:
        generate_portfolio_advice(_portfolio_path, target_date=_target_date)
    elif _stock_code:
        generate_stock_report(_stock_code, target_date=_target_date)
    else:
        generate_today_strategy(target_date=_target_date)
