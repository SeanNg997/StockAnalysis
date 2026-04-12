# py04_today.py 代码解释

## 文件概览

本文件是今日实盘决策模块，负责使用最新模型对当前所有可交易主板股票打分，结合回测最后一日的持仓状态，输出今日（下一个交易日）的操作建议（Markdown格式）。

## 逐行代码解释

### 第1-13行：文件头部注释

```python
"""
py04_today.py — 今日实盘决策模块
==================================
职责：
1. 使用最新模型对当前所有可交易主板股票打分
2. 结合回测最后一日的持仓状态
3. 输出今日（下一个交易日）的操作建议（Markdown格式）

支持：
- 全市场报告：生成 trading_strategy.md
- 单股票报告：python py04_today.py 600000 → 生成 trading_strategy_600000.md
  （支持输入 600000 或 sh.600000，自动识别前缀）
"""
```
- **第1-13行**：文件头部的文档字符串，说明该文件的职责和功能，以及支持的两种报告模式。

### 第15-24行：导入模块和设置警告

```python
import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import date, timedelta

from config import CONFIG

warnings.filterwarnings('ignore')
```
- **第15行**：导入pandas库，用于数据处理。
- **第16行**：导入numpy库，用于数值计算。
- **第17行**：导入os模块，用于文件和路径操作。
- **第18行**：导入sys模块，用于命令行参数处理。
- **第19行**：导入warnings模块，用于处理警告。
- **第20行**：从datetime模块导入date和timedelta类，用于日期处理。
- **第22行**：从config模块导入CONFIG配置。
- **第24行**：忽略所有警告。

### 第28-48行：next_trading_day函数

```python
def next_trading_day(d: date, known_trading_days=None) -> date:
    """返回 d 之后的下一个交易日。

    Args:
        d: 基准日期
        known_trading_days: 已知交易日的 set（date 类型）。
                            提供时优先查表，可正确处理节假日；
                            未提供时仅跳过周末（原有行为，作为 fallback）。
    """
    next_d = d + timedelta(days=1)
    if known_trading_days is not None:
        # 优先从已知交易日中找（最多向后查找 30 个日历日）
        for _ in range(30):
            if next_d in known_trading_days:
                return next_d
            next_d += timedelta(days=1)
        # 超出已知范围则 fallback
        next_d = d + timedelta(days=1)
    while next_d.weekday() >= 5:  # 5=周六, 6=周日
        next_d += timedelta(days=1)
    return next_d
```
- **第28-48行**：定义next_trading_day函数，返回d之后的下一个交易日。
  - **第29-36行**：函数文档字符串，说明函数的作用和参数。
  - **第37行**：计算下一天的日期。
  - **第38-45行**：如果提供了已知交易日集合，则优先从已知交易日中查找。
  - **第46-47行**：如果没有提供已知交易日集合，或者超出已知范围，则仅跳过周末。
  - **第48行**：返回下一个交易日。

### 第50-53行：设置路径

```python
BASE_DIR = CONFIG['paths']['BASE_DIR']
PREDICT_PKL = CONFIG['paths']['PREDICT_PKL']
FEATURE_PKL = CONFIG['paths']['FEATURE_PKL']
OUTPUT_DIR = CONFIG['paths']['OUTPUT_DIR']
```
- **第50行**：获取基础目录路径。
- **第51行**：获取预测结果文件路径。
- **第52行**：获取特征数据文件路径。
- **第53行**：获取输出目录路径。

### 第56-69行：normalize_stock_code函数

```python
def normalize_stock_code(code):
    """将6位数字代码自动补全为带前缀的完整代码。

    规则：60开头 → sh.6xxxxx，00开头 → sz.0xxxxx
    已有前缀的代码原样返回。
    """
    if '.' in code:
        return code  # 已经是完整格式
    if len(code) == 6 and code.isdigit():
        if code.startswith('60'):
            return f'sh.{code}'
        elif code.startswith('00'):
            return f'sz.{code}'
    raise ValueError(f"无法识别股票代码: {code}，请输入6位主板代码（如 600000 或 002202）")
```
- **第56-69行**：定义normalize_stock_code函数，将6位数字代码自动补全为带前缀的完整代码。
  - **第57-61行**：函数文档字符串，说明函数的作用和规则。
  - **第62-63行**：如果代码中已经包含'.'，则原样返回。
  - **第64-68行**：如果代码是6位数字，则根据开头数字添加前缀。
  - **第69行**：如果代码格式不正确，则抛出ValueError异常。

### 第72-103行：_load_latest_data函数

```python
def _load_latest_data(target_date=None):
    """加载预测结果并返回指定日（或最新日）数据

    Args:
        target_date: 目标日期字符串 'YYYY-MM-DD'，None 表示取最新日期
    """
    # 只加载需要的列
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
    
    # 只加载需要的日期和列
    df = pd.read_pickle(FEATURE_PKL)
    latest_price = df[df['date'] == latest_date][['代码', '名称', 'open', 'close', 'volume', 'amount']].copy()
    
    latest = latest_pred.merge(latest_price, on='代码', how='left')
    latest = latest.sort_values('pred_return', ascending=False)

    return latest, latest_date, pred_df
```
- **第72-103行**：定义_load_latest_data函数，加载预测结果并返回指定日（或最新日）数据。
  - **第73-77行**：函数文档字符串，说明函数的作用和参数。
  - **第79行**：加载预测结果数据。
  - **第81-90行**：如果指定了目标日期，则获取该日期之前的最新交易日的数据。
  - **第92-94行**：获取最新日期的数据。
  - **第96-98行**：加载特征数据，获取最新日期的价格数据。
  - **第100-101行**：合并预测数据和价格数据，并按预测收益率降序排序。
  - **第103行**：返回合并后的数据、最新日期和预测数据。

### 第106-212行：generate_stock_report函数

```python
def generate_stock_report(stock_code, target_date=None):
    """生成单只股票的预测报告

    Args:
        stock_code: 股票代码（6位数字或带前缀）
        target_date: 目标日期 'YYYY-MM-DD'，None 表示最新日期
    """
    full_code = normalize_stock_code(stock_code)
    short_code = full_code.split('.')[1]  # 6位数字代码，用于文件名
    print(f"生成 {full_code} 的交易决策...")
    latest, latest_date, pred_df = _load_latest_data(target_date)

    stock = latest[latest['代码'] == full_code].copy()
    if stock.empty:
        print(f"未找到股票 {full_code} 的预测数据")
        sys.exit(1)

    row = stock.iloc[0]
    stock_name = row.get('名称', 'N/A')
    close_price = row.get('close', 0)
    open_price = row.get('open', 0)
    volume = row.get('volume', 0)
    amount = row.get('amount', 0)
    pred_return = row['pred_return']
    confidence = row['confidence']
    pred_std = row.get('pred_std', 0)

    # 计算全市场排名
    total = len(latest)
    rank = (latest['pred_return'] > pred_return).sum() + 1
    pct = 1 - (rank - 1) / total

    # 判断建议
    min_pred_return = CONFIG['backtest']['MIN_PRED_RETURN']
    min_confidence = CONFIG['backtest']['MIN_CONFIDENCE']
    hold_days = CONFIG['backtest']['HOLD_DAYS']
    
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

    # 获取历史趋势：最近几天的预测变化
    stock_history = pred_df[pred_df['代码'] == full_code].sort_values('date').tail(5)

    known_days = set(pred_df['date'].dt.date.unique())
    exec_date = next_trading_day(latest_date.date(), known_trading_days=known_days)
    lines = []
    lines.append(f"# 📊 {stock_name}（{full_code}）· 今日决策\n")
    lines.append(f"> **决策基准日**：{latest_date.date()}  \n")
    lines.append(f"> **决策应用日期**：{exec_date}（盘前集合竞价）\n")

    lines.append("---\n")
    lines.append("## 📈 预测摘要\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|:----:|")
    lines.append(f"| 收盘价 | ¥{close_price:.2f} |")
    lines.append(f"| 开盘价 | ¥{open_price:.2f} |")
    lines.append(f"| 预测收益率（T+1开盘买入→T+{hold_days+1}开盘卖出，持有{hold_days}天） | **{pred_return:+.4%}** |")
    lines.append(f"| 置信度 | {confidence:.2%} |")
    lines.append(f"| 全市场排名 | {rank} / {total}（前 {pct:.1%}） |")
    lines.append(f"| 成交量 | {volume:,.0f} |")
    lines.append(f"| 成交额 | {amount:,.0f} |")

    lines.append(f"\n## 💡 操作建议：{advice}\n")
    lines.append(f"> {advice_detail}\n")

    # 历史趋势
    lines.append("## 🕐 近5日预测趋势\n")
    lines.append("| 日期 | 预测收益率 | 置信度 |")
    lines.append("|:----:|----------:|-------:|")
    for _, h in stock_history.iterrows():
        lines.append(f"| {h['date'].date()} | {h['pred_return']:+.4%} | {h['confidence']:.2%} |")

    # 市场对比
    lines.append("\n## 🌐 市场对比\n")
    rel_strength = pred_return - latest['pred_return'].mean()
    lines.append("| 指标 | 数值 |")
    lines.append("|------|-----:|")
    lines.append(f"| 全市场预测均值 | {latest['pred_return'].mean():.4%} |")
    lines.append(f"| 全市场预测中位数 | {latest['pred_return'].median():.4%} |")
    lines.append(f"| 相对强度（vs均值） | {rel_strength:+.4%} |")
    lines.append(f"| 置信度排名 | 前 {(latest['confidence'] < confidence).sum() + 1} / {total} |")

    lines.append("\n---\n")
    lines.append("⚠️ *以上为模型预测结果，仅供参考，不构成投资建议。严格遵守 T+1 规则，以集合竞价开盘价成交。*")

    report = "\n".join(lines)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, f'trading_strategy_{short_code}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n策略已保存至 {report_path}")
    return report
```
- **第106-212行**：定义generate_stock_report函数，生成单只股票的预测报告。
  - **第107-112行**：函数文档字符串，说明函数的作用和参数。
  - **第113-115行**：规范化股票代码并打印信息。
  - **第116行**：加载最新数据。
  - **第118-121行**：检查股票是否存在于预测数据中。
  - **第123-131行**：获取股票的各项数据。
  - **第134-136行**：计算全市场排名。
  - **第139-157行**：根据预测收益率和置信度生成操作建议。
  - **第160行**：获取股票的历史预测数据。
  - **第162-163行**：计算下一个交易日。
  - **第164-202行**：构建Markdown报告内容。
  - **第204-208行**：保存报告到文件。
  - **第210-212行**：打印报告并返回。

### 第215-336行：generate_today_strategy函数

```python
def generate_today_strategy(target_date=None):
    """生成全市场实盘交易决策（Markdown格式）

    Args:
        target_date: 目标日期 'YYYY-MM-DD'，None 表示最新日期
    """
    print("生成今日交易决策...")
    latest, latest_date, pred_df = _load_latest_data(target_date)

    # 过滤：使用配置文件中的阈值
    min_pred_return = CONFIG['backtest']['MIN_PRED_RETURN']
    min_confidence = CONFIG['backtest']['MIN_CONFIDENCE']
    hold_days = CONFIG['backtest']['HOLD_DAYS']
    
    qualified = latest[
        (latest['pred_return'] > min_pred_return) &
        (latest['confidence'] > min_confidence)
    ].copy()
    if len(qualified) > 0:
        qualified['score'] = qualified['pred_return'] * 0.6 + \
                             qualified['confidence'] * qualified['pred_return'] * 0.4
        qualified = qualified.sort_values('score', ascending=False).head(20)
    else:
        qualified = qualified.head(0)

    exec_date = next_trading_day(latest_date.date(), known_trading_days=set(
        pred_df['date'].dt.date.unique()
    ))

    # ===== 构建Markdown报告 =====
    lines = []
    lines.append(f"# 📊 A股量化策略 · 今日决策\n")
    lines.append(f"| 决策基准日 | 执行日期 |")
    lines.append(f"|:----------:|:--------:|")
    lines.append(f"| {latest_date.date()} | **{exec_date}**（盘前集合竞价） |\n")
    lines.append("---\n")

    # 市场概况
    n_total = len(latest)
    n_positive = (latest['pred_return'] > 0).sum()
    n_qualified = (latest['pred_return'] > 0.002).sum()
    mkt_mean = latest['pred_return'].mean()
    mkt_median = latest['pred_return'].median()
    # 市场情绪：多头/空头/中性
    bullish_pct = n_positive / n_total if n_total > 0 else 0
    if bullish_pct >= 0.65:
        sentiment = "🟢 偏多"
    elif bullish_pct >= 0.45:
        sentiment = "🟡 中性"
    else:
        sentiment = "🔴 偏空"

    lines.append("## 📈 市场概况\n")
    lines.append("| 指标 | 数值 | 指标 | 数值 |")
    lines.append("|------|-----:|------|-----:|")
    lines.append(f"| 可预测股票数 | {n_total:,} 只 | 市场情绪 | {sentiment} |")
    lines.append(f"| 预测收益率 > 0 | {n_positive:,} 只（{bullish_pct:.0%}） | 突破 0.2% 阈值 | **{n_qualified:,} 只** |")
    lines.append(f"| 全市场预测均值 | {mkt_mean:+.4%} | 全市场预测中位数 | {mkt_median:+.4%} |\n")
    lines.append("---\n")

    # 推荐买入TOP5
    lines.append("## 🏆 今日推荐买入\n")
    top5 = qualified.head(5)
    n_buy = len(top5)

    if n_buy == 0:
        lines.append(f"> **建议空仓**：当前无股票满足买入条件（预测收益率 > {min_pred_return:.2%} 且置信度 > {min_confidence:.0%}）\n")
    else:
        lines.append(f"> 共 **{n_buy}** 只股票满足买入条件，建议各 **{1/n_buy:.0%}** 仓位，持有 {hold_days} 个交易日\n")
        lines.append("| # | 名称 | 代码 | 收盘价 | 预测收益率 | 置信度 | 建议仓位 |")
        lines.append("|:-:|:----:|:----:|-------:|-----------:|-------:|:--------:|")
        for i, (_, row) in enumerate(top5.iterrows()):
            weight = 1.0 / max(n_buy, 1)
            lines.append(
                f"| **{i+1}** | {row.get('名称', 'N/A')} | `{row['代码']}`"
                f" | ¥{row.get('close', 0):.2f}"
                f" | 🔺 **{row['pred_return']:+.4%}**"
                f" | {row['confidence']:.1%}"
                f" | {weight:.0%} |"
            )
    lines.append("\n---\n")

    # TOP20详细列表
    lines.append("## 📋 全市场潜力排名 TOP 20\n")
    lines.append("> 全市场排名，不限制阈值\n")
    lines.append("| # | 名称 | 代码 | 收盘价 | 预测收益率 | 置信度 |")
    lines.append("|:-:|:----:|:----:|-------:|-----------:|-------:|")
    top20_all = latest.head(20)
    for i, (_, row) in enumerate(top20_all.iterrows()):
        arrow = "🔺" if row['pred_return'] > 0 else "🔻"
        in_top5 = "**" if i < n_buy else ""
        lines.append(
            f"| {in_top5}{i+1}{in_top5} | {in_top5}{row.get('名称', 'N/A')}{in_top5}"
            f" | `{row['代码']}`"
            f" | ¥{row.get('close', 0):.2f}"
            f" | {arrow} {row['pred_return']:+.4%}"
            f" | {row['confidence']:.1%} |"
        )

    lines.append("\n---\n")
    lines.append("> ⚠️ 以上为模型预测结果，仅供参考，不构成投资建议。严格遵守 T+1 规则，以集合竞价开盘价成交。")

    report = "\n".join(lines)

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'trading_strategy.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # 归档到历史目录
    history_dir = os.path.join(OUTPUT_DIR, 'history')
    os.makedirs(history_dir, exist_ok=True)
    archive_path = os.path.join(history_dir, f'strategy_{latest_date.date()}.md')
    with open(archive_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n策略已保存至 {report_path}")
    print(f"历史归档: {archive_path}")

    return report
```
- **第215-336行**：定义generate_today_strategy函数，生成全市场实盘交易决策（Markdown格式）。
  - **第216-220行**：函数文档字符串，说明函数的作用和参数。
  - **第221行**：打印生成今日交易决策的信息。
  - **第222行**：加载最新数据。
  - **第225-232行**：根据配置文件中的阈值过滤股票。
  - **第233-238行**：计算得分并排序，取前20只股票。
  - **第240-242行**：计算下一个交易日。
  - **第245-315行**：构建Markdown报告内容，包括市场概况、推荐买入TOP5和全市场潜力排名TOP20。
  - **第317-323行**：保存报告到文件。
  - **第326-330行**：将报告归档到历史目录。
  - **第332-335行**：打印报告并返回。

### 第339-356行：主程序入口

```python
if __name__ == '__main__':
    # 解析参数：支持 --date YYYY-MM-DD 和可选的股票代码
    _target_date = None
    _stock_code = None
    _args = sys.argv[1:]

    if '--date' in _args:
        _idx = _args.index('--date')
        _target_date = _args[_idx + 1]
        _args = [a for i, a in enumerate(_args) if i != _idx and i != _idx + 1]

    if _args:
        _stock_code = _args[0]

    if _stock_code:
        generate_stock_report(_stock_code, target_date=_target_date)
    else:
        generate_today_strategy(target_date=_target_date)
```
- **第339-356行**：主程序入口，解析命令行参数，支持--date参数和可选的股票代码。
  - **第341-342行**：初始化_target_date和_stock_code为None。
  - **第343行**：获取命令行参数。
  - **第345-348行**：如果命令行参数中包含'--date'，则获取指定的日期。
  - **第350-351行**：如果命令行参数中包含股票代码，则获取股票代码。
  - **第353-356行**：根据是否指定了股票代码，调用不同的函数生成报告。