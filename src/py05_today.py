"""
py05_today.py — 今日实盘决策模块
==================================
职责：
1. 使用最新模型对当前所有可交易主板股票打分
2. 结合回测最后一日的持仓状态
3. 输出今日（下一个交易日）的操作建议（Markdown格式）

支持：
- 全市场报告：生成 today_strategy.md
- 单股票报告：python py05_today.py 600000 → 生成 today_strategy_600000.md
  （支持输入 600000 或 sh.600000，自动识别前缀）
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT_PKL = os.path.join(BASE_DIR, 'data', 'predictions.pkl')
FEATURE_PKL = os.path.join(BASE_DIR, 'data', 'features.pkl')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


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


def _load_latest_data():
    """加载预测结果并返回最新日数据"""
    pred_df = pd.read_pickle(PREDICT_PKL)
    df = pd.read_pickle(FEATURE_PKL)

    latest_date = pred_df['date'].max()
    print(f"最新数据日期: {latest_date.date()}")

    latest_pred = pred_df[pred_df['date'] == latest_date].copy()
    latest_price = df[df['date'] == latest_date][['代码', '名称', 'open', 'close', 'volume', 'amount']].copy()
    latest = latest_pred.merge(latest_price, on='代码', how='left')
    latest = latest.sort_values('pred_return', ascending=False)

    return latest, latest_date


def generate_stock_report(stock_code):
    """生成单只股票的预测报告"""
    full_code = normalize_stock_code(stock_code)
    short_code = full_code.split('.')[1]  # 6位数字代码，用于文件名
    print(f"生成 {full_code} 的交易决策...")
    latest, latest_date = _load_latest_data()

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
    if pred_return > 0.003 and confidence > 0.6:
        advice = "强烈推荐买入"
        advice_detail = "预测收益率较高且置信度强，建议积极关注"
    elif pred_return > 0.001 and confidence > latest['confidence'].median():
        advice = "建议买入"
        advice_detail = "预测有正向收益且置信度高于中位数"
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
    pred_df = pd.read_pickle(PREDICT_PKL)
    stock_history = pred_df[pred_df['代码'] == full_code].sort_values('date').tail(5)

    lines = []
    lines.append(f"# {stock_name}（{full_code}）— 今日交易决策\n")
    lines.append(f"- **决策基准日**: {latest_date.date()}（使用当日收盘数据）")
    lines.append(f"- **执行日期**: 下一个交易日 盘前集合竞价\n")

    lines.append("## 预测摘要\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 收盘价 | {close_price:.2f} |")
    lines.append(f"| 开盘价 | {open_price:.2f} |")
    lines.append(f"| 预测收益率（T+1→T+2） | **{pred_return:+.4%}** |")
    lines.append(f"| 预测标准差 | {pred_std:.6f} |")
    lines.append(f"| 置信度 | {confidence:.4f} |")
    lines.append(f"| 全市场排名 | {rank} / {total}（前 {pct:.1%}） |")
    lines.append(f"| 成交量 | {volume:,.0f} |")
    lines.append(f"| 成交额 | {amount:,.0f} |")

    lines.append(f"\n## 操作建议: {advice}\n")
    lines.append(f"> {advice_detail}\n")

    # 历史趋势
    lines.append("## 近5日预测趋势\n")
    lines.append("| 日期 | 预测收益率 | 置信度 |")
    lines.append("|------|-----------|--------|")
    for _, h in stock_history.iterrows():
        lines.append(f"| {h['date'].date()} | {h['pred_return']:+.4%} | {h['confidence']:.4f} |")

    # 市场对比
    lines.append("\n## 市场对比\n")
    lines.append(f"- 全市场预测收益率均值: {latest['pred_return'].mean():.4%}")
    lines.append(f"- 全市场预测收益率中位数: {latest['pred_return'].median():.4%}")
    rel_strength = pred_return - latest['pred_return'].mean()
    lines.append(f"- 相对强度（vs均值）: {rel_strength:+.4%}")
    lines.append(f"- 置信度排名: 前 {(latest['confidence'] < confidence).sum() + 1} / {total}")

    lines.append("\n---\n")
    lines.append("*风险提示：以上为模型预测结果，仅供参考，不构成投资建议。策略严格遵守T+1规则。*")

    report = "\n".join(lines)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, f'today_strategy_{short_code}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n策略已保存至 {report_path}")
    return report


def generate_today_strategy():
    """生成全市场今日实盘交易决策（Markdown格式）"""
    print("生成今日交易决策...")
    latest, latest_date = _load_latest_data()

    # 过滤：预测收益率 > 0.1% 且 置信度较高
    qualified = latest[
        (latest['pred_return'] > 0.001) &
        (latest['confidence'] > latest['confidence'].median())
    ].head(20)

    # ===== 构建Markdown报告 =====
    lines = []
    lines.append(f"# A股量化策略 — 今日实盘交易决策\n")
    lines.append(f"- **决策基准日**: {latest_date.date()}（使用当日收盘数据）")
    lines.append(f"- **执行日期**: 下一个交易日 盘前集合竞价\n")

    # 推荐买入TOP5
    lines.append("## 推荐买入 TOP 5\n")
    lines.append("| 排名 | 代码 | 名称 | 收盘价 | 预测收益率 | 置信度 | 建议仓位 |")
    lines.append("|:----:|------|------|-------:|----------:|-------:|---------:|")

    top5 = qualified.head(5)
    n_buy = len(top5)
    for i, (_, row) in enumerate(top5.iterrows()):
        weight = 1.0 / max(n_buy, 1)
        lines.append(
            f"| {i+1} | {row['代码']} | {row.get('名称', 'N/A')} "
            f"| {row.get('close', 0):.2f} "
            f"| {row['pred_return']:+.4%} "
            f"| {row['confidence']:.4f} "
            f"| {weight:.1%} |"
        )

    if n_buy == 0:
        lines.append("\n> **建议空仓**：当前无股票满足买入条件\n")

    # 市场概况
    lines.append("\n## 市场概况\n")
    lines.append(f"- 当日可预测股票数: **{len(latest)}**")
    lines.append(f"- 预测收益率 > 0 的股票数: {(latest['pred_return'] > 0).sum()}")
    lines.append(f"- 预测收益率 > 0.1% 的股票数: {(latest['pred_return'] > 0.001).sum()}")
    lines.append(f"- 全市场预测收益率均值: {latest['pred_return'].mean():.4%}")
    lines.append(f"- 全市场预测收益率中位数: {latest['pred_return'].median():.4%}")

    # TOP20详细列表
    lines.append("\n## 潜力排名 TOP 20\n")
    lines.append("> 全市场排名，不限制阈值\n")
    lines.append("| 排名 | 代码 | 名称 | 预测收益率 | 置信度 | 收盘价 |")
    lines.append("|:----:|------|------|----------:|-------:|-------:|")
    top20_all = latest.head(20)
    for i, (_, row) in enumerate(top20_all.iterrows()):
        lines.append(
            f"| {i+1} | {row['代码']} | {row.get('名称', 'N/A')} "
            f"| {row['pred_return']:+.4%} "
            f"| {row['confidence']:.4f} "
            f"| {row.get('close', 0):.2f} |"
        )

    lines.append("\n---\n")
    lines.append("*风险提示：以上为模型预测结果，仅供参考，不构成投资建议。策略严格遵守T+1规则，以集合竞价开盘价成交。*")

    report = "\n".join(lines)

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'today_strategy.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n策略已保存至 {report_path}")

    return report


if __name__ == '__main__':
    if len(sys.argv) > 1:
        stock_code = sys.argv[1]
        generate_stock_report(stock_code)
    else:
        generate_today_strategy()
