"""
py05_today.py — 今日实盘决策模块
==================================
职责：
1. 使用最新模型对当前所有可交易主板股票打分
2. 结合回测最后一日的持仓状态
3. 输出今日（下一个交易日）的操作建议
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT_PKL = os.path.join(BASE_DIR, 'data', 'predictions.pkl')
FEATURE_PKL = os.path.join(BASE_DIR, 'data', 'features.pkl')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def generate_today_strategy():
    """生成今日实盘交易决策"""
    print("生成今日交易决策...")

    # 加载预测结果
    pred_df = pd.read_pickle(PREDICT_PKL)
    df = pd.read_pickle(FEATURE_PKL)

    # 找到最新交易日
    latest_date = pred_df['date'].max()
    print(f"最新数据日期: {latest_date.date()}")

    # 获取最新一日的预测
    latest_pred = pred_df[pred_df['date'] == latest_date].copy()

    # 合并价格信息
    latest_price = df[df['date'] == latest_date][['代码', '名称', 'open', 'close', 'volume', 'amount']].copy()
    latest = latest_pred.merge(latest_price, on='代码', how='left')

    # 按预测收益率排序
    latest = latest.sort_values('pred_return', ascending=False)

    # 过滤：预测收益率 > 0.1% 且 置信度较高
    qualified = latest[
        (latest['pred_return'] > 0.001) &
        (latest['confidence'] > latest['confidence'].median())
    ].head(20)

    # ===== 构建报告 =====
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("A股量化策略 — 今日实盘交易决策")
    report_lines.append(f"决策基准日: {latest_date.date()} (使用当日收盘数据)")
    report_lines.append(f"执行日期: 下一个交易日 盘前集合竞价")
    report_lines.append("=" * 70)

    # 推荐买入TOP5
    report_lines.append("\n【推荐买入 TOP 5】")
    report_lines.append("-" * 70)
    report_lines.append(f"{'排名':<6}{'代码':<12}{'名称':<10}{'收盘价':<10}{'预测收益率':<12}{'置信度':<10}{'建议仓位':<10}")
    report_lines.append("-" * 70)

    top5 = qualified.head(5)
    n_buy = len(top5)
    for i, (_, row) in enumerate(top5.iterrows()):
        weight = 1.0 / max(n_buy, 1)
        report_lines.append(
            f"  {i+1:<4}"
            f"{row['代码']:<12}"
            f"{row.get('名称', 'N/A'):<10}"
            f"{row.get('close', 0):>8.2f}"
            f"{row['pred_return']:>10.4%}"
            f"{row['confidence']:>10.4f}"
            f"{weight:>8.1%}"
        )

    if n_buy == 0:
        report_lines.append("  *** 建议空仓：当前无股票满足买入条件 ***")

    # 全市场统计
    report_lines.append(f"\n【市场概况】")
    report_lines.append(f"  当日可预测股票数: {len(latest)}")
    report_lines.append(f"  预测收益率 > 0 的股票数: {(latest['pred_return'] > 0).sum()}")
    report_lines.append(f"  预测收益率 > 0.1% 的股票数: {(latest['pred_return'] > 0.001).sum()}")
    report_lines.append(f"  全市场预测收益率均值: {latest['pred_return'].mean():.4%}")
    report_lines.append(f"  全市场预测收益率中位数: {latest['pred_return'].median():.4%}")

    # TOP20详细列表（无论是否达到买入阈值都展示排名最高的20只）
    report_lines.append(f"\n【潜力排名 TOP 20（全市场排名，不限制阈值）】")
    report_lines.append("-" * 70)
    top20_all = latest.head(20)
    for i, (_, row) in enumerate(top20_all.iterrows()):
        report_lines.append(
            f"  {i+1:>2}. {row['代码']:<12}{row.get('名称', 'N/A'):<10}"
            f"预测: {row['pred_return']:>+.4%}  "
            f"置信度: {row['confidence']:.4f}  "
            f"收盘: {row.get('close', 0):.2f}"
        )

    report_lines.append("\n" + "=" * 70)
    report_lines.append("风险提示：以上为模型预测结果，仅供参考，不构成投资建议。")
    report_lines.append("策略严格遵守T+1规则，以集合竞价开盘价成交。")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'today_strategy.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n✅ 今日策略已保存至 {report_path}")

    return report


if __name__ == '__main__':
    generate_today_strategy()
