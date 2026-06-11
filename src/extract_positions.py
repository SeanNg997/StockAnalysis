"""从回测状态文件提取最新持仓，输出为 portfolio.json 格式"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import CONFIG


def extract_positions(state_path=None, output_path=None):
    """从 backtest_state.json 提取当前持仓，输出为 portfolio.json 格式

    Returns:
        list: 持仓列表，每个元素包含 code/buy_price/buy_date/shares 及回测扩展字段
    """
    if state_path is None:
        state_path = os.path.join(CONFIG['paths']['BACKTEST_OUTPUT_DIR'], 'backtest_state.json')
    if output_path is None:
        output_path = os.path.join(CONFIG['paths']['OUTPUT_DIR'], 'portfolio.json')

    if not os.path.exists(state_path):
        print(f"[extract_positions] 回测状态文件不存在: {state_path}")
        return []

    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    raw_positions = state.get('positions', {})
    if not raw_positions:
        print("[extract_positions] 回测状态中无持仓")
        return []

    result = []
    for code, pos in raw_positions.items():
        short_code = code.split('.')[1] if '.' in code else code
        buy_date = pos.get('buy_date', '')
        if buy_date:
            buy_date = pd.Timestamp(buy_date).strftime('%Y-%m-%d')

        result.append({
            'code': short_code,
            'buy_price': float(pos.get('buy_price', 0)),
            'buy_date': buy_date,
            'shares': int(float(pos.get('shares', 0))),
            'basis_amount': float(pos.get('basis_amount', 0)),
            'buy_cost': float(pos.get('buy_cost', 0)),
            'cash_dividends_received': float(pos.get('cash_dividends_received', 0)),
            'max_profit_pct': float(pos.get('max_profit_pct', 0)),
            'current_price': float(pos.get('current_price', 0)),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[extract_positions] 提取 {len(result)} 条持仓 → {output_path}")
    return result


if __name__ == '__main__':
    extract_positions()
