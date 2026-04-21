"""回测引擎 — 原始执行价 + 公司行为稳定处理"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from config import CONFIG
from price_adjust import load_dividend_events
import trading_rules as rules

warnings.filterwarnings("ignore")

CLEAN_PKL = CONFIG["paths"]["CLEAN_PKL"]
PREDICT_PKL = CONFIG["paths"]["PREDICT_PKL"]
MARKET_PKL = CONFIG["paths"]["MARKET_PKL"]
BACKTEST_MARKET_PKL = CONFIG["paths"].get("BACKTEST_MARKET_PKL", MARKET_PKL)
DIVIDEND_PKL = CONFIG["paths"].get("DIVIDEND_PKL")
OUTPUT_DIR = CONFIG["paths"]["BACKTEST_OUTPUT_DIR"]
INITIAL_CAPITAL = float(CONFIG["backtest"]["INITIAL_CAPITAL"])
LIVE_PROGRESS_FILE = os.environ.get("STOCK_ANALYSIS_PROGRESS_FILE")
MAX_OPEN_TRADE_AMOUNT_RATIO = float(CONFIG["backtest"].get("MAX_OPEN_TRADE_AMOUNT_RATIO", 0.02))
SPECIAL_LIMIT_GAP_TOL = float(CONFIG["backtest"].get("SPECIAL_LIMIT_GAP_TOL", 0.03))
CORP_ACTION_TOL = float(CONFIG["backtest"].get("CORP_ACTION_TOL", 1e-4))
LOT_SIZE = 100


def _json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _reset_live_progress():
    if not LIVE_PROGRESS_FILE:
        return
    os.makedirs(os.path.dirname(LIVE_PROGRESS_FILE), exist_ok=True)
    with open(LIVE_PROGRESS_FILE, "w", encoding="utf-8"):
        pass


def _emit_live_event(event_type: str, payload: dict):
    if not LIVE_PROGRESS_FILE:
        return
    event = {"type": event_type, "payload": payload}
    with open(LIVE_PROGRESS_FILE, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
        fp.flush()


def _build_live_snapshot(record: dict, progress_pct: float, total_days: int, current_idx: int) -> dict:
    portfolio_value = float(record["portfolio_value"])
    return {
        "date": pd.Timestamp(record["date"]).date().isoformat(),
        "cash": float(record["cash"]),
        "portfolio_value": portfolio_value,
        "n_positions": int(record["n_positions"]),
        "n_trades": int(record["n_trades"]),
        "progress_pct": round(float(progress_pct), 4),
        "total_days": int(total_days),
        "current_index": int(current_idx),
        "return_pct": round((portfolio_value / INITIAL_CAPITAL - 1) * 100, 4),
    }


def _max_buy_fill_shares(exec_row: dict, exec_price: float) -> int:
    if exec_price <= 0:
        return 0
    amount = float(exec_row.get("amount", 0) or 0)
    if amount <= 0:
        return 0
    ratio = min(max(MAX_OPEN_TRADE_AMOUNT_RATIO, 0.0), 1.0)
    max_amount = amount * ratio
    return int(max_amount / exec_price / LOT_SIZE) * LOT_SIZE


def _max_sell_fill_shares(exec_row: dict, exec_price: float, held_shares: float) -> float:
    if exec_price <= 0 or held_shares <= 0:
        return 0.0
    amount = float(exec_row.get("amount", 0) or 0)
    if amount <= 0:
        return 0.0
    ratio = min(max(MAX_OPEN_TRADE_AMOUNT_RATIO, 0.0), 1.0)
    max_amount = amount * ratio
    cap_shares = max_amount / exec_price
    return max(0.0, min(float(held_shares), float(cap_shares)))


def _is_special_limit_context(prev_close: float, t1_open: float, code: str, is_st: bool) -> bool:
    if prev_close is None or prev_close <= 0 or t1_open <= 0:
        return False
    limit_up, _ = rules.get_limit_price(prev_close, code, is_st=is_st)
    normal_limit_pct = abs(limit_up / prev_close - 1.0)
    open_move_pct = abs(t1_open / prev_close - 1.0)
    return open_move_pct > normal_limit_pct + max(SPECIAL_LIMIT_GAP_TOL, 0.0)


def _position_cost_total(pos: dict) -> float:
    return float(pos.get("basis_amount", 0.0)) + float(pos.get("buy_cost", 0.0))


def _reference_close_for_limits(exec_row: dict, fallback_prev_close: Optional[float]) -> Optional[float]:
    preclose = float(exec_row.get("preclose", 0) or 0)
    if preclose > 0:
        return preclose
    if fallback_prev_close is None:
        return None
    fallback = float(fallback_prev_close or 0)
    return fallback if fallback > 0 else None


def _prepare_market_for_backtest(market_df: pd.DataFrame, all_dates: list[pd.Timestamp]) -> pd.DataFrame:
    if not all_dates:
        return market_df.iloc[0:0].copy()

    needed_dates = set(all_dates)
    buffer_days = max(int(rules.MIN_PRICE_DAYS) - 1, 0)
    if buffer_days > 0:
        first_date = all_dates[0]
        market_dates = np.sort(market_df["date"].unique())
        history_dates = market_dates[market_dates < first_date]
        if len(history_dates) > 0:
            needed_dates.update(history_dates[-buffer_days:])

    return market_df.loc[market_df["date"].isin(needed_dates)].copy()


def _prepare_action_lookup(dividend_df: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], dict]:
    if dividend_df is None or dividend_df.empty:
        return {}

    events = dividend_df.copy()
    events["operate_date"] = pd.to_datetime(events["operate_date"], errors="coerce").dt.normalize()
    events = events.loc[events["operate_date"].notna()].copy()
    if events.empty:
        return {}

    agg = (
        events.groupby(["code", "operate_date"], as_index=False)
        .agg(
            cash_dividend_ps=("cash_dividend_ps", "sum"),
            stock_ratio=("stock_ratio", "sum"),
            event_text=("event_text", lambda x: " | ".join(sorted({str(v) for v in x if str(v).strip()}))),
        )
    )
    return {
        (row["code"], row["operate_date"]): {
            "cash_dividend_ps": float(row["cash_dividend_ps"] or 0.0),
            "stock_ratio": float(row["stock_ratio"] or 0.0),
            "event_text": row["event_text"],
        }
        for _, row in agg.iterrows()
    }


def load_data():
    print("加载数据...")

    df = pd.read_pickle(CLEAN_PKL)
    price_cols = ["code", "name", "date", "open", "close", "preclose", "pctChg"]
    price_df = df[price_cols].copy()
    del df

    pred_df = pd.read_pickle(PREDICT_PKL)
    pred_merge = pred_df[["date", "code", "pred_return", "pred_std", "confidence"]].copy()
    del pred_df

    price_df["code"] = price_df["code"].astype("category")
    pred_merge["code"] = pred_merge["code"].astype("category")
    all_codes = price_df["code"].cat.categories.union(pred_merge["code"].cat.categories)
    price_df["code"] = price_df["code"].cat.set_categories(all_codes)
    pred_merge["code"] = pred_merge["code"].cat.set_categories(all_codes)

    merged = price_df.merge(pred_merge, on=["date", "code"], how="inner")
    merged = merged.sort_values(["date", "code"]).reset_index(drop=True)
    del price_df, pred_merge

    market_cols = ["code", "date", "isST", "isTrading", "open", "close", "preclose", "amount"]
    market_source = BACKTEST_MARKET_PKL if BACKTEST_MARKET_PKL and os.path.exists(BACKTEST_MARKET_PKL) else MARKET_PKL
    market_df = pd.read_pickle(market_source)[market_cols].copy()

    merged_min_date = merged["date"].min()
    merged_max_date = merged["date"].max()
    if (
        market_source != MARKET_PKL
        and (market_df["date"].min() > merged_min_date or market_df["date"].max() < merged_max_date)
    ):
        market_source = MARKET_PKL
        market_df = pd.read_pickle(MARKET_PKL)[market_cols].copy()

    market_df["code"] = market_df["code"].astype("category")
    market_df["isST"] = market_df["isST"].fillna(0).astype("int8")
    market_df["isTrading"] = market_df["isTrading"].fillna(0).astype("int8")

    dividend_df = load_dividend_events(DIVIDEND_PKL)
    action_lookup = _prepare_action_lookup(dividend_df)

    print(f"合并后数据: {len(merged):,} 行, {merged['code'].nunique()} 只股票")
    print(f"市场状态: {len(market_df):,} 行 (来源: {os.path.basename(market_source)})")
    print(f"公司行为事件: {len(action_lookup):,} 条")
    return merged, market_df, action_lookup


def _apply_corporate_actions(
    exec_date: pd.Timestamp,
    positions: dict,
    exec_market: dict,
    decision_prices: dict,
    action_lookup: dict,
    corp_action_log: list[dict],
    cash: float,
) -> float:
    if not positions:
        return cash

    action_date = pd.Timestamp(exec_date).normalize()
    for code, pos in positions.items():
        event = action_lookup.get((code, action_date))
        exec_row = exec_market.get(code, {})
        prev_close = float(decision_prices.get(code, {}).get("close", pos.get("current_price", 0)) or 0)
        preclose = float(exec_row.get("preclose", 0) or 0)

        if event and (abs(event["cash_dividend_ps"]) > 0 or abs(event["stock_ratio"]) > 0):
            old_shares = float(pos["shares"])
            stock_factor = max(1.0 + float(event["stock_ratio"]), 1e-8)
            cash_dividend = old_shares * float(event["cash_dividend_ps"])
            if abs(cash_dividend) > 0:
                cash += cash_dividend
                pos["cash_dividends_received"] += cash_dividend
            pos["shares"] = old_shares * stock_factor
            pos["ref_price"] = max((float(pos["ref_price"]) - float(event["cash_dividend_ps"])) / stock_factor, 1e-8)
            pos["buy_price"] = pos["ref_price"]
            corp_action_log.append(
                {
                    "date": exec_date,
                    "code": code,
                    "action": "CORP_ACTION_EXPLICIT",
                    "cash_dividend": cash_dividend,
                    "cash_dividend_ps": float(event["cash_dividend_ps"]),
                    "stock_ratio": float(event["stock_ratio"]),
                    "shares_before": old_shares,
                    "shares_after": pos["shares"],
                    "event_text": event.get("event_text", ""),
                }
            )
            continue

        if prev_close > 0 and preclose > 0:
            synthetic_ratio = prev_close / preclose
            if abs(synthetic_ratio - 1.0) > CORP_ACTION_TOL:
                synthetic_cash_dividend_ps = max(prev_close - preclose, 0.0)
                synthetic_cash = float(pos["shares"]) * synthetic_cash_dividend_ps
                if synthetic_cash > 0:
                    cash += synthetic_cash
                    pos["cash_dividends_received"] += synthetic_cash
                pos["ref_price"] = max(float(pos["ref_price"]) - synthetic_cash_dividend_ps, 1e-8)
                pos["buy_price"] = pos["ref_price"]
                corp_action_log.append(
                    {
                        "date": exec_date,
                        "code": code,
                        "action": "CORP_ACTION_SYNTHETIC",
                        "cash_dividend": synthetic_cash,
                        "cash_dividend_ps": synthetic_cash_dividend_ps,
                        "stock_ratio": 0.0,
                        "shares_before": float(pos["shares"]),
                        "shares_after": float(pos["shares"]),
                        "event_text": "synthetic_cash_from_preclose_gap",
                    }
                )

    return cash


def _validate_buy(code: str, market_cache: dict, fallback_prev_close: Optional[float]) -> tuple[bool, str, Optional[float]]:
    if code not in market_cache:
        return False, "NO_DATA", None

    row = market_cache[code]
    if row.get("isTrading") != 1:
        return False, "SUSPENDED", None
    if row.get("isST", 0) == 1 and not bool(rules.ALLOW_ST_BUY):
        return False, "ST_DISABLED", None

    t1_open = float(row.get("open", 0) or 0)
    if t1_open <= 0:
        return False, "INVALID_PRICE", None
    if float(row.get("amount", 0) or 0) < rules.MIN_EXEC_AMOUNT:
        return False, "LOW_LIQUIDITY", None

    ref_close = _reference_close_for_limits(row, fallback_prev_close)
    if ref_close is not None:
        is_st = row.get("isST", 0) == 1
        if _is_special_limit_context(ref_close, t1_open, code, is_st):
            return True, "OK", t1_open
        limit_up, _ = rules.get_limit_price(ref_close, code, is_st=is_st)
        if t1_open >= limit_up - 0.001:
            return False, "LIMIT_UP", None

    return True, "OK", t1_open


def _validate_sell(code: str, market_cache: dict, fallback_prev_close: Optional[float], sell_reason: str = "") -> tuple[bool, str, Optional[float]]:
    if code not in market_cache:
        if sell_reason == "DELIST_FORCE_SELL":
            return True, "OK", 0.0
        return False, "NO_DATA", None

    row = market_cache[code]
    if row.get("isTrading") != 1:
        return False, "SUSPENDED", None

    t1_open = float(row.get("open", 0) or 0)
    if t1_open <= 0:
        return False, "INVALID_PRICE", None

    if sell_reason != "DELIST_FORCE_SELL":
        ref_close = _reference_close_for_limits(row, fallback_prev_close)
        if ref_close is not None:
            is_st = row.get("isST", 0) == 1
            if _is_special_limit_context(ref_close, t1_open, code, is_st):
                return True, "OK", t1_open
            _, limit_down = rules.get_limit_price(ref_close, code, is_st=is_st)
            if t1_open <= limit_down + 0.001:
                return False, "LIMIT_DOWN", None

    return True, "OK", t1_open


def _build_position_record(date, code, pos, price_cache: dict, primary_date, fallback_date=None):
    name = ""
    for dt in [primary_date, fallback_date]:
        if dt is None:
            continue
        cache = price_cache.get(dt, {})
        if code in cache:
            name = cache[code].get("name", "")
            if name:
                break

    total_cost = _position_cost_total(pos)
    float_profit = float(pos["shares"]) * float(pos["current_price"]) + float(pos.get("cash_dividends_received", 0.0)) - total_cost
    float_profit_pct = float_profit / total_cost if total_cost > 0 else 0.0
    return {
        "date": date,
        "code": code,
        "name": name,
        "buy_price": pos["buy_price"],
        "buy_date": pos["buy_date"],
        "hold_days": pos["hold_days"],
        "current_price": pos["current_price"],
        "shares": pos["shares"],
        "basis_amount": pos.get("basis_amount", np.nan),
        "buy_cost": pos.get("buy_cost", np.nan),
        "cash_dividends_received": pos.get("cash_dividends_received", 0.0),
        "market_value": float(pos["shares"]) * float(pos["current_price"]),
        "float_profit": float_profit,
        "float_profit_pct": float_profit_pct,
    }


def run_backtest(merged: pd.DataFrame, market_df: pd.DataFrame, action_lookup: dict, scoring_method="confidence_weighted") -> dict:
    all_dates = sorted(merged["date"].unique())
    print(f"回测期间: {all_dates[0].date()} ~ {all_dates[-1].date()}, 共 {len(all_dates)} 个交易日")

    market_df = _prepare_market_for_backtest(market_df, all_dates)
    print(f"市场状态裁剪后: {len(market_df):,} 行, {market_df['date'].nunique()} 个交易日")

    date_grouped = merged.groupby("date")
    market_grouped = market_df.groupby("date")
    next_date_map = {all_dates[i]: all_dates[i + 1] for i in range(len(all_dates) - 1)}
    daily_mkt_ret = merged.groupby("date")["pctChg"].mean().sort_index()

    print("预处理价格数据...")
    price_cache = {
        date: group.set_index("code")[["close", "open", "name"]].to_dict("index")
        for date, group in merged.groupby("date", sort=False)
    }
    print("预处理市场状态...")
    market_cache = {
        date: group.set_index("code")[["open", "close", "preclose", "isTrading", "isST", "amount"]].to_dict("index")
        for date, group in market_df.groupby("date", sort=False)
    }

    cash = INITIAL_CAPITAL
    positions: dict[str, dict] = {}
    daily_records = []
    trade_log = []
    position_log = []
    corp_action_log = []

    daily_records.append(
        {
            "date": all_dates[0],
            "cash": INITIAL_CAPITAL,
            "portfolio_value": INITIAL_CAPITAL,
            "n_positions": 0,
            "n_trades": 0,
        }
    )
    _emit_live_event("equity", _build_live_snapshot(daily_records[-1], 0.0, len(all_dates), 0))

    print("开始回测模拟...")
    progress_interval = max(1, len(all_dates) // 20)
    for day_idx, decision_date in enumerate(all_dates):
        exec_date = next_date_map.get(decision_date)

        try:
            decision_data = date_grouped.get_group(decision_date)
        except KeyError:
            decision_data = pd.DataFrame()

        if exec_date is None:
            total_value = cash
            for code, pos in positions.items():
                current_price = price_cache.get(decision_date, {}).get(code, {}).get("close", pos["current_price"])
                pos["current_price"] = current_price
                total_value += float(pos["shares"]) * float(current_price)
                position_log.append(_build_position_record(decision_date, code, pos, price_cache, decision_date))

            if not daily_records or daily_records[-1]["date"] != decision_date:
                daily_records.append(
                    {
                        "date": decision_date,
                        "cash": cash,
                        "portfolio_value": total_value,
                        "n_positions": len(positions),
                        "n_trades": 0,
                    }
                )
                progress_pct = (day_idx + 1) / len(all_dates) * 100
                _emit_live_event("equity", _build_live_snapshot(daily_records[-1], progress_pct, len(all_dates), day_idx + 1))
            continue

        for code in positions:
            positions[code]["hold_days"] += 1
        n_trades_today = 0

        exec_market = market_cache.get(exec_date, {})
        decision_prices = price_cache.get(decision_date, {})
        cash = _apply_corporate_actions(exec_date, positions, exec_market, decision_prices, action_lookup, corp_action_log, cash)

        sell_list = rules.decide_sells(positions, decision_data)
        actually_sold = set()
        if sell_list:
            for code, sell_reason in sell_list:
                if code not in positions:
                    continue
                pos = positions[code]
                fallback_prev_close = decision_prices.get(code, {}).get("close")
                stock_name = decision_prices.get(code, {}).get("name", "")

                if sell_reason == "DELIST_FORCE_SELL":
                    exec_row = exec_market.get(code, {})
                    t1_open = float(exec_row.get("open", 0) or 0)
                    can_trade = exec_row.get("isTrading") == 1 and t1_open > 0
                    exec_price = t1_open if can_trade else 0.0
                    shares = float(pos["shares"])
                    sell_amount = shares * exec_price
                    sell_cost = rules.calc_sell_cost(sell_amount) if exec_price > 0 else 0.0
                    dividend_cash = float(pos.get("cash_dividends_received", 0.0))
                    basis_amount = float(pos.get("basis_amount", 0.0))
                    buy_cost_alloc = float(pos.get("buy_cost", 0.0))
                    if exec_price > 0:
                        cash += sell_amount - sell_cost
                    profit = sell_amount - sell_cost + dividend_cash - basis_amount - buy_cost_alloc
                    denom = basis_amount + buy_cost_alloc
                    profit_pct_val = profit / denom if denom > 0 else 0.0
                    trade_log.append(
                        {
                            "date": exec_date,
                            "code": code,
                            "name": stock_name,
                            "action": "SELL",
                            "price": exec_price,
                            "shares": shares,
                            "amount": sell_amount,
                            "cost": sell_cost,
                            "profit": profit,
                            "profit_pct": profit_pct_val,
                            "dividend_cash": dividend_cash,
                            "reason": "DELIST_FORCE_SELL" if can_trade else "DELIST_WRITE_OFF",
                            "hold_days": pos["hold_days"],
                        }
                    )
                    n_trades_today += 1
                    actually_sold.add(code)
                    del positions[code]
                    continue

                can_sell, fail_reason, t1_open = _validate_sell(code, exec_market, fallback_prev_close, sell_reason)
                if not can_sell:
                    trade_log.append(
                        {
                            "date": exec_date,
                            "code": code,
                            "name": stock_name,
                            "action": f"SELL_FAILED_{fail_reason}",
                            "price": 0,
                            "shares": pos["shares"],
                            "amount": 0,
                            "cost": 0,
                            "profit": 0,
                            "profit_pct": np.nan,
                            "dividend_cash": 0.0,
                            "reason": f"{sell_reason}_BLOCKED_{fail_reason}",
                            "hold_days": pos["hold_days"],
                        }
                    )
                    continue

                exec_row = exec_market.get(code, {})
                exec_price = float(t1_open or 0)
                max_fill_shares = _max_sell_fill_shares(exec_row, exec_price, pos["shares"])
                if max_fill_shares <= 0:
                    trade_log.append(
                        {
                            "date": exec_date,
                            "code": code,
                            "name": stock_name,
                            "action": "SELL_FAILED_CAPACITY",
                            "price": exec_price,
                            "shares": pos["shares"],
                            "amount": 0,
                            "cost": 0,
                            "profit": 0,
                            "profit_pct": np.nan,
                            "dividend_cash": 0.0,
                            "reason": f"{sell_reason}_BLOCKED_CAPACITY",
                            "hold_days": pos["hold_days"],
                        }
                    )
                    continue

                filled_shares = min(float(pos["shares"]), float(max_fill_shares))
                if filled_shares < 1e-8:
                    continue

                original_shares = float(pos["shares"])
                filled_ratio = filled_shares / original_shares
                basis_alloc = float(pos["basis_amount"]) * filled_ratio
                buy_cost_alloc = float(pos["buy_cost"]) * filled_ratio
                dividend_alloc = float(pos.get("cash_dividends_received", 0.0)) * filled_ratio
                sell_amount = filled_shares * exec_price
                sell_cost = rules.calc_sell_cost(sell_amount) if exec_price > 0 else 0.0
                cash += sell_amount - sell_cost
                profit = sell_amount - sell_cost + dividend_alloc - basis_alloc - buy_cost_alloc
                denom = basis_alloc + buy_cost_alloc
                profit_pct_val = profit / denom if denom > 0 else 0.0
                is_partial = filled_shares + 1e-8 < original_shares

                trade_log.append(
                    {
                        "date": exec_date,
                        "code": code,
                        "name": stock_name,
                        "action": "SELL",
                        "price": exec_price,
                        "shares": filled_shares,
                        "amount": sell_amount,
                        "cost": sell_cost,
                        "profit": profit,
                        "profit_pct": profit_pct_val,
                        "dividend_cash": dividend_alloc,
                        "reason": f"{sell_reason}_PARTIAL" if is_partial else sell_reason,
                        "hold_days": pos["hold_days"],
                    }
                )
                n_trades_today += 1

                if is_partial:
                    pos["shares"] = original_shares - filled_shares
                    pos["basis_amount"] = max(float(pos["basis_amount"]) - basis_alloc, 0.0)
                    pos["buy_cost"] = max(float(pos["buy_cost"]) - buy_cost_alloc, 0.0)
                    pos["cash_dividends_received"] = float(pos.get("cash_dividends_received", 0.0)) - dividend_alloc
                else:
                    actually_sold.add(code)
                    del positions[code]

        mkt_factor = rules.check_market_regime(daily_mkt_ret, decision_date)
        n_slots = rules.compute_buy_slots(len(positions), mkt_factor)

        if n_slots > 0 and not decision_data.empty:
            try:
                market_day = market_grouped.get_group(decision_date)
            except KeyError:
                market_day = pd.DataFrame()

            if not market_day.empty:
                pool = rules.filter_stock_pool(decision_data, market_day, market_history=market_df)
                buy_codes = rules.select_buys(pool, set(positions.keys()), actually_sold, n_slots, scoring_method)
                if buy_codes:
                    buy_pool = pool.loc[pool["code"].isin(buy_codes)].copy()
                    allocation = rules.compute_weighted_allocation(cash, buy_pool)

                    for code in buy_codes:
                        if code not in allocation:
                            continue

                        fallback_prev_close = decision_prices.get(code, {}).get("close")
                        can_buy, fail_reason, t1_open = _validate_buy(code, exec_market, fallback_prev_close)
                        stock_name = decision_prices.get(code, {}).get("name", "")
                        if not can_buy:
                            trade_log.append(
                                {
                                    "date": exec_date,
                                    "code": code,
                                    "name": stock_name,
                                    "action": f"BUY_FAILED_{fail_reason}",
                                    "price": t1_open or 0,
                                    "shares": 0,
                                    "amount": 0,
                                    "cost": 0,
                                    "profit": 0,
                                    "profit_pct": np.nan,
                                    "dividend_cash": 0.0,
                                    "reason": f"{fail_reason}_BLOCKED",
                                    "hold_days": 0,
                                }
                            )
                            continue

                        allocated_cash = float(allocation[code])
                        exec_row = exec_market.get(code, {})
                        exec_price = float(t1_open or 0)
                        max_shares_cash = int(allocated_cash / exec_price / LOT_SIZE) * LOT_SIZE
                        max_shares_capacity = _max_buy_fill_shares(exec_row, exec_price)
                        shares = min(max_shares_cash, max_shares_capacity)
                        if shares < LOT_SIZE:
                            trade_log.append(
                                {
                                    "date": exec_date,
                                    "code": code,
                                    "name": stock_name,
                                    "action": "BUY_FAILED_CAPACITY",
                                    "price": exec_price,
                                    "shares": 0,
                                    "amount": 0,
                                    "cost": 0,
                                    "profit": 0,
                                    "profit_pct": np.nan,
                                    "dividend_cash": 0.0,
                                    "reason": "CAPACITY_BLOCKED",
                                    "hold_days": 0,
                                }
                            )
                            continue

                        buy_amount = shares * exec_price
                        buy_cost = rules.calc_buy_cost(buy_amount)
                        total_cost = buy_amount + buy_cost
                        if total_cost > cash:
                            continue

                        cash -= total_cost
                        is_partial_fill = shares < max_shares_cash
                        positions[code] = {
                            "shares": float(shares),
                            "buy_price": float(exec_price),
                            "ref_price": float(exec_price),
                            "basis_amount": float(buy_amount),
                            "buy_cost": float(buy_cost),
                            "buy_date": exec_date,
                            "current_price": float(exec_price),
                            "hold_days": 0,
                            "cash_dividends_received": 0.0,
                        }
                        trade_log.append(
                            {
                                "date": exec_date,
                                "code": code,
                                "name": stock_name,
                                "action": "BUY",
                                "price": exec_price,
                                "shares": shares,
                                "amount": buy_amount,
                                "cost": buy_cost,
                                "profit": 0,
                                "profit_pct": np.nan,
                                "dividend_cash": 0.0,
                                "reason": "SIGNAL_PARTIAL" if is_partial_fill else "SIGNAL",
                                "hold_days": 0,
                            }
                        )
                        n_trades_today += 1

        total_value = cash
        exec_prices = price_cache.get(exec_date, {})
        for code, pos in positions.items():
            current_price = exec_prices.get(code, {}).get("close")
            if current_price is None:
                current_price = decision_prices.get(code, {}).get("close", pos["current_price"])
            pos["current_price"] = float(current_price)
            total_value += float(pos["shares"]) * float(current_price)
            position_log.append(
                _build_position_record(exec_date, code, pos, price_cache, exec_date, fallback_date=decision_date)
            )

        daily_records.append(
            {
                "date": exec_date,
                "cash": cash,
                "portfolio_value": total_value,
                "n_positions": len(positions),
                "n_trades": n_trades_today,
            }
        )
        progress_pct = (day_idx + 1) / len(all_dates) * 100
        _emit_live_event("equity", _build_live_snapshot(daily_records[-1], progress_pct, len(all_dates), day_idx + 1))

        if day_idx % progress_interval == 0 or day_idx == len(all_dates) - 1:
            ret = (total_value / INITIAL_CAPITAL - 1) * 100
            print(
                f"  进度: {progress_pct:.1f}% ({day_idx + 1}/{len(all_dates)}) | "
                f"日期: {decision_date.date()} | 收益: {ret:+.2f}% | 持仓: {len(positions)}只"
            )

    daily_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_log)
    position_df = pd.DataFrame(position_log)
    corp_action_df = pd.DataFrame(corp_action_log)

    return {
        "daily": daily_df,
        "trades": trade_df,
        "positions_log": position_df,
        "corp_actions": corp_action_df,
        "final_value": daily_df.iloc[-1]["portfolio_value"],
        "positions": positions,
    }


def compute_metrics(daily_df: pd.DataFrame) -> dict:
    daily_df = daily_df.copy()
    daily_df["daily_return"] = daily_df["portfolio_value"].pct_change()

    total_days = len(daily_df)
    total_trading_days = daily_df["date"].nunique()
    total_return = daily_df.iloc[-1]["portfolio_value"] / INITIAL_CAPITAL - 1
    annual_return = (1 + total_return) ** (252 / max(total_trading_days, 1)) - 1

    rf_daily = 0.025 / 252
    daily_returns = daily_df["daily_return"].dropna()
    excess_returns = daily_returns - rf_daily
    sharpe = excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(252)

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-10
    sortino = excess_returns.mean() / (downside_std + 1e-10) * np.sqrt(252)

    cummax = daily_df["portfolio_value"].cummax()
    drawdown = (daily_df["portfolio_value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    annual_vol = daily_returns.std() * np.sqrt(252)
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "初始资金": f"{INITIAL_CAPITAL:,.0f}",
        "期末资产": f"{daily_df.iloc[-1]['portfolio_value']:,.0f}",
        "总回报率": f"{total_return:.2%}",
        "年化收益率": f"{annual_return:.2%}",
        "夏普比率": f"{sharpe:.3f}",
        "Sortino比率": f"{sortino:.3f}",
        "最大回撤": f"{max_drawdown:.2%}",
        "最大回撤持续天数": max_dd_duration,
        "年化波动率": f"{annual_vol:.2%}",
        "Calmar比率": f"{calmar:.3f}",
        "交易天数": total_days,
    }


def compute_trade_metrics(trade_df: pd.DataFrame) -> dict:
    if len(trade_df) == 0:
        return {}

    buys = trade_df[trade_df["action"] == "BUY"]
    sells = trade_df[trade_df["action"] == "SELL"]
    failed = trade_df[trade_df["action"].str.startswith(("BUY_FAILED", "SELL_FAILED"), na=False)]
    fail_counts = failed["action"].value_counts().to_dict() if len(failed) > 0 else {}

    if len(sells) == 0:
        result = {"总交易笔数(买入)": len(buys)}
        result.update(fail_counts)
        return result

    wins = sells[sells["profit"] > 0]
    losses = sells[sells["profit"] <= 0]

    win_rate = len(wins) / len(sells)
    avg_win = wins["profit"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["profit"].mean()) if len(losses) > 0 else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    total_days = (trade_df["date"].max() - trade_df["date"].min()).days
    sell_holds = sells["hold_days"]
    avg_hold = sell_holds.mean() if len(sell_holds) > 0 else 0

    reason_stats = ""
    if "reason" in sells.columns:
        reasons = sells["reason"].value_counts()
        reason_stats = ", ".join([f"{k}:{v}" for k, v in reasons.items()])

    result = {
        "总交易笔数(买入)": len(buys),
        "总交易笔数(卖出)": len(sells),
        "胜率": f"{win_rate:.2%}",
        "盈亏比": f"{profit_loss_ratio:.3f}",
        "平均盈利": f"{avg_win:,.0f}",
        "平均亏损": f"{-losses['profit'].mean() if len(losses) > 0 else 0:,.0f}",
        "平均持有天数": f"{avg_hold:.1f}",
        "日均交易次数": f"{len(trade_df) / max(total_days, 1) * 365 / 252:.2f}",
        "卖出原因": reason_stats,
    }
    if fail_counts:
        result["执行失败统计"] = ", ".join(f"{k}:{v}" for k, v in fail_counts.items())
    return result


def run_pipeline(scoring_method="confidence_weighted"):
    _reset_live_progress()
    _emit_live_event("status", {"stage": "load_data", "message": "开始加载回测数据"})
    merged, market_df, action_lookup = load_data()
    _emit_live_event("status", {"stage": "backtest", "message": "开始执行回测模拟"})
    results = run_backtest(merged, market_df, action_lookup, scoring_method=scoring_method)

    daily_df = results["daily"]
    trade_df = results["trades"]
    position_df = results["positions_log"]
    corp_action_df = results["corp_actions"]

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
    daily_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_daily.csv"), index=False)
    trade_df.to_csv(os.path.join(OUTPUT_DIR, "trade_log.csv"), index=False)
    position_df.to_csv(os.path.join(OUTPUT_DIR, "position_log.csv"), index=False)
    corp_action_df.to_csv(os.path.join(OUTPUT_DIR, "corp_action_log.csv"), index=False)

    md_path = os.path.join(OUTPUT_DIR, "backtest_metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        start_date = daily_df.iloc[0]["date"].date()
        end_date = daily_df.iloc[-1]["date"].date()
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

        if len(corp_action_df) > 0:
            f.write("## 公司行为处理\n\n")
            f.write(f"本次回测共处理 **{len(corp_action_df)}** 条持仓层公司行为调整，详见 `corp_action_log.csv`。\n\n")

        f.write("---\n\n")
        f.write("*以上为历史回测结果，不代表未来收益，仅供参考。*\n")

    print(f"\n回测结果已保存至 {OUTPUT_DIR}/")
    print("   - backtest_daily.csv  (每日资产)")
    print("   - trade_log.csv       (交易日志)")
    print("   - position_log.csv    (每日持仓快照)")
    print("   - corp_action_log.csv (公司行为处理日志)")
    print("   - backtest_metrics.md (回测指标)")
    _emit_live_event(
        "summary",
        {
            "metrics": metrics,
            "trade_metrics": trade_metrics,
            "corp_action_count": int(len(corp_action_df)),
            "output_dir": OUTPUT_DIR,
        },
    )

    return results, metrics, trade_metrics


if __name__ == "__main__":
    run_pipeline()
