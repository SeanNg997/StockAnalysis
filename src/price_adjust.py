"""Raw price -> point-in-time adjusted research price helpers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from config import CONFIG


DIVIDEND_PKL = CONFIG["paths"]["DIVIDEND_PKL"]


def build_pt_adjusted_prices(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Build a stable total-return price path from raw daily bars.

    We anchor the research price path to historical daily total returns
    (`pctChg` / `preclose`) instead of vendor-provided forward-adjusted bars.
    That keeps historical features stable when future dividends happen.
    """

    if stock_df.empty:
        return pd.DataFrame(
            columns=["adj_open", "adj_high", "adj_low", "adj_close", "pt_adjust_factor"],
            index=stock_df.index,
        )

    g = stock_df.sort_values("date").copy()
    raw_open = pd.to_numeric(g["open"], errors="coerce").to_numpy(dtype=np.float64)
    raw_high = pd.to_numeric(g["high"], errors="coerce").to_numpy(dtype=np.float64)
    raw_low = pd.to_numeric(g["low"], errors="coerce").to_numpy(dtype=np.float64)
    raw_close = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=np.float64)
    preclose = pd.to_numeric(g.get("preclose"), errors="coerce").to_numpy(dtype=np.float64)
    pct_chg = pd.to_numeric(g.get("pctChg"), errors="coerce").to_numpy(dtype=np.float64)

    n = len(g)
    adj_close = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return pd.DataFrame(index=g.index)

    first_valid = None
    for i, price in enumerate(raw_close):
        if np.isfinite(price) and price > 0:
            first_valid = i
            break

    if first_valid is None:
        factor = np.ones(n, dtype=np.float64)
        return pd.DataFrame(
            {
                "adj_open": raw_open,
                "adj_high": raw_high,
                "adj_low": raw_low,
                "adj_close": raw_close,
                "pt_adjust_factor": factor,
            },
            index=g.index,
        )

    adj_close[first_valid] = raw_close[first_valid]
    for i in range(first_valid + 1, n):
        prev_adj = adj_close[i - 1]
        if not np.isfinite(prev_adj) or prev_adj <= 0:
            prev_adj = raw_close[i - 1] if np.isfinite(raw_close[i - 1]) and raw_close[i - 1] > 0 else raw_close[first_valid]

        if np.isfinite(pct_chg[i]):
            growth = 1.0 + pct_chg[i] / 100.0
        elif np.isfinite(preclose[i]) and preclose[i] > 0 and np.isfinite(raw_close[i]) and raw_close[i] > 0:
            growth = raw_close[i] / preclose[i]
        elif np.isfinite(raw_close[i - 1]) and raw_close[i - 1] > 0 and np.isfinite(raw_close[i]) and raw_close[i] > 0:
            growth = raw_close[i] / raw_close[i - 1]
        else:
            growth = 1.0

        growth = max(growth, 1e-8)
        adj_close[i] = prev_adj * growth

    # Keep the pre-history segment on the same scale as the first usable bar.
    for i in range(first_valid - 1, -1, -1):
        adj_close[i] = adj_close[i + 1]

    factor = np.ones(n, dtype=np.float64)
    valid_close = np.isfinite(raw_close) & (raw_close > 0)
    factor[valid_close] = adj_close[valid_close] / raw_close[valid_close]
    for i in range(1, n):
        if not np.isfinite(factor[i]) or factor[i] <= 0:
            factor[i] = factor[i - 1]
    if not np.isfinite(factor[0]) or factor[0] <= 0:
        factor[0] = 1.0

    adj_open = raw_open * factor
    adj_high = raw_high * factor
    adj_low = raw_low * factor

    return pd.DataFrame(
        {
            "adj_open": adj_open.astype(np.float32),
            "adj_high": adj_high.astype(np.float32),
            "adj_low": adj_low.astype(np.float32),
            "adj_close": adj_close.astype(np.float32),
            "pt_adjust_factor": factor.astype(np.float32),
        },
        index=g.index,
    )


def normalize_dividend_events(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["code", "operate_date", "cash_dividend_ps", "stock_ratio", "event_text"]
        )

    out = df.copy()
    out["operate_date"] = pd.to_datetime(out["dividOperateDate"], errors="coerce")
    out["cash_dividend_ps"] = pd.to_numeric(out["dividCashPsBeforeTax"], errors="coerce").fillna(0.0)
    stock_ps = pd.to_numeric(out["dividStocksPs"], errors="coerce").fillna(0.0)
    reserve_ps = pd.to_numeric(out["dividReserveToStockPs"], errors="coerce").fillna(0.0)
    out["stock_ratio"] = stock_ps + reserve_ps
    out["event_text"] = out.get("dividCashStock", "").fillna("")
    out = out[out["operate_date"].notna()].copy()
    out = out[["code", "operate_date", "cash_dividend_ps", "stock_ratio", "event_text"]]
    out = out.drop_duplicates(subset=["code", "operate_date"], keep="last")
    out = out.sort_values(["code", "operate_date"]).reset_index(drop=True)
    return out


def load_dividend_events(path: str | None = None) -> pd.DataFrame:
    dividend_path = path or DIVIDEND_PKL
    if not dividend_path or not os.path.exists(dividend_path):
        return normalize_dividend_events(pd.DataFrame())
    return normalize_dividend_events(pd.read_pickle(dividend_path))
