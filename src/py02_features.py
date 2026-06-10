"""特征工程 — 基于原始行情构建稳定点时复权研究价格并生成标签"""

from __future__ import annotations

import gc
import os
import warnings

import numpy as np
import pandas as pd

from config import CONFIG
from price_adjust import build_pt_adjusted_prices

warnings.filterwarnings("ignore")

CLEAN_PKL = CONFIG["paths"]["CLEAN_PKL"]
FEATURE_PKL = CONFIG["paths"]["FEATURE_PKL"]
TRADE_DAYS_TXT = CONFIG["paths"]["TRADE_DAYS_TXT"]


def compute_features_for_stock(g: pd.DataFrame, trade_day_idx: dict | None = None) -> pd.DataFrame:
    """对单只股票计算所有特征（输入须按 date 升序，OHLC 为研究价）。"""
    close = g["close"].to_numpy(dtype=np.float32)
    open_ = g["open"].to_numpy(dtype=np.float32)
    high = g["high"].to_numpy(dtype=np.float32)
    low = g["low"].to_numpy(dtype=np.float32)
    volume = g["volume"].to_numpy(dtype=np.float32)
    amount = g["amount"].to_numpy(dtype=np.float32)
    turn = g["turn"].to_numpy(dtype=np.float32)
    is_trading = g["isTrading"].fillna(0).to_numpy(dtype=np.int8)
    n = len(g)

    feats: dict[str, np.ndarray] = {}

    consecutive_suspend = np.zeros(n, dtype=np.int16)
    current_suspend = 0
    for i in range(n):
        if is_trading[i] != 1:
            current_suspend += 1
        else:
            current_suspend = 0
        consecutive_suspend[i] = current_suspend
    feats["consecutive_suspend"] = consecutive_suspend

    recent_5d_suspend = np.zeros(n, dtype=np.int8)
    for i in range(n):
        start = max(0, i - 4)
        recent_5d_suspend[i] = 1 if np.any(is_trading[start : i + 1] != 1) else 0
    feats["recent_5d_suspend"] = recent_5d_suspend

    for period in [1, 3, 5, 10, 20, 60]:
        ret = np.full(n, np.nan, dtype=np.float32)
        if n > period:
            ret[period:] = close[period:] / np.maximum(close[:-period], 1e-10) - 1.0
        feats[f"ret_{period}d"] = ret

    log_ret = np.full(n, np.nan, dtype=np.float32)
    if n > 1:
        log_ret[1:] = np.log(np.maximum(close[1:], 1e-10) / np.maximum(close[:-1], 1e-10))
    feats["log_ret_1d"] = log_ret

    def ewma(x: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1)
        out = np.zeros(len(x), dtype=np.float32)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        return out

    for w in [5, 10, 20, 60]:
        ma = pd.Series(close).rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32)
        feats[f"ma_{w}"] = ma
        feats[f"ma_bias_{w}"] = (close - ma) / (ma + 1e-10)

    feats["ma_bull"] = (
        (feats["ma_5"] > feats["ma_10"]).astype(np.float32)
        + (feats["ma_10"] > feats["ma_20"]).astype(np.float32)
        + (feats["ma_20"] > feats["ma_60"]).astype(np.float32)
    ) / 3.0

    ema12 = ewma(close, 12)
    ema26 = ewma(close, 26)
    dif = ema12 - ema26
    dea = ewma(dif, 9)
    macd_hist = 2.0 * (dif - dea)
    feats["macd_dif"] = dif / (close + 1e-10)
    feats["macd_dea"] = dea / (close + 1e-10)
    feats["macd_hist"] = macd_hist / (close + 1e-10)

    def wilder_ema(x: np.ndarray, period: int) -> np.ndarray:
        alpha = 1.0 / period
        out = np.zeros(len(x), dtype=np.float32)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        return out

    rsi_delta = np.zeros(n, dtype=np.float32)
    if n > 1:
        rsi_delta[1:] = close[1:] - close[:-1]
    rsi_gain = np.where(rsi_delta > 0, rsi_delta, 0.0)
    rsi_loss = np.where(rsi_delta < 0, -rsi_delta, 0.0)
    for period in [6, 12, 24]:
        avg_gain = wilder_ema(rsi_gain, period)
        avg_loss = wilder_ema(rsi_loss, period)
        rs = avg_gain / (avg_loss + 1e-10)
        feats[f"rsi_{period}"] = 100.0 - 100.0 / (1.0 + rs)

    bb_period = 20
    bb_ma = pd.Series(close).rolling(bb_period, min_periods=1).mean().to_numpy(dtype=np.float32)
    bb_std = pd.Series(close).rolling(bb_period, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float32)
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    feats["bb_pctb"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    feats["bb_width"] = (bb_upper - bb_lower) / (bb_ma + 1e-10)

    hl = high - low
    tr = np.empty(n, dtype=np.float32)
    tr[0] = hl[0]
    if n > 1:
        tr[1:] = np.maximum(
            hl[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
    atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy(dtype=np.float32)
    feats["atr14_ratio"] = atr14 / (close + 1e-10)

    kdj_period = 9
    low_min = pd.Series(low).rolling(kdj_period, min_periods=1).min().to_numpy(dtype=np.float32)
    high_max = pd.Series(high).rolling(kdj_period, min_periods=1).max().to_numpy(dtype=np.float32)
    rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100.0
    k = pd.Series(rsv).ewm(com=2, adjust=False).mean().to_numpy(dtype=np.float32)
    d = pd.Series(k).ewm(com=2, adjust=False).mean().to_numpy(dtype=np.float32)
    j = 3 * k - 2 * d
    feats["kdj_k"] = k
    feats["kdj_d"] = d
    feats["kdj_j"] = j

    vol_ma5 = pd.Series(volume).rolling(5, min_periods=1).mean().to_numpy(dtype=np.float32)
    feats["vol_ratio"] = volume / (vol_ma5 + 1e-10)

    obv_sign = np.sign(np.diff(close, prepend=close[0])).astype(np.float32)
    obv = np.cumsum(obv_sign * volume, dtype=np.float64)
    obv_ma = pd.Series(obv).rolling(10, min_periods=1).mean().to_numpy(dtype=np.float32)
    feats["obv_diff"] = (obv.astype(np.float32) - obv_ma) / (np.abs(obv_ma) + 1e-10)

    turn_ma5 = pd.Series(turn).rolling(5, min_periods=1).mean().to_numpy(dtype=np.float32)
    feats["turn_ratio"] = turn / (turn_ma5 + 1e-10)
    feats["turn_ma5"] = turn_ma5

    amt_ma20 = pd.Series(amount).rolling(20, min_periods=1).mean().to_numpy(dtype=np.float32)
    feats["amt_ratio"] = amount / (amt_ma20 + 1e-10)

    log_ret_s = pd.Series(feats["log_ret_1d"])
    for w in [5, 10, 20]:
        feats[f"volatility_{w}d"] = log_ret_s.rolling(w, min_periods=2).std().to_numpy(dtype=np.float32)

    pos_ret = log_ret_s.clip(lower=0)
    neg_ret = log_ret_s.clip(upper=0)
    feats["upside_vol_20"] = pos_ret.rolling(20, min_periods=2).std().to_numpy(dtype=np.float32)
    feats["downside_vol_20"] = neg_ret.rolling(20, min_periods=2).std().to_numpy(dtype=np.float32)

    rolling_max = pd.Series(close).rolling(20, min_periods=1).max().to_numpy(dtype=np.float32)
    feats["drawdown_20d"] = (close - rolling_max) / (rolling_max + 1e-10)

    for w in [10, 20, 60]:
        h = pd.Series(high).rolling(w, min_periods=1).max().to_numpy(dtype=np.float32)
        l = pd.Series(low).rolling(w, min_periods=1).min().to_numpy(dtype=np.float32)
        feats[f"price_pos_{w}d"] = (close - l) / (h - l + 1e-10)

    for col_name in ["peTTM", "pbMRQ", "psTTM"]:
        vals = pd.to_numeric(g[col_name], errors="coerce").to_numpy(dtype=np.float32)
        ma20 = pd.Series(vals).rolling(20, min_periods=1).mean().to_numpy(dtype=np.float32)
        feats[f"{col_name}_chg"] = (vals - ma20) / (np.abs(ma20) + 1e-10)

    feats["reversal_5d"] = -feats["ret_5d"]
    safe_vol = np.where(np.isnan(feats["volatility_5d"]) | (np.abs(feats["volatility_5d"]) < 1e-10), 1e-10, feats["volatility_5d"])
    feats["risk_adj_mom_20d"] = feats["ret_20d"] / safe_vol

    amt_5ma = pd.Series(amount).rolling(5, min_periods=1).mean().to_numpy(dtype=np.float32)
    amt_20ma = pd.Series(amount).rolling(20, min_periods=1).mean().to_numpy(dtype=np.float32)
    feats["amt_trend"] = amt_5ma / (amt_20ma + 1e-10)

    ret_1d_s = pd.Series(feats["ret_1d"])
    feats["momentum_acc"] = (
        ret_1d_s.rolling(5, min_periods=2).mean() - ret_1d_s.rolling(20, min_periods=5).mean()
    ).to_numpy(dtype=np.float32)

    price_trend = pd.Series(close).pct_change(5).to_numpy(dtype=np.float32)
    vol_trend = pd.Series(volume).pct_change(5).to_numpy(dtype=np.float32)
    feats["vol_price_diverge"] = np.where(
        np.isnan(price_trend) | np.isnan(vol_trend),
        np.nan,
        price_trend - vol_trend,
    ).astype(np.float32)

    total_range = high - low + 1e-10
    feats["upper_shadow"] = (high - np.maximum(close, open_)) / total_range
    feats["lower_shadow"] = (np.minimum(close, open_) - low) / total_range

    hold_days = int(CONFIG["features"]["HOLD_DAYS"])
    sell_offset = hold_days + 1

    # 标签使用的是“研究价 open”而不是原始 open。
    # 这里的研究价来自同一套点时复权因子，因此 open 比值表示的是
    # 持有期间的总回报口径（含分红/送转），与回测里 raw open + 公司行为入账
    # 在设计目标上保持一致。
    open_next_1 = np.roll(open_, -1)
    open_next_1[-1] = np.nan
    open_next_n = np.roll(open_, -sell_offset)
    open_next_n[-sell_offset:] = np.nan
    if n <= sell_offset:
        open_next_n[:] = np.nan

    buy_cost = float(CONFIG["features"]["BUY_COST"])
    sell_cost = float(CONFIG["features"]["SELL_COST"])
    raw_label = (open_next_n * sell_cost) / (open_next_1 * buy_cost + 1e-10) - 1.0

    if trade_day_idx is not None:
        dates = g["date"].to_numpy()
        for i in range(n - sell_offset):
            buy_date = dates[i + 1]
            sell_date = dates[i + sell_offset]
            buy_idx = trade_day_idx.get(buy_date)
            sell_idx = trade_day_idx.get(sell_date)
            if buy_idx is None or sell_idx is None or (sell_idx - buy_idx) != hold_days:
                raw_label[i] = np.nan

    label_min = float(CONFIG["features"]["LABEL_WINSORIZE_MIN"])
    label_max = float(CONFIG["features"]["LABEL_WINSORIZE_MAX"])
    feats["label"] = np.clip(raw_label, label_min, label_max).astype(np.float32)

    return pd.DataFrame(feats, index=g.index)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "code",
        "name",
        "date",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "pctChg",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "label",
        "isST",
        "isTrading",
        "industry",
        "industryClassification",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_60",
        "pt_adjust_factor",
    }
    return [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]


def _build_research_prices(df: pd.DataFrame) -> pd.DataFrame:
    print("  构建点时复权研究价格...")
    frames = []
    for _, group in df.groupby("code", sort=False):
        group = group.sort_values("date").copy()
        adj = build_pt_adjusted_prices(group)
        group["open"] = adj["adj_open"].to_numpy(dtype=np.float32)
        group["high"] = adj["adj_high"].to_numpy(dtype=np.float32)
        group["low"] = adj["adj_low"].to_numpy(dtype=np.float32)
        group["close"] = adj["adj_close"].to_numpy(dtype=np.float32)
        group["pt_adjust_factor"] = adj["pt_adjust_factor"].to_numpy(dtype=np.float32)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    print("开始特征工程...")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = _build_research_prices(df)

    trade_days = pd.read_csv(TRADE_DAYS_TXT, header=None, names=["date"])
    trade_days["date"] = pd.to_datetime(trade_days["date"], errors="coerce")
    trade_day_idx = {d: i for i, d in enumerate(trade_days["date"].dropna().to_numpy())}

    print("  计算个股技术指标与标签...")
    feat_dfs = []
    for _, group in df.groupby("code", sort=False):
        feat_dfs.append(compute_features_for_stock(group, trade_day_idx=trade_day_idx))

    feat_all = pd.concat(feat_dfs)
    new_cols = [c for c in feat_all.columns if c not in df.columns]
    df = df.join(feat_all[new_cols], how="left")

    print("  计算市场截面特征...")
    daily = (
        df.groupby("date")
        .agg(
            mkt_ret_mean=("ret_1d", "mean"),
            mkt_ret_std=("ret_1d", "std"),
            mkt_advance_ratio=("ret_1d", lambda x: (x > 0).mean()),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["mkt_mom_5d"] = daily["mkt_ret_mean"].rolling(5, min_periods=1).sum()
    df = df.merge(daily, on="date", how="left")
    df["excess_ret_1d"] = df["ret_1d"] - df["mkt_ret_mean"]

    print("  计算行业截面特征...")
    industry_daily = (
        df.groupby(["date", "industry"])
        .agg(
            industry_ret_mean=("ret_1d", "mean"),
            industry_ret_std=("ret_1d", "std"),
            industry_advance_ratio=("ret_1d", lambda x: (x > 0).mean()),
            industry_count=("code", "count"),
        )
        .reset_index()
        .sort_values(["industry", "date"])
    )
    industry_daily["industry_mom_5d"] = (
        industry_daily.groupby("industry")["industry_ret_mean"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df = df.merge(industry_daily, on=["date", "industry"], how="left")
    df["excess_ret_industry"] = df["ret_1d"] - df["industry_ret_mean"]

    industry_class_daily = (
        df.groupby(["date", "industryClassification"])
        .agg(
            industry_class_ret_mean=("ret_1d", "mean"),
            industry_class_ret_std=("ret_1d", "std"),
            industry_class_advance_ratio=("ret_1d", lambda x: (x > 0).mean()),
            industry_class_count=("code", "count"),
        )
        .reset_index()
        .sort_values(["industryClassification", "date"])
    )
    industry_class_daily["industry_class_mom_5d"] = (
        industry_class_daily.groupby("industryClassification")["industry_class_ret_mean"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df = df.merge(industry_class_daily, on=["date", "industryClassification"], how="left")
    df["excess_ret_industry_class"] = df["ret_1d"] - df["industry_class_ret_mean"]

    print("  计算截面排名...")
    for col in ["peTTM", "pbMRQ", "psTTM"]:
        df[f"{col}_rank"] = df.groupby("date")[col].transform(lambda x: x.replace(0, np.nan).rank(pct=True))
    for period in [5, 10, 20]:
        df[f"ret_{period}d_rank"] = df.groupby("date")[f"ret_{period}d"].rank(pct=True)

    print("  清理预热期样本...")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["_rank"] = df.groupby("code").cumcount()
    df = df.loc[df["_rank"] >= 60].drop(columns=["_rank"]).reset_index(drop=True)

    feat_cols = get_feature_columns(df)
    float_cols = [col for col in feat_cols if df[col].dtype == np.float64]
    if float_cols:
        df[float_cols] = df[float_cols].astype(np.float32)
    gc.collect()

    print(f"\n✅ 特征工程完成! {df.shape[0]:,} 行, {df.shape[1]} 列")
    return df


def _features_config_hash() -> str:
    """特征配置哈希，参数变化时强制重算"""
    import hashlib, json
    feat_cfg = CONFIG["features"]
    snapshot = {k: feat_cfg.get(k) for k in ["HOLD_DAYS", "LABEL_WINSORIZE_MIN", "LABEL_WINSORIZE_MAX"]}
    raw = json.dumps(snapshot, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_hit(end_date=None) -> pd.DataFrame | None:
    if not (os.path.exists(FEATURE_PKL) and os.path.exists(CLEAN_PKL)):
        return None
    try:
        cached = pd.read_pickle(FEATURE_PKL)
        clean_df = pd.read_pickle(CLEAN_PKL)
    except Exception as exc:
        print(f"  [缓存读取失败: {exc}]，继续重算...")
        return None

    # 配置变更检测
    cached_hash = cached.attrs.get("config_hash", "")
    current_hash = _features_config_hash()
    if cached_hash != current_hash:
        print(f"  [配置变更] 特征参数已变化 (旧={cached_hash}, 新={current_hash})，强制重算...")
        return None

    if "pt_adjust_factor" not in cached.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(cached["date"]):
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(clean_df["date"]):
        clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")

    if end_date is not None:
        target = pd.Timestamp(end_date)
        if cached["date"].max() < target:
            return None
        return cached.loc[cached["date"] <= target].copy()

    if cached["date"].max() != clean_df["date"].max():
        return None
    return cached


def run_pipeline(end_date=None):
    cached = _cache_hit(end_date=end_date)
    if cached is not None:
        if end_date is not None:
            print(f"✅ [缓存命中] {os.path.basename(FEATURE_PKL)} 已覆盖 {end_date}，跳过重算")
        else:
            print(f"✅ [缓存命中] {os.path.basename(FEATURE_PKL)} 已是最新 ({cached['date'].max().date()})，跳过重算")
        return cached.sort_values(["date", "code"]).reset_index(drop=True)

    df = pd.read_pickle(CLEAN_PKL)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if end_date is not None:
        target = pd.Timestamp(end_date)
        df = df.loc[df["date"] <= target].copy()
        print(f"  [date filter] 数据截断至 {target.date()}")

    df = compute_all_features(df)

    os.makedirs(os.path.dirname(FEATURE_PKL), exist_ok=True)
    df.attrs["config_hash"] = _features_config_hash()
    df.to_pickle(FEATURE_PKL)
    print(f"保存至 {FEATURE_PKL}")
    print(f"特征列数: {len(get_feature_columns(df))}")
    return df


if __name__ == "__main__":
    import sys as _sys

    _end_date = None
    _args = _sys.argv[1:]
    if "--date" in _args:
        _idx = _args.index("--date")
        _end_date = _args[_idx + 1]
    run_pipeline(end_date=_end_date)
