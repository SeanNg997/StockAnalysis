"""数据加载与清洗 — 原始行情主表 + 市场执行快照"""

from __future__ import annotations

import gc
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

from config import CONFIG

warnings.filterwarnings("ignore")

DATA_DIR = CONFIG["paths"]["DATA_DIR"]
STOCK_LIST_CSV = CONFIG["paths"]["STOCK_LIST_CSV"]
TRADE_DAYS_TXT = CONFIG["paths"]["TRADE_DAYS_TXT"]
CLEAN_PKL = CONFIG["paths"]["CLEAN_PKL"]
MARKET_PKL = CONFIG["paths"]["MARKET_PKL"]
BACKTEST_MARKET_PKL = CONFIG["paths"].get("BACKTEST_MARKET_PKL")
DATA_META_JSON = CONFIG["paths"].get("DATA_META_JSON")

BACKTEST_START_YEAR = int(CONFIG["model"].get("BACKTEST_START_YEAR", 2023))
BACKTEST_HISTORY_BUFFER_DAYS = max(
    int(CONFIG["backtest"].get("MIN_PRICE_DAYS", 5)),
    30,
)

REQUIRED_RAW_COLS = {
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
    "isST",
    "isTrading",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
}


def load_raw_data() -> pd.DataFrame:
    """加载所有月度 CSV 并合并。"""
    print("[1/4] 加载月度 CSV 数据...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))
    if not files:
        raise FileNotFoundError(f"未找到月度 CSV 文件: {DATA_DIR}/Stock_dailyK_*.csv")

    dfs = [pd.read_csv(path, encoding="utf-8-sig") for path in files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    missing = REQUIRED_RAW_COLS.difference(df.columns)
    if missing:
        raise ValueError(
            "原始行情缺少字段: "
            f"{sorted(missing)}。当前本地 CSV 仍是旧版前复权数据，"
            "请先运行 `python src/py00_fetch_stock_data.py` 让程序自动切换到原始行情重建。"
        )

    print(f"  共读取 {len(files)} 个月度文件")
    print(f"  股票数: {df['code'].nunique()} 只")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    return df


def _cache_has_required_schema(df: pd.DataFrame, required_cols: set[str]) -> bool:
    return (df is not None) and (not df.empty) and required_cols.issubset(df.columns)


def _load_dataset_meta() -> dict:
    if not DATA_META_JSON or not os.path.exists(DATA_META_JSON):
        return {}
    try:
        with open(DATA_META_JSON, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _latest_csv_date() -> pd.Timestamp | None:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))
    if not files:
        return None
    latest = pd.read_csv(files[-1], encoding="utf-8-sig", usecols=["date"])
    if latest.empty:
        return None
    return pd.to_datetime(latest["date"].max())


def _expected_backtest_market_start_date(date_values) -> pd.Timestamp | None:
    unique_dates = pd.DatetimeIndex(np.sort(pd.to_datetime(pd.Series(date_values)).dropna().unique()))
    if len(unique_dates) == 0:
        return None

    target_start = pd.Timestamp(f"{BACKTEST_START_YEAR}-01-01")
    start_trade_idx = unique_dates.searchsorted(target_start, side="left")
    if start_trade_idx >= len(unique_dates):
        return None

    start_idx = max(0, start_trade_idx - BACKTEST_HISTORY_BUFFER_DAYS)
    return unique_dates[start_idx]


def _build_backtest_market_status(market_df: pd.DataFrame) -> pd.DataFrame:
    """按固定回测起始年份构建执行市场快照，并保留少量历史缓冲。"""
    if market_df.empty:
        return market_df.copy()

    expected_start = _expected_backtest_market_start_date(market_df["date"])
    if expected_start is None:
        raise ValueError(
            f"配置的 BACKTEST_START_YEAR={BACKTEST_START_YEAR} 超出数据范围，"
            f"当前最新交易日为 {pd.to_datetime(market_df['date']).max().date()}"
        )

    return market_df.loc[market_df["date"] >= expected_start].copy()


def build_market_status(raw_df: pd.DataFrame, end_date=None) -> pd.DataFrame:
    """生成执行层市场状态快照，保留原始价格口径。"""
    print("[2/4] 生成市场状态快照...")
    cols = ["code", "date", "isST", "isTrading", "open", "close", "preclose", "volume", "amount"]
    market_df = raw_df[cols].copy()
    market_df["date"] = pd.to_datetime(market_df["date"])

    if end_date is not None:
        market_df = market_df.loc[market_df["date"] <= pd.Timestamp(end_date)].copy()

    market_df = market_df.sort_values(["date", "code"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(MARKET_PKL), exist_ok=True)
    market_df.to_pickle(MARKET_PKL)

    if BACKTEST_MARKET_PKL:
        bt_market_df = _build_backtest_market_status(market_df)
        os.makedirs(os.path.dirname(BACKTEST_MARKET_PKL), exist_ok=True)
        bt_market_df.to_pickle(BACKTEST_MARKET_PKL)

    print(f"  市场快照: {len(market_df):,} 行")
    return market_df


def add_is_new_flag(df: pd.DataFrame) -> pd.DataFrame:
    """添加新股标记，并补充行业字段。"""
    print("[3/4] 添加 isNew 与行业字段...")
    stock_list_df = pd.read_csv(STOCK_LIST_CSV, encoding="utf-8-sig")
    stock_list_df["list_date"] = pd.to_datetime(stock_list_df["list_date"], errors="coerce")

    trade_days = pd.read_csv(TRADE_DAYS_TXT, header=None, names=["date"])
    trade_days["date"] = pd.to_datetime(trade_days["date"], errors="coerce")
    trade_dates = trade_days["date"].dropna().to_numpy()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.merge(
        stock_list_df[["code", "list_date", "industry", "industryClassification"]],
        on="code",
        how="left",
    )

    valid_mask = stock_list_df["list_date"].notna()
    indices = trade_dates.searchsorted(stock_list_df.loc[valid_mask, "list_date"].to_numpy(), side="left")
    cutoff_indices = indices + 29
    cutoff_dates = np.full(len(stock_list_df), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_rows = stock_list_df.index[valid_mask]
    in_range = cutoff_indices < len(trade_dates)
    cutoff_dates[valid_rows[in_range]] = trade_dates[cutoff_indices[in_range]]
    stock_list_df["cutoff_date"] = cutoff_dates

    df = df.merge(stock_list_df[["code", "cutoff_date"]], on="code", how="left")
    df["isNew"] = ((df["cutoff_date"].notna()) & (df["date"] <= df["cutoff_date"])).astype("int8")
    df = df.drop(columns=["list_date", "cutoff_date"])

    print(f"  isNew=1 行数: {int(df['isNew'].sum()):,}")
    return df


def _ffill_with_long_gap_zero(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """连续缺失 <=2 天前值填充，超过 2 天整段置 0。"""
    out = df.sort_values(["code", "date"]).copy()
    group = out.groupby("code", sort=False)

    for col in numeric_cols:
        original_missing = out[col].isna()
        filled = group[col].transform(lambda s: s.ffill())
        out[col] = filled

        prev_not_missing = (~original_missing).groupby(out["code"]).cumsum()
        miss_run_len = original_missing.groupby([out["code"], prev_not_missing]).transform("sum")
        long_gap_mask = original_missing & (miss_run_len > 2)
        out.loc[long_gap_mask, col] = 0.0

    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/4] 处理数值缺失...")
    core_numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turn",
        "pctChg",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
    ]
    tracked_cols = core_numeric_cols + ["preclose"]
    before_missing = int(df[tracked_cols].isna().sum().sum())
    df = _ffill_with_long_gap_zero(df, core_numeric_cols)

    # preclose 用于涨跌停参考价与公司行为识别，缺失时宁可保留为空/置0，
    # 不做前值填充，避免把除权日参考价错误传播到后续逻辑。
    df["preclose"] = pd.to_numeric(df["preclose"], errors="coerce").fillna(0.0)

    after_missing = int(df[tracked_cols].isna().sum().sum())
    print(f"  原始缺失值: {before_missing:,}")
    print(f"  处理后缺失值: {after_missing:,}")
    return df


def _cache_hit(end_date=None) -> pd.DataFrame | None:
    if not (os.path.exists(CLEAN_PKL) and os.path.exists(MARKET_PKL)):
        return None

    try:
        clean_df = pd.read_pickle(CLEAN_PKL)
        market_df = pd.read_pickle(MARKET_PKL)
    except Exception as exc:
        print(f"  [缓存读取失败: {exc}]，继续重建...")
        return None

    required_clean_cols = REQUIRED_RAW_COLS.union({"industry", "industryClassification", "isNew"})
    required_market_cols = {"code", "date", "isST", "isTrading", "open", "close", "preclose", "amount"}
    if not _cache_has_required_schema(clean_df, required_clean_cols):
        return None
    if not _cache_has_required_schema(market_df, required_market_cols):
        return None

    if not pd.api.types.is_datetime64_any_dtype(clean_df["date"]):
        clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(market_df["date"]):
        market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")

    meta = _load_dataset_meta()
    if meta.get("dataset_version") and meta.get("dataset_version") != "raw_price_pt_v2":
        return None

    if end_date is not None:
        target = pd.Timestamp(end_date)
        if clean_df["date"].max() < target or market_df["date"].max() < target:
            return None
        if BACKTEST_MARKET_PKL and not os.path.exists(BACKTEST_MARKET_PKL):
            return None
        return clean_df.loc[clean_df["date"] <= target].copy()

    latest_csv_date = _latest_csv_date()
    if latest_csv_date is None:
        return None
    if clean_df["date"].max() != latest_csv_date:
        return None
    if market_df["date"].max() != latest_csv_date:
        return None
    if BACKTEST_MARKET_PKL:
        if not os.path.exists(BACKTEST_MARKET_PKL):
            return None
        try:
            bt_market_df = pd.read_pickle(BACKTEST_MARKET_PKL)
            if not _cache_has_required_schema(bt_market_df, required_market_cols):
                return None
            if not pd.api.types.is_datetime64_any_dtype(bt_market_df["date"]):
                bt_market_df["date"] = pd.to_datetime(bt_market_df["date"], errors="coerce")
            if bt_market_df["date"].max() < latest_csv_date:
                return None
            expected_bt_start = _expected_backtest_market_start_date(market_df["date"])
            if expected_bt_start is None:
                return None
            if bt_market_df["date"].min() > expected_bt_start:
                return None
        except Exception:
            return None
    return clean_df


def run_pipeline(end_date=None) -> pd.DataFrame:
    cached = _cache_hit(end_date=end_date)
    if cached is not None:
        if end_date is not None:
            print(f"✅ [缓存命中] {os.path.basename(CLEAN_PKL)} 已覆盖 {end_date}，跳过重建")
        else:
            max_date = pd.to_datetime(cached["date"]).max().date()
            print(f"✅ [缓存命中] {os.path.basename(CLEAN_PKL)} 已是最新 ({max_date})，跳过重建")
        return cached.sort_values(["date", "code"]).reset_index(drop=True)

    raw_df = load_raw_data()
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    raw_df = raw_df.sort_values(["date", "code"]).reset_index(drop=True)

    build_market_status(raw_df, end_date=end_date)

    df = add_is_new_flag(raw_df)
    df = handle_missing(df)

    if end_date is not None:
        target = pd.Timestamp(end_date)
        df = df.loc[df["date"] <= target].copy()
        print(f"  [date filter] 数据截断至 {target.date()}")

    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(CLEAN_PKL), exist_ok=True)
    df.to_pickle(CLEAN_PKL)

    print(f"\n✅ 清洗完成，保存至 {CLEAN_PKL}")
    print(f"✅ 市场快照已保存至 {MARKET_PKL}")
    if BACKTEST_MARKET_PKL:
        print(f"✅ 回测市场快照已保存至 {BACKTEST_MARKET_PKL}")
    print(f"  最终数据: {df.shape[0]:,} 行, {df['code'].nunique()} 只股票")
    print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


if __name__ == "__main__":
    import sys as _sys

    _end_date = None
    _args = _sys.argv[1:]
    if "--date" in _args:
        _idx = _args.index("--date")
        _end_date = _args[_idx + 1]
    run_pipeline(end_date=_end_date)
