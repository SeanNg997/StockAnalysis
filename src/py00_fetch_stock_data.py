"""A股日K线数据获取 — 原始行情 + 复权因子 + 分红事件"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import baostock as bs
import pandas as pd
from tqdm import tqdm

from config import CONFIG


BASE_DIR = CONFIG["paths"]["BASE_DIR"]
DATA_DIR = CONFIG["paths"]["DATA_DIR"]
OUTPUT_DIR = CONFIG["paths"]["OUTPUT_DIR"]
STOCK_LIST_CSV = CONFIG["paths"]["STOCK_LIST_CSV"]
TRADE_DAYS_TXT = CONFIG["paths"]["TRADE_DAYS_TXT"]
ADJUST_FACTOR_PKL = CONFIG["paths"]["ADJUST_FACTOR_PKL"]
DIVIDEND_PKL = CONFIG["paths"]["DIVIDEND_PKL"]
DATA_META_JSON = CONFIG["paths"]["DATA_META_JSON"]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tmp"), exist_ok=True)

START_DATE = CONFIG["data_fetch"]["START_DATE"]
ADJUST_FLAG = CONFIG["data_fetch"]["ADJUST_FLAG"]
SAVE_EVERY = CONFIG["data_fetch"]["SAVE_EVERY"]
MARKET_DATA_READY_HOUR = CONFIG["data_fetch"]["MARKET_DATA_READY_HOUR"]
END_DATE = datetime.now().strftime("%Y-%m-%d")

DATASET_VERSION = "raw_price_pt_v2"
KLINE_FIELDS = (
    "date,open,high,low,close,preclose,volume,amount,turn,pctChg,"
    "isST,tradestatus,peTTM,pbMRQ,psTTM,pcfNcfTTM"
)
DIVIDEND_CACHE_REQUIRED_COLS = {
    "code",
    "dividOperateDate",
    "dividPlanDate",
    "dividCashPsBeforeTax",
    "dividStocksPs",
    "dividReserveToStockPs",
}
ADJUST_FACTOR_CACHE_REQUIRED_COLS = {
    "code",
    "dividOperateDate",
}


def get_expected_latest_date() -> str:
    """根据当前时间和交易日历，确定 baostock 中预期已入库的最新交易日。"""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    start_str = (now - timedelta(days=60)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start_str, end_date=today_str)
    trading_days = []
    while rs.next():
        row = rs.get_row_data()
        if row[1] == "1":
            trading_days.append(row[0])

    if not trading_days:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    if trading_days[-1] == today_str and now.hour >= MARKET_DATA_READY_HOUR:
        return today_str
    prev_days = [d for d in trading_days if d < today_str]
    return prev_days[-1] if prev_days else trading_days[-1]


def load_completed() -> set[str]:
    path = os.path.join(DATA_DIR, ".download_status.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
            return set(line.strip() for line in fp if line.strip())
    return set()


def save_completed(completed: set[str]) -> None:
    path = os.path.join(DATA_DIR, ".download_status.txt")
    with open(path, "w", encoding="utf-8") as fp:
        for code in sorted(completed):
            fp.write(code + "\n")


def load_dataset_meta() -> dict:
    if not os.path.exists(DATA_META_JSON):
        return {}
    try:
        with open(DATA_META_JSON, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def save_dataset_meta(expected_latest_date: str) -> None:
    payload = {
        "dataset_version": DATASET_VERSION,
        "adjustflag": ADJUST_FLAG,
        "price_mode": "raw",
        "expected_latest_date": expected_latest_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(DATA_META_JSON, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _is_raw_dataset(existing_df: pd.DataFrame) -> bool:
    required_cols = {"preclose", "open", "high", "low", "close", "pctChg"}
    meta = load_dataset_meta()
    return required_cols.issubset(existing_df.columns) and meta.get("dataset_version") == DATASET_VERSION


def load_existing_csv() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    return df


def save_to_csv(df: pd.DataFrame) -> None:
    if df.empty:
        return
    temp = df.copy()
    temp["_month"] = pd.to_datetime(temp["date"]).dt.strftime("%Y%m")
    for month, group in temp.groupby("_month"):
        monthly_file = os.path.join(DATA_DIR, f"Stock_dailyK_{month}.csv")
        group.drop(columns=["_month"]).to_csv(monthly_file, index=False, encoding="utf-8-sig")


def save_incremental_months(new_df: pd.DataFrame) -> None:
    if new_df.empty:
        return
    temp = new_df.copy()
    temp["_month"] = pd.to_datetime(temp["date"]).dt.strftime("%Y%m")
    for month, group in temp.groupby("_month"):
        group = group.drop(columns=["_month"])
        monthly_file = os.path.join(DATA_DIR, f"Stock_dailyK_{month}.csv")
        if os.path.exists(monthly_file):
            existing_month = pd.read_csv(monthly_file, encoding="utf-8-sig")
            combined = pd.concat([existing_month, group], ignore_index=True)
            combined = combined.drop_duplicates(subset=["code", "date"], keep="last")
            combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
            combined.to_csv(monthly_file, index=False, encoding="utf-8-sig")
        else:
            group.sort_values(["code", "date"]).reset_index(drop=True).to_csv(
                monthly_file, index=False, encoding="utf-8-sig"
            )


def load_action_cache(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_pickle(path)
    except Exception:
        return pd.DataFrame()


def save_action_cache(df: pd.DataFrame, path: str, subset: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if df.empty:
        df.to_pickle(path)
        return
    df = df.drop_duplicates(subset=subset, keep="last").sort_values(subset).reset_index(drop=True)
    df.to_pickle(path)


def has_valid_action_cache(path: str, required_cols: set[str]) -> bool:
    if not os.path.exists(path):
        return False
    df = load_action_cache(path)
    return (df is not None) and (not df.empty) and required_cols.issubset(df.columns)


def get_stock_list():
    print("正在获取沪深主板股票列表...")
    rs = bs.query_stock_basic()
    stock_list = []
    while rs.next():
        row = rs.get_row_data()
        if row[4] == "1":
            code = row[0]
            if code.startswith("sh.60") or code.startswith("sz.00"):
                is_delisted = row[5] != "1"
                stock_list.append((code, row[1], row[2], row[3], is_delisted))

    print("正在获取股票行业信息...")
    industry_list = []
    rs_industry = bs.query_stock_industry()
    while rs_industry.next():
        industry_list.append(rs_industry.get_row_data())

    stock_list_df = pd.DataFrame(
        stock_list,
        columns=["code", "name", "list_date", "delisted_date", "isDelisted"],
    )
    industry_df = pd.DataFrame(
        industry_list,
        columns=["updateDate", "code", "code_name", "industry", "industryClassification"],
    )
    stock_list_df = stock_list_df.merge(
        industry_df[["code", "industry", "industryClassification"]],
        on="code",
        how="left",
    )
    stock_list_df.to_csv(STOCK_LIST_CSV, index=False, encoding="utf-8-sig")
    print(f"共获取 {len(stock_list)} 只主板股票（含退市股）")
    return stock_list


def get_trade_days_list(start_date: str = START_DATE, end_date: str = END_DATE) -> list[str]:
    if start_date == START_DATE:
        start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    print(f"正在获取 {start_date} 至 {end_date} 交易日列表...")
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    trade_days = []
    while rs.next():
        row = rs.get_row_data()
        if row[1] == "1":
            trade_days.append(row[0])

    with open(TRADE_DAYS_TXT, "w", encoding="utf-8") as fp:
        for day in trade_days:
            fp.write(day + "\n")

    print(f"共获取 {len(trade_days)} 个交易日交易日")
    return trade_days


def _to_dataframe(rows, fields, symbol, name):
    df = pd.DataFrame(rows, columns=fields)
    df.insert(0, "name", name)
    df.insert(0, "code", symbol)
    numeric_cols = [
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
        "tradestatus",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if "tradestatus" in df.columns:
        df["isTrading"] = (df["tradestatus"] == 1).astype(int)
        df = df.drop(columns=["tradestatus"])
    return df


def fetch_daily_k(symbol: str, name: str, start_date: str = START_DATE, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    query_end_date = end_date or END_DATE
    rs = bs.query_history_k_data_plus(
        symbol,
        KLINE_FIELDS,
        start_date=start_date,
        end_date=query_end_date,
        frequency="d",
        adjustflag=ADJUST_FLAG,
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if rows:
        return _to_dataframe(rows, rs.fields, symbol, name)
    return None


def fetch_adjust_factors(code: str, start_date: str = START_DATE, end_date: Optional[str] = None) -> pd.DataFrame:
    query_end_date = end_date or END_DATE
    rs = bs.query_adjust_factor(code, start_date=start_date, end_date=query_end_date)
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=["code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor", "adjustFactor"])
    df = pd.DataFrame(rows, columns=rs.fields)
    for col in ["foreAdjustFactor", "backAdjustFactor", "adjustFactor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dividOperateDate"] = pd.to_datetime(df["dividOperateDate"], errors="coerce")
    return df


def fetch_dividend_events(code: str, start_year: int, end_year: int) -> pd.DataFrame:
    parts = []
    for year in range(start_year, end_year + 1):
        rs = bs.query_dividend_data(code, year=str(year), yearType="operate")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            continue
        part = pd.DataFrame(rows, columns=rs.fields)
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    date_cols = [
        "dividPreNoticeDate",
        "dividAgmPumDate",
        "dividPlanAnnounceDate",
        "dividPlanDate",
        "dividRegistDate",
        "dividOperateDate",
        "dividPayDate",
        "dividStockMarketDate",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["dividCashPsBeforeTax", "dividCashPsAfterTax", "dividStocksPs", "dividReserveToStockPs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def is_complete(existing_df: pd.DataFrame, code: str, expected_date: str, is_delisted: bool = False) -> bool:
    stock_data = existing_df[existing_df["code"] == code]
    if stock_data.empty:
        return False
    if is_delisted:
        return True
    last_date = stock_data["date"].max()
    return last_date >= expected_date


def _update_max_date(current: Optional[str], batch: pd.DataFrame) -> str:
    batch_max = batch["date"].max()
    return batch_max if current is None else max(current, batch_max)


def _refresh_action_tables(
    codes: list[str],
    expected_latest_date: str,
    refresh_adjust_factors: bool = False,
    refresh_dividends: bool = True,
) -> None:
    if not codes or (not refresh_adjust_factors and not refresh_dividends):
        return

    print("=" * 50)
    print("刷新复权因子与分红事件")
    print("=" * 50)

    existing_factors = load_action_cache(ADJUST_FACTOR_PKL) if refresh_adjust_factors else pd.DataFrame()
    existing_dividends = load_action_cache(DIVIDEND_PKL) if refresh_dividends else pd.DataFrame()
    start_year = datetime.strptime(START_DATE, "%Y-%m-%d").year
    end_year = datetime.strptime(expected_latest_date, "%Y-%m-%d").year

    factor_chunks = []
    dividend_chunks = []
    for idx, code in enumerate(tqdm(codes, desc="公司行为数据", ncols=80), start=1):
        if refresh_adjust_factors:
            factor_df = fetch_adjust_factors(code, start_date=START_DATE, end_date=expected_latest_date)
            if not factor_df.empty:
                factor_chunks.append(factor_df)

        if refresh_dividends:
            dividend_df = fetch_dividend_events(code, start_year=start_year, end_year=end_year)
            if not dividend_df.empty:
                dividend_chunks.append(dividend_df)

        if idx % SAVE_EVERY == 0:
            if refresh_adjust_factors and factor_chunks:
                existing_factors = pd.concat([existing_factors, *factor_chunks], ignore_index=True)
                factor_chunks = []
                save_action_cache(existing_factors, ADJUST_FACTOR_PKL, ["code", "dividOperateDate"])
            if refresh_dividends and dividend_chunks:
                existing_dividends = pd.concat([existing_dividends, *dividend_chunks], ignore_index=True)
                dividend_chunks = []
                save_action_cache(existing_dividends, DIVIDEND_PKL, ["code", "dividOperateDate", "dividPlanDate"])

    if refresh_adjust_factors and factor_chunks:
        existing_factors = pd.concat([existing_factors, *factor_chunks], ignore_index=True)
    if refresh_dividends and dividend_chunks:
        existing_dividends = pd.concat([existing_dividends, *dividend_chunks], ignore_index=True)

    if refresh_adjust_factors:
        save_action_cache(existing_factors, ADJUST_FACTOR_PKL, ["code", "dividOperateDate"])
        print(f"✅ 复权因子已保存至 {ADJUST_FACTOR_PKL}")
    if refresh_dividends:
        save_action_cache(existing_dividends, DIVIDEND_PKL, ["code", "dividOperateDate", "dividPlanDate"])
        print(f"✅ 分红事件已保存至 {DIVIDEND_PKL}")


def _print_code_sample(title: str, codes: list[str], limit: int = 10) -> None:
    if not codes:
        return
    sample = ", ".join(codes[:limit])
    suffix = " ..." if len(codes) > limit else ""
    print(f"{title}: {sample}{suffix}")


def main(full: bool = False):
    lg = bs.login()
    if lg.error_code != "0":
        print(f"BaoStock 登录失败: {lg.error_msg}")
        return

    try:
        expected_latest_date = get_expected_latest_date()
        print(
            f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"数据库最新日期: {expected_latest_date}\n"
            f"价格口径: 不复权(raw)"
        )

        stock_list = get_stock_list()
        get_trade_days_list(end_date=expected_latest_date)

        existing_df = load_existing_csv()
        dataset_ready = not existing_df.empty and _is_raw_dataset(existing_df)
        if existing_df.empty:
            full = True
            print("未检测到已存在数据，将切换为全量下载模式")
        elif not dataset_ready:
            print("检测到旧版前复权/缺少 preclose 的数据集，自动切换为全量重建原始行情")
            existing_df = pd.DataFrame()
            full = True

        codes_for_actions: list[str] = []

        if not full:
            print("=" * 50)
            print("增量更新模式")
            print("=" * 50)
            print("正在分析已有数据...")
            last_date_map = existing_df.groupby("code")["date"].max().to_dict() if not existing_df.empty else {}
            existing_df = None

            pending_update = []
            pending_full = []
            skip_delisted_no_data = 0
            for code, name, _, _, is_delisted in stock_list:
                last_date = last_date_map.get(code)
                if last_date is None:
                    if is_delisted:
                        skip_delisted_no_data += 1
                    else:
                        pending_full.append((code, name))
                elif is_delisted:
                    pass
                elif last_date < expected_latest_date:
                    pending_update.append((code, name, last_date))

            has_valid_dividend_cache = has_valid_action_cache(DIVIDEND_PKL, DIVIDEND_CACHE_REQUIRED_COLS)
            has_valid_adjust_factor_cache = has_valid_action_cache(
                ADJUST_FACTOR_PKL,
                ADJUST_FACTOR_CACHE_REQUIRED_COLS,
            )
            needs_dividend_bootstrap = not has_valid_dividend_cache
            if needs_dividend_bootstrap:
                print("未检测到可用的分红事件缓存，将补抓 dividend_events.pkl")
            elif not has_valid_adjust_factor_cache:
                print("未检测到可用的复权因子缓存，但当前主流程不依赖它，跳过 adjust_factors.pkl 全量补抓")
            if not pending_update and not pending_full and not needs_dividend_bootstrap:
                print("✅ 所有股票已是最新状态，无需更新。")
                save_dataset_meta(expected_latest_date)
                return

            print(
                f"需要更新股票 {len(pending_update)} 只, 新增股票 {len(pending_full)} 只"
                f"{f'，跳过退市无数据股票 {skip_delisted_no_data} 只' if skip_delisted_no_data else ''}"
            )

            update_count = 0
            update_fail_codes: list[str] = []
            max_new_date = None
            new_chunks = []
            for idx, (code, name, last_date) in enumerate(tqdm(pending_update, desc="增量更新中", ncols=80)):
                from_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                new_df = fetch_daily_k(code, name, from_date, end_date=expected_latest_date)
                if new_df is not None and not new_df.empty:
                    new_chunks.append(new_df)
                    update_count += 1
                    codes_for_actions.append(code)
                else:
                    update_fail_codes.append(code)

                if (idx + 1) % SAVE_EVERY == 0 and new_chunks:
                    batch = pd.concat(new_chunks, ignore_index=True)
                    save_incremental_months(batch)
                    max_new_date = _update_max_date(max_new_date, batch)
                    new_chunks = []
                    print(f"  已自动保存 (更新 {idx + 1}/{len(pending_update)})")

            if new_chunks:
                batch = pd.concat(new_chunks, ignore_index=True)
                save_incremental_months(batch)
                max_new_date = _update_max_date(max_new_date, batch)

            new_count = 0
            new_fail_codes: list[str] = []
            new_chunks = []
            if pending_full:
                for idx, (code, name) in enumerate(tqdm(pending_full, desc="新增股票中", ncols=80)):
                    new_df = fetch_daily_k(code, name, end_date=expected_latest_date)
                    if new_df is not None and not new_df.empty:
                        new_chunks.append(new_df)
                        new_count += 1
                        codes_for_actions.append(code)
                    else:
                        new_fail_codes.append(code)

                    if (idx + 1) % SAVE_EVERY == 0 and new_chunks:
                        batch = pd.concat(new_chunks, ignore_index=True)
                        save_incremental_months(batch)
                        max_new_date = _update_max_date(max_new_date, batch)
                        new_chunks = []
                        print(f"  已自动保存 (新增 {idx + 1}/{len(pending_full)})")

                if new_chunks:
                    batch = pd.concat(new_chunks, ignore_index=True)
                    save_incremental_months(batch)
                    max_new_date = _update_max_date(max_new_date, batch)

            if max_new_date:
                with open(os.path.join(DATA_DIR, ".last_date.txt"), "w", encoding="utf-8") as fp:
                    fp.write(str(max_new_date))

            if needs_dividend_bootstrap:
                codes_for_actions = [code for code, _, _, _, is_delisted in stock_list if not is_delisted]

            _refresh_action_tables(
                sorted(set(codes_for_actions)),
                expected_latest_date,
                refresh_adjust_factors=has_valid_adjust_factor_cache,
                refresh_dividends=True,
            )
            save_dataset_meta(expected_latest_date)
            print(f"\n✅ 增量更新完成! 成功更新股票 {update_count} 只, 新增股票 {new_count} 只")
            if skip_delisted_no_data:
                print(f"    退市无数据(已跳过): {skip_delisted_no_data} 只")
            if update_fail_codes or new_fail_codes:
                total_fail = len(update_fail_codes) + len(new_fail_codes)
                print(f"    真正下载失败: {total_fail} 只")
                _print_code_sample("    更新失败样本", update_fail_codes)
                _print_code_sample("    新增失败样本", new_fail_codes)
            return

        print("=" * 50)
        print("全量下载模式")
        print("=" * 50)
        completed = set()
        existing_df = pd.DataFrame()
        success_count = 0
        skipped_delisted_no_data = 0
        fail_count = 0
        fail_codes: list[str] = []
        pending = [(code, name, is_delisted) for code, name, _, _, is_delisted in stock_list]

        for idx, (code, name, is_delisted) in enumerate(tqdm(pending, desc="正在下载原始日K数据", ncols=80)):
            if is_delisted and code in completed:
                continue
            df = fetch_daily_k(code, name, end_date=expected_latest_date)
            if df is not None and not df.empty:
                existing_df = pd.concat([existing_df, df], ignore_index=True)
                completed.add(code)
                success_count += 1
            else:
                if is_delisted:
                    skipped_delisted_no_data += 1
                else:
                    fail_count += 1
                    fail_codes.append(code)

            if (idx + 1) % SAVE_EVERY == 0:
                save_to_csv(existing_df)
                save_completed(completed)
                print(f"  已自动保存 (进度 {idx + 1}/{len(pending)})")

        save_to_csv(existing_df)
        save_completed(completed)

        if not existing_df.empty:
            latest = existing_df["date"].max()
            with open(os.path.join(DATA_DIR, ".last_date.txt"), "w", encoding="utf-8") as fp:
                fp.write(str(latest))

        active_codes = [code for code, _, _, _, is_delisted in stock_list if not is_delisted]
        _refresh_action_tables(
            active_codes,
            expected_latest_date,
            refresh_adjust_factors=True,
            refresh_dividends=True,
        )
        save_dataset_meta(expected_latest_date)

        files = sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))
        total_stocks = len(existing_df["code"].unique())
        print("✅ 全量下载完成!")
        print(f"数据已保存至 data/ 目录，共 {len(files)} 个文件")
        print(f"总记录数: {len(existing_df):,} 条")
        print(f"总股票数: {total_stocks} 只")
        print(f"    成功: {success_count} 只")
        if skipped_delisted_no_data:
            print(f"    退市无数据(已跳过): {skipped_delisted_no_data} 只")
        print(f"    真正下载失败: {fail_count} 只")
        _print_code_sample("    失败股票样本", fail_codes)
    finally:
        bs.logout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股原始日K数据下载")
    parser.add_argument("-f", "--full", action="store_true", help="全量下载模式，重新下载所有股票数据")
    args = parser.parse_args()
    main(full=args.full)
