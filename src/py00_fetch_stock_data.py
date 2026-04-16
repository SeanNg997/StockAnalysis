"""A股日K线数据获取 — 全量下载 / 断点续传 / 增量更新"""

import baostock as bs
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import gc
import glob
import argparse
from config import CONFIG

# Path
BASE_DIR = CONFIG['paths']['BASE_DIR']
DATA_DIR = CONFIG['paths']['DATA_DIR']
STOCK_LIST_CSV = CONFIG['paths']['STOCK_LIST_CSV']
TRADE_DAYS_TXT = CONFIG['paths']['TRADE_DAYS_TXT']
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = CONFIG['data_fetch']['START_DATE']
ADJUST_FLAG = CONFIG['data_fetch']['ADJUST_FLAG']
SAVE_EVERY = CONFIG['data_fetch']['SAVE_EVERY']
MARKET_DATA_READY_HOUR = CONFIG['data_fetch']['MARKET_DATA_READY_HOUR']
END_DATE = datetime.now().strftime("%Y-%m-%d")


def get_expected_latest_date() -> str:
    """根据当前时间和交易日历，确定baostock中预期已入库的最新交易日"""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # 查近 60 个日历日的交易日历（覆盖节假日连休场景）
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
    else:
        prev_days = [d for d in trading_days if d < today_str]
        return prev_days[-1] if prev_days else trading_days[-1]


# ── 下载状态管理 ──

def load_completed():
    """从状态文件读取已完成的股票代码集合"""
    path = os.path.join(DATA_DIR, ".download_status.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_completed(completed: set):
    """持久化已完成的股票代码集合"""
    path = os.path.join(DATA_DIR, ".download_status.txt")
    with open(path, "w") as f:
        for code in sorted(completed):
            f.write(code + "\n")


# ── CSV 读写 ──

def load_existing_csv() -> pd.DataFrame:
    """加载所有月度CSV合并返回"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))
    df = pd.DataFrame()
    if not files:
        return df
    for f in files:
        temp_df = pd.read_csv(f, encoding="utf-8-sig")
        df = pd.concat([df, temp_df], ignore_index=True)
        del temp_df
        gc.collect()
    return df


def save_to_csv(df: pd.DataFrame):
    """按月拆分DataFrame写入各月度CSV（全量覆盖）"""
    if df.empty:
        return
    df = df.copy()
    df["_month"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m")
    for month, group in df.groupby("_month"):
        monthly_file = os.path.join(DATA_DIR, f"Stock_dailyK_{month}.csv")
        group.drop(columns=["_month"]).to_csv(monthly_file, index=False, encoding="utf-8-sig")


def save_incremental_months(new_df: pd.DataFrame):
    """增量更新：仅重写涉及的月度文件，按代码+日期去重(新数据优先)"""
    if new_df.empty:
        return
    new_df = new_df.copy()
    new_df["_month"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y%m")
    for month, group in new_df.groupby("_month"):
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


# ── 数据源 ──

def get_stock_list():
    """获取沪深主板A股股票列表（含退市股）"""
    print("正在获取沪深主板股票列表...")
    rs = bs.query_stock_basic()
    stock_list = []
    while rs.next():
        row = rs.get_row_data()
        if row[4] == "1":
            code = row[0]
            if code.startswith('sh.60') or code.startswith('sz.00'):
                isDelisted = (row[5] != "1")
                stock_list.append((code, row[1], row[2], row[3], isDelisted))
    stock_list_df = pd.DataFrame(stock_list, columns=["code", "name", "list_date", "delisted_date", "isDelisted"])
    stock_list_df.to_csv(STOCK_LIST_CSV, index=False, encoding="utf-8-sig")
    print(f"共获取 {len(stock_list)} 只主板股票（含退市股）")
    return stock_list

def get_trade_days_list(start_date: str = START_DATE, end_date: str = END_DATE) -> list:
    """获取指定范围内的交易日列表"""
    # 往前推1年确保覆盖
    if start_date == START_DATE:
        start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    print(f"正在获取 {start_date} 至 {end_date} 交易日列表...")
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    trade_days = []
    while rs.next():
        row = rs.get_row_data()
        if row[1] == "1":
            trade_days.append(row[0])

    with open(TRADE_DAYS_TXT, "w") as f:
        for day in trade_days:
            f.write(day + "\n") 

    print(f"共获取 {len(trade_days)} 个交易日交易日")
    return trade_days

def _to_dataframe(rows, fields, symbol, name):
    """将baostock返回数据转为DataFrame"""
    df = pd.DataFrame(rows, columns=fields)
    df.insert(0, "name", name)
    df.insert(0, "code", symbol)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turn",
                    "pctChg", "isST", "tradestatus", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if "tradestatus" in df.columns:
        df["isTrading"] = (df["tradestatus"] == 1).astype(int)
        df = df.drop(columns=["tradestatus"])
    return df

def fetch_dailyK(symbol: str, name: str, start_date: str = START_DATE) -> pd.DataFrame | None:
    """获取单只股票日K线数据"""
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,open,high,low,close,volume,amount,turn,pctChg,isST,tradestatus,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=start_date,
        end_date=END_DATE,
        frequency="d",
        adjustflag=ADJUST_FLAG
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if rows:
        return _to_dataframe(rows, rs.fields, symbol, name)
    return None


# ── 核心逻辑 ──

def is_complete(existing_df: pd.DataFrame, code: str, expected_date: str, is_delisted: bool = False) -> bool:
    """判断某只股票是否已完成下载（退市股有数据即视为完成）"""
    stock_data = existing_df[existing_df["code"] == code]
    if stock_data.empty:
        return False
    if is_delisted:
        return True
    last_date = stock_data["date"].max()
    return last_date >= expected_date


def _update_max_date(current: str | None, batch: pd.DataFrame) -> str:
    """返回 current 与 batch 中最大日期的较大值"""
    batch_max = batch["date"].max()
    return batch_max if current is None else max(current, batch_max)


def main(full: bool = False):
    lg = bs.login()
    if lg.error_code != "0":
        print(f"BaoStock 登录失败: {lg.error_msg}")
        return

    try:
        # 确定预期最新数据日期
        expected_latest_date = get_expected_latest_date()
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
              f"数据库最新日期: {expected_latest_date}")
              
        # 获取股票列表和交易日列表
        stock_list = get_stock_list()
        trade_days = get_trade_days_list()

        existing_df = load_existing_csv()
        if existing_df.empty:
            full = True
            print("未检测到已存在数据，将切换为全量下载模式")

        # 增量更新模式
        if not full:
            print("=" * 50)
            print("增量更新模式")
            print("=" * 50)
            print("正在分析已有数据...")
            last_date_map = {}

            if not existing_df.empty:
                last_date_map = existing_df.groupby("code")["date"].max().to_dict()
            existing_df = None  # 释放内存

            pending_update = []
            pending_full = []
            skip_delisted_no_data = 0
            for code, name, list_date, delisted_date, is_delisted in stock_list:
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

            if not pending_update and not pending_full:
                print("✅ 所有股票已是最新状态，无需更新。")
                return

            print(f"需要更新股票 {len(pending_update)} 只, 新增股票 {len(pending_full)} 只"
                  f"{f'，跳过退市无数据股票 {skip_delisted_no_data} 只' if skip_delisted_no_data else ''}")

            update_count = 0
            max_new_date = None
            new_chunks = []
            for idx, (code, name, last_date) in enumerate(tqdm(pending_update, desc="增量更新中", ncols=80)):
                from_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                new_df = fetch_dailyK(code, name, from_date)
                if new_df is not None and not new_df.empty:
                    new_chunks.append(new_df)
                    update_count += 1

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
            new_chunks = []
            if pending_full:
                for idx, (code, name) in enumerate(tqdm(pending_full, desc="新增股票中", ncols=80)):
                    new_df = fetch_dailyK(code, name)
                    if new_df is not None and not new_df.empty:
                        new_chunks.append(new_df)
                        new_count += 1

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
                with open(os.path.join(DATA_DIR, ".last_date.txt"), "w") as f:
                    f.write(str(max_new_date))

            print(f"\n✅ 增量更新完成! 成功更新股票 {update_count} 只, 新增股票 {new_count} 只")
            return

        # ── 全量下载 / 断点续传 ──
        print("=" * 50)
        print("全量下载模式")
        print("=" * 50)
        completed = load_completed()
        if not existing_df.empty:
            print(f"已有数据: {len(existing_df):,} 条, 已完成 {len(completed)} 只股票")

        skip_count = 0
        pending = []
        for code, name, _, _, is_delisted in stock_list:
            if not existing_df.empty and is_complete(existing_df, code, expected_latest_date, is_delisted):
                skip_count += 1
            else:
                pending.append((code, name, is_delisted))

        if skip_count > 0:
            print(f"跳过已完成: {skip_count} 只, 待下载/补全: {len(pending)} 只")

        if not pending:
            print("✅ 所有股票已是最新状态，无需更新。")
            return

        success_count = skip_count
        fail_count = 0

        for idx, (code, name, _) in enumerate(tqdm(pending, desc="正在下载日K数据", ncols=80)):
            df = fetch_dailyK(code, name)
            if df is not None and not df.empty:
                if not existing_df.empty and code in existing_df["code"].values:
                    existing_df = existing_df[existing_df["code"] != code]
                existing_df = pd.concat([existing_df, df], ignore_index=True)
                completed.add(code)
                success_count += 1
            else:
                fail_count += 1

            if (idx + 1) % SAVE_EVERY == 0:
                save_to_csv(existing_df)
                save_completed(completed)
                print(f"  已自动保存 (进度 {skip_count + idx + 1}/{len(stock_list)})")

        # 最终保存所有数据
        save_to_csv(existing_df)
        save_completed(completed)

        if not existing_df.empty:
            latest = existing_df["date"].max()
            with open(os.path.join(DATA_DIR, ".last_date.txt"), "w") as f:
                f.write(str(latest))

        files = sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))
        total_stocks = len(existing_df["code"].unique())
        print("✅ 全量下载完成!")
        print(f"数据已保存至 data/ 目录，共 {len(files)} 个文件")
        print(f"总记录数: {len(existing_df):,} 条")
        print(f"总股票数: {total_stocks} 只")
        print(f"    成功: {success_count} 只")
        print(f"    失败: {fail_count} 只")
        print(f"    跳过: {skip_count} 只")

    finally:
        bs.logout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股日K数据下载")
    parser.add_argument("-f", "--full", action="store_true", help="全量下载模式，重新下载所有股票数据")
    args = parser.parse_args()
    main(full=args.full)
