"""
爬取A股近10年日K线数据
使用 baostock 免费数据源，数据保存至 data/ 目录

功能:
  - 全量下载 / 断点续传 / 增量更新
  - 数据按月拆分存储，文件名格式: Stock_dailyK_YYYYMM.csv
  - 每n只股票自动保存，防止中断丢失
"""

import baostock as bs
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import gc
import time
import glob
from config import CONFIG

# Path
BASE_DIR = CONFIG['paths']['BASE_DIR']
DATA_DIR = CONFIG['paths']['DATA_DIR']
os.makedirs(DATA_DIR, exist_ok=True)

# Date
START_DATE = CONFIG['data_fetch']['START_DATE']
def get_end_date():
    """获取今天日期，用于 baostock API 的 end_date 参数"""
    return datetime.now().strftime("%Y-%m-%d")
END_DATE = get_end_date()

# Other parameters
ADJUST_FLAG = CONFIG['data_fetch']['ADJUST_FLAG']  # 调整标志位（1：后复权；2：前复权；3：不复权）
SAVE_EVERY = CONFIG['data_fetch']['SAVE_EVERY']  # 每下载指定数量只股票保存一次
MARKET_DATA_READY_HOUR = CONFIG['data_fetch']['MARKET_DATA_READY_HOUR']  # baostock数据更新时间


def get_expected_latest_date() -> str:
    """
    根据当前时间和 baostock 交易日历，确定当前 baostock 中
    预期已入库的最新交易日期。

    规则：
    - 当前时间 >= 18:00 且今天是交易日 → 预期最新 = 今天
    - 当前时间 < 18:00，或今天是非交易日（周末/节假日） → 预期最新 = 上一个交易日

    注意：调用前须已登录 baostock（bs.login()）。
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # 查近 60 个日历日的交易日历（覆盖节假日连休场景）
    start_str = (now - timedelta(days=60)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start_str, end_date=today_str)
    trading_days = []
    while rs.next():
        row = rs.get_row_data()
        if row[1] == "1":
            trading_days.append(row[0])  # 'YYYY-MM-DD'

    if not trading_days:
        # 降级：若接口异常则回退到昨天
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    if trading_days[-1] == today_str and now.hour >= MARKET_DATA_READY_HOUR:
        # 今天是交易日且已过数据就绪时间
        return today_str
    else:
        # 盘前 / 非交易日：预期最新数据为今天之前最近一个交易日
        prev_days = [d for d in trading_days if d < today_str]
        return prev_days[-1] if prev_days else trading_days[-1]

# ── 月度文件管理 ──────────────────────────────────────────────

def get_monthly_file(yyyymm: str) -> str:
    """返回指定年月的CSV文件路径，yyyymm 格式如 '202004'"""
    return os.path.join(DATA_DIR, f"Stock_dailyK_{yyyymm}.csv")


def list_monthly_files() -> list:
    """返回所有月度CSV文件路径（按时间排序）"""
    return sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))


# ── 下载状态管理 ──────────────────────────────────────────────
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


# ── CSV 读写 ──────────────────────────────────────────────

def load_existing_csv() -> pd.DataFrame:
    """加载所有月度CSV数据合并返回，无文件则返回空DataFrame"""
    files = list_monthly_files()
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
    """按月拆分DataFrame，写入各月度CSV文件（全量覆盖）"""
    if df.empty:
        return
    df = df.copy()
    df["_month"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m")
    for month, group in df.groupby("_month"):
        monthly_file = get_monthly_file(month)
        group.drop(columns=["_month"]).to_csv(monthly_file, index=False, encoding="utf-8-sig")


def save_incremental_months(new_df: pd.DataFrame):
    """
    增量更新专用：仅重写 new_df 中涉及的月度文件。
    对每个月：先剔除该月文件中与新数据重复的行（按代码+日期去重），再追加新行。
    保留该月文件中已有的其他日期数据，避免丢失。

    注意：去重时 keep="last"，即新数据优先于旧数据（通常新数据更完整）。
    """
    if new_df.empty:
        return
    new_df = new_df.copy()
    new_df["_month"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y%m")
    for month, group in new_df.groupby("_month"):
        group = group.drop(columns=["_month"])
        monthly_file = get_monthly_file(month)
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


# ── 数据源 ──────────────────────────────────────────────

def get_stock_list():
    """获取沪深主板A股股票列表（包含退市股）"""
    print("正在获取沪深主板股票列表...")
    rs = bs.query_stock_basic()
    stock_list = []
    while rs.next():
        row = rs.get_row_data()
        # row[0-5]分别是证券代码、证券na me、上市日期、退市日期、证券类型、上市状态
        if row[4] == "1":
            code = row[0]
            if code.startswith('sh.60') or code.startswith('sz.00'):
                isDelisted = (row[5] != "1")  # row[5] != "1" 表示已退市
                stock_list.append((code, row[1], isDelisted))
    stock_list_df = pd.DataFrame(stock_list, columns=["code", "name", "isDelisted"])
    stock_list_df.to_csv(os.path.join(DATA_DIR, "stock_list.csv"), index=False, encoding="utf-8-sig")
    print(f"共获取 {len(stock_list)} 只主板股票（含退市股）")
    return stock_list


def _to_dataframe(rows, fields, symbol, name):
    """将baostock返回数据转为DataFrame"""
    df = pd.DataFrame(rows, columns=fields)
    df.insert(0, "name", name)
    df.insert(0, "code", symbol)
    # 批量转换数值列
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turn",
                    "pctChg", "isST", "tradestatus", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # 新增 isTrading 列，基于 tradestatus 字段
    if "tradestatus" in df.columns:
        df["isTrading"] = df["tradestatus"].apply(lambda x: 1 if x == 1 else 0)
        df = df.drop(columns=["tradestatus"])
    return df

def fetch_dailyK(symbol: str, name: str, start_date: str = START_DATE) -> pd.DataFrame | None:
    """获取单只股票日K线数据
    
    Args:
        symbol: 股票代码
        name: 股票名称  
        start_date: 起始日期，默认为全局 START_DATE
    """
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


# ── 核心逻辑 ──────────────────────────────────────────────

def is_complete(existing_df: pd.DataFrame, code: str, expected_date: str, is_delisted: bool = False) -> bool:
    """
    判断某只股票在已有数据中是否已完成下载。

    Args:
        existing_df: 已有数据DataFrame
        code: 股票代码
        expected_date: 由 get_expected_latest_date() 确定的预期最新交易日（字符串）
        is_delisted: 是否为退市股（退市股无需更新日期检查，只要有数据即视为完成）
    """
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
        # 登录后立即确定预期最新数据日期（考虑交易日历和当前时间）
        expected_latest_date = get_expected_latest_date()
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}, "
              f"预期 BaoStock 数据库的最新交易日期: {expected_latest_date}")
        # 获取沪深主板A股股票列表
        stock_list = get_stock_list()

        # 加载已存在数据
        existing_df = load_existing_csv()
        if existing_df.empty:
            full = True
            print("  未检测到已存在数据，将切换为全量下载模式")

        # 默认模式：增量更新
        if not full:
            print("=" * 50)
            print("  增量更新模式")
            print("=" * 50)
            # 预先计算每只股票的最后日期
            print("正在分析已有数据...")
            last_date_map = {}

            if not existing_df.empty:
                last_date_map = existing_df.groupby("code")["date"].max().to_dict()
            existing_df = None  # 释放内存，增量模式不再需要全量数据

            # 增量模式：检查数据是否已包含最新交易日
            # 用 expected_latest_date 判断，而非原始"今天"：
            # - 盘前运行时 expected_latest_date = 上一交易日，数据已是最新 → 无需更新
            # - 盘后运行时 expected_latest_date = 今天，有新数据 → 触发更新
            # - 周末/节假日 expected_latest_date = 上一交易日，同盘前逻辑
            # - 新股票（CSV中无数据）：全量下载一次
            pending_update = []
            pending_full = []
            for code, name, is_delisted in stock_list:
                last_date = last_date_map.get(code)
                if last_date is None:
                    pending_full.append((code, name))
                elif is_delisted:
                    pass  # 退市股有历史数据即视为完整，跳过更新
                elif last_date < expected_latest_date:
                    pending_update.append((code, name, last_date))

            if not pending_update and not pending_full:
                print("所有股票已是最新状态，无需更新。")
                return

            print(f"需要更新股票 {len(pending_update)} 只, 新增股票 {len(pending_full)} 只")

            update_count = 0
            max_new_date = None
            new_chunks = []
            for idx, (code, name, last_date) in enumerate(tqdm(pending_update, desc="增量更新中", ncols=50)):
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
                for idx, (code, name) in enumerate(tqdm(pending_full, desc="新增股票中", ncols=50)):
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

            print(f"\n增量更新完成! 成功更新股票 {update_count} 只, 新增股票 {new_count} 只")
            return

        # ── 全量下载 / 断点续传 ──
        print("=" * 50)
        print("  全量下载模式")
        print("=" * 50)

        # 加载完成状态
        completed = load_completed()
        if not existing_df.empty:
            print(f"已有数据: {len(existing_df):,} 条, 已完成 {len(completed)} 只股票")
        else:
            print("暂无已有数据，将进行全量下载")

        skip_count = 0
        pending = []
        for code, name, is_delisted in stock_list:
            if not existing_df.empty and is_complete(existing_df, code, expected_latest_date, is_delisted):
                skip_count += 1
            else:
                pending.append((code, name, is_delisted))

        if skip_count > 0:
            print(f"跳过已完成: {skip_count} 只, 待下载/补全: {len(pending)} 只")

        if not pending:
            print("所有股票已是最新状态，无需更新。")
            return

        success_count = skip_count
        fail_count = 0

        for idx, (code, name, _) in enumerate(tqdm(pending, desc="正在下载日K数据", ncols=50)):
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

        # 最终保存
        save_to_csv(existing_df)
        save_completed(completed)

        if not existing_df.empty:
            latest = existing_df["date"].max()
            with open(os.path.join(DATA_DIR, ".last_date.txt"), "w") as f:
                f.write(str(latest))

        files = list_monthly_files()
        total_stocks = len(existing_df["code"].unique())
        print(f"\n数据已保存至 data/ 目录，共 {len(files)} 个文件")
        print(f"总记录数: {len(existing_df):,} 条")
        print(f"总股票数: {total_stocks} 只")
        print(f"    成功: {success_count} 只")
        print(f"    失败: {fail_count} 只")
        print(f"    跳过: {skip_count} 只")

    finally:
        bs.logout()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股日K数据下载")
    parser.add_argument("-f", "--full", action="store_true", help="全量下载模式，重新下载所有股票数据")
    args = parser.parse_args()
    main(full=args.full)
