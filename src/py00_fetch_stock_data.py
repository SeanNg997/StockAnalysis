"""
爬取A股近10年日K线数据
使用 baostock 免费数据源，数据保存至 data/ 目录

功能:
  - 全量下载 / 断点续传 / 增量更新
  - 数据按月拆分存储，文件名格式: Stock_dailyK_YYYYMM.csv
  - 每200只股票自动保存，防止中断丢失
"""

import baostock as bs
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import time
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = "2019-12-01"

def get_end_date():
    """获取今天日期，用于 baostock API 的 end_date 参数"""
    return datetime.now().strftime("%Y-%m-%d")

END_DATE = get_end_date()

SAVE_EVERY = 200  # 每下载200只股票保存一次

# baostock 一般在收盘后、约 17:00 前完成当日数据入库
# 早于此时间认为当日数据尚未就绪，预期最新数据仍为上一交易日
MARKET_DATA_READY_HOUR = 17


def get_expected_latest_date() -> str:
    """
    根据当前时间和 baostock 交易日历，确定当前 baostock 中
    预期已入库的最新交易日期。

    规则：
    - 当前时间 >= 17:00 且今天是交易日 → 预期最新 = 今天
    - 当前时间 < 17:00，或今天是非交易日（周末/节假日） → 预期最新 = 上一个交易日

    注意：调用前须已登录 baostock（bs.login()）。
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # 查近 45 个日历日的交易日历（覆盖节假日连休场景）
    start_str = (now - timedelta(days=45)).strftime("%Y-%m-%d")
    rs = bs.query_trade_dates(start_date=start_str, end_date=today_str)
    trading_days = []
    while rs.next():
        row = rs.get_row_data()
        if row[1] == "1":
            trading_days.append(row[0])  # 'YYYY-MM-DD'

    if not trading_days:
        # 降级：若接口异常则回退到昨天
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    today_is_trading = trading_days[-1] == today_str

    if today_is_trading and now.hour >= MARKET_DATA_READY_HOUR:
        # 今天是交易日且已过数据就绪时间
        return today_str
    else:
        # 盘前 / 非交易日：预期最新数据为今天之前最近一个交易日
        prev_days = [d for d in trading_days if d < today_str]
        return prev_days[-1] if prev_days else trading_days[-1]

# 测试用的10只股票
TEST_STOCKS = [
    ("sh.600000", "浦发银行"),
    ("sh.600036", "招商银行"),
    ("sh.601318", "中国平安"),
    ("sz.000001", "平安银行"),
    ("sz.000002", "万科A"),
    ("sh.600519", "贵州茅台"),
    ("sz.000858", "五粮液"),
    ("sh.600887", "伊利股份"),
    ("sz.000333", "美的集团"),
    ("sh.601398", "工商银行"),
]


# ── 月度文件管理 ──────────────────────────────────────────────

def get_monthly_file(yyyymm: str) -> str:
    """返回指定年月的CSV文件路径，yyyymm 格式如 '202004'"""
    return os.path.join(DATA_DIR, f"Stock_dailyK_{yyyymm}.csv")


def list_monthly_files() -> list:
    """返回所有月度CSV文件路径（按时间排序）"""
    return sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))


# ── 数据库文件 ──────────────────────────────────────────────

def _db_file():
    return os.path.join(DATA_DIR, ".download_status.txt")


def load_completed():
    """从状态文件读取已完成的股票代码集合"""
    path = _db_file()
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_completed(completed: set):
    """持久化已完成的股票代码集合"""
    path = _db_file()
    with open(path, "w") as f:
        for code in sorted(completed):
            f.write(code + "\n")


# ── CSV 读写 ──────────────────────────────────────────────

def load_existing_csv() -> pd.DataFrame:
    """加载所有月度CSV数据合并返回，无文件则返回空DataFrame"""
    files = list_monthly_files()
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    return pd.concat(dfs, ignore_index=True)


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
            # 仅删除与新数据代码+日期完全重复的行，保留同股票其他日期的旧数据
            new_keys = group[["代码", "date"]].drop_duplicates()
            new_keys["_dup"] = True
            existing_month = existing_month.merge(new_keys, on=["代码", "date"], how="left")
            existing_month = existing_month[existing_month["_dup"] != True].drop(columns=["_dup"])
            combined = pd.concat([existing_month, group], ignore_index=True)
            combined = combined.sort_values(["代码", "date"]).reset_index(drop=True)
            combined.to_csv(monthly_file, index=False, encoding="utf-8-sig")
        else:
            group.sort_values(["代码", "date"]).reset_index(drop=True).to_csv(
                monthly_file, index=False, encoding="utf-8-sig"
            )


# ── 数据源 ──────────────────────────────────────────────

def get_stock_list():
    """获取沪深主板A股股票列表（sh.60开头或sz.00开头）"""
    print("正在获取沪深主板股票列表...")
    rs = bs.query_stock_basic()
    stock_list = []
    while rs.next():
        row = rs.get_row_data()
        # row[0] 是股票代码，row[4] 和 row[5] 是上市状态
        # 筛选条件：上市状态=1 且 仅保留主板（sh.60* 或 sz.00*）
        if row[4] == "1" and row[5] == "1":
            code = row[0]
            if code.startswith('sh.60') or code.startswith('sz.00'):
                stock_list.append((code, row[1]))
    print(f"共获取 {len(stock_list)} 只主板上市股票")
    return stock_list


def _to_dataframe(rows, fields, symbol, name):
    """将baostock返回数据转为DataFrame"""
    df = pd.DataFrame(rows, columns=fields)
    df.insert(0, "名称", name)
    df.insert(0, "代码", symbol)
    for col in ["open", "high", "low", "close", "volume", "amount", "turn",
                 "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_daily_full(symbol: str, name: str) -> pd.DataFrame | None:
    """获取单只股票完整10年日K线"""
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=START_DATE,
        end_date=END_DATE,
        frequency="d",
        adjustflag="2",
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if rows:
        return _to_dataframe(rows, rs.fields, symbol, name)
    return None


def fetch_daily_increment(symbol: str, name: str, from_date: str) -> pd.DataFrame | None:
    """获取单只股票从 from_date 到 END_DATE 的增量日K线"""
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=from_date,
        end_date=END_DATE,
        frequency="d",
        adjustflag="2",
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if rows:
        return _to_dataframe(rows, rs.fields, symbol, name)
    return None


# ── 核心逻辑 ──────────────────────────────────────────────

def is_complete(existing_df: pd.DataFrame, code: str, expected_date: str) -> bool:
    """
    判断某只股票在已有数据中是否已包含 expected_date 的数据。

    Args:
        expected_date: 由 get_expected_latest_date() 确定的预期最新交易日（字符串）
    """
    stock_data = existing_df[existing_df["代码"] == code]
    if stock_data.empty:
        return False
    last_date = stock_data["date"].max()
    return last_date >= expected_date


def main(limit: int = 0, update: bool = False):
    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock 登录失败: {lg.error_msg}")
        return

    try:
        # 登录后立即确定预期最新数据日期（考虑交易日历和当前时间）
        expected_latest_date = get_expected_latest_date()
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}, "
              f"预期最新数据日期: {expected_latest_date}")
        # 获取股票列表
        if limit > 0 and limit <= len(TEST_STOCKS):
            stock_list = TEST_STOCKS[:limit]
            print(f"测试模式: 下载前 {limit} 只股票")
        else:
            stock_list = get_stock_list()

        # 加载已有数据与完成状态
        existing_df = load_existing_csv()
        completed = load_completed()

        if not existing_df.empty:
            print(f"已有数据: {len(existing_df):,} 条, 已完成 {len(completed)} 只股票")
        else:
            print("暂无已有数据，将从头开始下载")

        if update:
            print("=" * 50)
            print("  增量更新模式")
            print("=" * 50)
            # 预先计算每只股票的最后日期
            print("正在分析已有数据...")
            last_date_map = {}
            if not existing_df.empty:
                last_date_map = existing_df.groupby("代码")["date"].max().to_dict()
            existing_df = None  # 释放内存，增量模式不再需要全量数据

            # 增量模式：检查数据是否已包含最新交易日
            # 用 expected_latest_date 判断，而非原始"今天"：
            # - 盘前运行时 expected_latest_date = 上一交易日，数据已是最新 → 无需更新
            # - 盘后运行时 expected_latest_date = 今天，有新数据 → 触发更新
            # - 周末/节假日 expected_latest_date = 上一交易日，同盘前逻辑
            codes_to_update = []
            for code, name in stock_list:
                last_date = last_date_map.get(code)
                if last_date is None:
                    continue  # 该股票在CSV中无数据，跳过（增量模式只更新已有数据的股票）
                if last_date < expected_latest_date:
                    codes_to_update.append((code, name, last_date))

            if not codes_to_update:
                print(f"所有股票数据已是最新（{expected_latest_date}），无需更新")
                return

            print(f"需要更新 {len(codes_to_update)} 只股票")

            update_count = 0
            max_new_date = None
            new_chunks = []
            for idx, (code, name, last_date) in enumerate(tqdm(codes_to_update, desc="增量更新中", ncols=80)):
                from_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                new_df = fetch_daily_increment(code, name, from_date)
                if new_df is not None and not new_df.empty:
                    new_chunks.append(new_df)
                    update_count += 1
                time.sleep(0.1)

                if (idx + 1) % SAVE_EVERY == 0 and new_chunks:
                    batch = pd.concat(new_chunks, ignore_index=True)
                    save_incremental_months(batch)
                    if max_new_date is None:
                        max_new_date = batch["date"].max()
                    else:
                        max_new_date = max(max_new_date, batch["date"].max())
                    new_chunks = []
                    print(f"  已自动保存 (更新 {idx + 1}/{len(codes_to_update)})")

            if new_chunks:
                batch = pd.concat(new_chunks, ignore_index=True)
                save_incremental_months(batch)
                if max_new_date is None:
                    max_new_date = batch["date"].max()
                else:
                    max_new_date = max(max_new_date, batch["date"].max())

            if max_new_date:
                with open(os.path.join(DATA_DIR, ".last_date.txt"), "w") as f:
                    f.write(str(max_new_date))

            files = list_monthly_files()
            print(f"\n增量更新完成! 成功更新 {update_count} 只, 月度文件共 {len(files)} 个")
            return

        # ── 全量下载 / 断点续传 ──
        skip_count = 0
        pending = []
        for code, name in stock_list:
            if code in completed and not existing_df.empty and is_complete(existing_df, code, expected_latest_date):
                skip_count += 1
            else:
                pending.append((code, name))

        if skip_count > 0:
            print(f"跳过已完成: {skip_count} 只, 待下载/补全: {len(pending)} 只")

        if not pending:
            print("所有股票已下载完毕!")
            return

        success_count = skip_count
        fail_count = 0

        for idx, (code, name) in enumerate(tqdm(pending, desc="正在下载日K数据", ncols=80)):
            df = fetch_daily_full(code, name)
            if df is not None and not df.empty:
                if not existing_df.empty and code in existing_df["代码"].values:
                    existing_df = existing_df[existing_df["代码"] != code]
                existing_df = pd.concat([existing_df, df], ignore_index=True)
                completed.add(code)
                success_count += 1
            else:
                fail_count += 1

            time.sleep(0.1)

            total_done = skip_count + idx + 1
            if total_done % SAVE_EVERY == 0:
                save_to_csv(existing_df)
                save_completed(completed)
                print(f"  已自动保存 (进度 {total_done}/{len(stock_list)})")

        # 最终保存
        save_to_csv(existing_df)
        save_completed(completed)

        if not existing_df.empty:
            latest = existing_df["date"].max()
            with open(os.path.join(DATA_DIR, ".last_date.txt"), "w") as f:
                f.write(str(latest))

        files = list_monthly_files()
        total_stocks = len(existing_df["代码"].unique())
        print(f"\n数据已保存至月度文件，共 {len(files)} 个文件")
        print(f"总记录数: {len(existing_df):,}, 股票数: {total_stocks}")
        print(f"成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}, 总计: {len(stock_list)}")

    finally:
        bs.logout()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股日K数据下载")
    parser.add_argument("-n", "--limit", type=int, default=0, help="限制下载数量，0表示全部")
    parser.add_argument("-u", "--update", action="store_true", help="增量更新模式，只下载缺失的最近数据")
    args = parser.parse_args()
    main(limit=args.limit, update=args.update)
