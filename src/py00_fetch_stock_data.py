"""
爬取A股近10年日K线数据
使用 baostock 免费数据源，数据保存至 data/ 目录

功能:
  - 全量下载 / 断点续传 / 增量更新
  - 每50只股票自动保存，防止中断丢失
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

OUTPUT_FILE = os.path.join(DATA_DIR, "a_stock_daily_k_10y.csv")

START_DATE = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")

SAVE_EVERY = 50  # 每下载50只股票保存一次

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
    """加载已有CSV数据，不存在则返回空DataFrame"""
    if os.path.exists(OUTPUT_FILE):
        return pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
    return pd.DataFrame()


def save_to_csv(df: pd.DataFrame):
    """保存DataFrame到CSV"""
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ── 数据源 ──────────────────────────────────────────────

def get_stock_list():
    """获取全部A股股票列表"""
    print("正在获取A股股票列表...")
    rs = bs.query_stock_basic()
    stock_list = []
    while rs.next():
        row = rs.get_row_data()
        if row[4] == "1" and row[5] == "1":
            stock_list.append((row[0], row[1]))
    print(f"共获取 {len(stock_list)} 只上市股票")
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

def is_complete(existing_df: pd.DataFrame, code: str) -> bool:
    """
    判断某只股票在已有数据中是否完整。
    逻辑：该股票的最后一天 >= END_DATE 的前一个交易日（允许1天偏差）
    """
    stock_data = existing_df[existing_df["代码"] == code]
    if stock_data.empty:
        return False
    last_date = pd.to_datetime(stock_data["date"].max())
    expected_last = pd.to_datetime(END_DATE) - timedelta(days=1)
    return last_date >= expected_last


def main(limit: int = 0, update: bool = False):
    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock 登录失败: {lg.error_msg}")
        return

    try:
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
            # 增量模式：只处理已完成且最后日期不是最新的股票
            codes_to_update = []
            for code, name in stock_list:
                if code not in completed:
                    continue
                stock_data = existing_df[existing_df["代码"] == code]
                if stock_data.empty:
                    continue
                last_date = stock_data["date"].max()
                if last_date < END_DATE:
                    codes_to_update.append((code, name, last_date))

            if not codes_to_update:
                print("所有股票数据已是最新，无需更新")
                return

            print(f"需要更新 {len(codes_to_update)} 只股票")

            update_count = 0
            for idx, (code, name, last_date) in enumerate(tqdm(codes_to_update, desc="增量更新中", ncols=80)):
                from_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
                new_df = fetch_daily_increment(code, name, from_date)
                if new_df is not None and not new_df.empty:
                    # 追加新数据（保留历史数据）
                    existing_df = pd.concat([existing_df, new_df], ignore_index=True)
                    update_count += 1
                time.sleep(0.1)

                if (idx + 1) % SAVE_EVERY == 0:
                    save_to_csv(existing_df)
                    print(f"  已自动保存 (更新 {idx + 1}/{len(codes_to_update)})")

            save_to_csv(existing_df)
            # 记录最新数据日期
            latest = existing_df["date"].max()
            with open(os.path.join(DATA_DIR, ".last_date.txt"), "w") as f:
                f.write(str(latest))
            print(f"\n增量更新完成! 成功更新 {update_count} 只, 数据保存至: {OUTPUT_FILE}")
            return

        # ── 全量下载 / 断点续传 ──
        skip_count = 0
        pending = []
        for code, name in stock_list:
            if code in completed and not existing_df.empty and is_complete(existing_df, code):
                skip_count += 1
            else:
                pending.append((code, name))

        if skip_count > 0:
            print(f"跳过已完成: {skip_count} 只, 待下载/补全: {len(pending)} 只")

        if not pending:
            print("所有股票已下载完毕!")
            return

        success_count = skip_count  # 继承之前已成功的数量
        fail_count = 0

        for idx, (code, name) in enumerate(tqdm(pending, desc="正在下载日K数据", ncols=80)):
            df = fetch_daily_full(code, name)
            if df is not None and not df.empty:
                # 如果已有该股票的部分数据，先移除再合并（处理中断情况）
                if not existing_df.empty and code in existing_df["代码"].values:
                    existing_df = existing_df[existing_df["代码"] != code]
                existing_df = pd.concat([existing_df, df], ignore_index=True)
                completed.add(code)
                success_count += 1
            else:
                fail_count += 1

            time.sleep(0.1)

            # 定期保存
            total_done = skip_count + idx + 1
            if total_done % SAVE_EVERY == 0:
                save_to_csv(existing_df)
                save_completed(completed)
                print(f"  已自动保存 (进度 {total_done}/{len(stock_list)})")

        # 最终保存
        save_to_csv(existing_df)
        save_completed(completed)

        # 记录最新数据日期
        if not existing_df.empty:
            latest = existing_df["date"].max()
            with open(os.path.join(DATA_DIR, ".last_date.txt"), "w") as f:
                f.write(str(latest))

        total_stocks = len(existing_df["代码"].unique())
        print(f"\n数据已保存至: {OUTPUT_FILE}")
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
