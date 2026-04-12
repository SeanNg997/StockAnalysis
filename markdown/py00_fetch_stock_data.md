# py00_fetch_stock_data.py 代码解释

## 文件概览

本文件用于爬取A股近10年日K线数据，使用baostock免费数据源，数据保存至data/目录。具有全量下载、断点续传、增量更新等功能，数据按月拆分存储。

## 逐行代码解释

### 第1-9行：文件头部注释

```python
"""
爬取A股近10年日K线数据
使用 baostock 免费数据源，数据保存至 data/ 目录

功能:
  - 全量下载 / 断点续传 / 增量更新
  - 数据按月拆分存储，文件名格式: Stock_dailyK_YYYYMM.csv
  - 每200只股票自动保存，防止中断丢失
"""
```
- **第1-9行**：文件头部的文档字符串，说明该文件的功能是爬取A股近10年日K线数据，使用baostock免费数据源，数据保存至data/目录，并列出了主要功能。

### 第11-24行：导入模块和设置基础路径

```python
import baostock as bs
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import gc
import time
import glob

from config import CONFIG

BASE_DIR = CONFIG['paths']['BASE_DIR']
DATA_DIR = CONFIG['paths']['DATA_DIR']
os.makedirs(DATA_DIR, exist_ok=True)
```
- **第11行**：导入baostock库，用于获取股票数据。
- **第12行**：导入pandas库，用于数据处理。
- **第13行**：导入tqdm库，用于显示进度条。
- **第14行**：从datetime模块导入datetime和timedelta类，用于日期处理。
- **第15行**：导入os模块，用于文件和路径操作。
- **第16行**：导入gc模块，用于垃圾回收。
- **第17行**：导入time模块，用于添加延迟。
- **第18行**：导入glob模块，用于文件匹配。
- **第20行**：从config模块导入CONFIG配置。
- **第22行**：获取基础目录路径。
- **第23行**：获取数据目录路径。
- **第24行**：创建数据目录，如果目录已存在则不报错。

### 第26-39行：设置配置参数

```python
START_DATE = CONFIG['data_fetch']['START_DATE']

def get_end_date():
    """获取今天日期，用于 baostock API 的 end_date 参数"""
    return datetime.now().strftime("%Y-%m-%d")

END_DATE = get_end_date()

ADJUST_FLAG = CONFIG['data_fetch']['ADJUST_FLAG']  # 调整标志位（1：后复权；2：前复权；3：不复权）
SAVE_EVERY = CONFIG['data_fetch']['SAVE_EVERY']  # 每下载指定数量只股票保存一次

# baostock 一般在收盘后、约 18:00 前完成当日数据入库
# 早于此时间认为当日数据尚未就绪，预期最新数据仍为上一交易日
MARKET_DATA_READY_HOUR = CONFIG['data_fetch']['MARKET_DATA_READY_HOUR']
```
- **第26行**：获取数据开始日期。
- **第28-30行**：定义get_end_date函数，用于获取今天日期，返回格式为"YYYY-MM-DD"。
- **第32行**：调用get_end_date函数获取结束日期。
- **第34行**：获取调整标志位，用于baostock API的adjustflag参数。
- **第35行**：获取每下载多少只股票保存一次的参数。
- **第39行**：获取市场数据就绪时间（小时）。

### 第42-78行：get_expected_latest_date函数

```python
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
```
- **第42-78行**：定义get_expected_latest_date函数，用于根据当前时间和baostock交易日历，确定当前baostock中预期已入库的最新交易日期。
  - **第43-52行**：函数文档字符串，说明函数的作用和规则。
  - **第53行**：获取当前时间。
  - **第54行**：将当前时间转换为字符串格式"YYYY-MM-DD"。
  - **第57行**：计算45天前的日期，用于查询交易日历。
  - **第58行**：调用bs.query_trade_dates查询交易日历。
  - **第59行**：初始化trading_days列表，用于存储交易日。
  - **第60-63行**：遍历查询结果，将交易日添加到trading_days列表中。
  - **第65-67行**：如果trading_days为空（接口异常），则回退到昨天。
  - **第69行**：判断今天是否是交易日。
  - **第71-73行**：如果今天是交易日且已过数据就绪时间，返回今天。
  - **第75-77行**：否则，返回今天之前最近一个交易日。

### 第81-88行：月度文件管理函数

```python
def get_monthly_file(yyyymm: str) -> str:
    """返回指定年月的CSV文件路径，yyyymm 格式如 '202004'"""
    return os.path.join(DATA_DIR, f"Stock_dailyK_{yyyymm}.csv")


def list_monthly_files() -> list:
    """返回所有月度CSV文件路径（按时间排序）"""
    return sorted(glob.glob(os.path.join(DATA_DIR, "Stock_dailyK_*.csv")))
```
- **第81-83行**：定义get_monthly_file函数，根据年月返回CSV文件路径。
- **第86-88行**：定义list_monthly_files函数，返回所有月度CSV文件路径（按时间排序）。

### 第93-111行：数据库文件操作函数

```python
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
```
- **第93-94行**：定义_db_file函数，返回状态文件路径。
- **第97-103行**：定义load_completed函数，从状态文件读取已完成的股票代码集合。
- **第106-111行**：定义save_completed函数，持久化已完成的股票代码集合。

### 第116-164行：CSV读写函数

```python
def load_existing_csv() -> pd.DataFrame:
    """加载所有月度CSV数据合并返回，无文件则返回空DataFrame"""
    files = list_monthly_files()
    if not files:
        return pd.DataFrame()
    df = pd.DataFrame()
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
                # 使用更高效的去重方法
                combined = pd.concat([existing_month, group], ignore_index=True)
                combined = combined.drop_duplicates(subset=["代码", "date"], keep="last")
                combined = combined.sort_values(["代码", "date"]).reset_index(drop=True)
                combined.to_csv(monthly_file, index=False, encoding="utf-8-sig")
        else:
            group.sort_values(["代码", "date"]).reset_index(drop=True).to_csv(
                monthly_file, index=False, encoding="utf-8-sig"
            )
```
- **第116-127行**：定义load_existing_csv函数，加载所有月度CSV数据合并返回，无文件则返回空DataFrame。
- **第130-138行**：定义save_to_csv函数，按月拆分DataFrame，写入各月度CSV文件（全量覆盖）。
- **第141-164行**：定义save_incremental_months函数，用于增量更新，仅重写new_df中涉及的月度文件。

### 第169-233行：数据源相关函数

```python
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
    # 批量转换数值列
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turn",
                    "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def fetch_daily_full(symbol: str, name: str) -> pd.DataFrame | None:
    """获取单只股票完整10年日K线"""
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=START_DATE,
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


def fetch_daily_increment(symbol: str, name: str, from_date: str) -> pd.DataFrame | None:
    """获取单只股票从 from_date 到 END_DATE 的增量日K线"""
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=from_date,
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
```
- **第169-183行**：定义get_stock_list函数，获取沪深主板A股股票列表（sh.60开头或sz.00开头）。
- **第186-197行**：定义_to_dataframe函数，将baostock返回数据转为DataFrame。
- **第200-215行**：定义fetch_daily_full函数，获取单只股票完整10年日K线。
- **第218-233行**：定义fetch_daily_increment函数，获取单只股票从from_date到END_DATE的增量日K线。

### 第238-250行：is_complete函数

```python
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
```
- **第238-250行**：定义is_complete函数，判断某只股票在已有数据中是否已包含expected_date的数据。

### 第252-399行：main函数

```python
def main(full: bool = False):
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
        stock_list = get_stock_list()

        # 加载已有数据与完成状态
        existing_df = load_existing_csv()
        completed = load_completed()
        if not existing_df.empty:
            print(f"已有数据: {len(existing_df):,} 条, 已完成 {len(completed)} 只股票")
        else:
            print("暂无已有数据，将从头开始下载")

        # 默认模式：增量更新
        if not full:
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
        print("=" * 50)
        print("  全量下载模式")
        print("=" * 50)
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
```
- **第252-399行**：定义main函数，是程序的核心逻辑。
  - **第253-256行**：登录baostock，如果登录失败则退出。
  - **第258-398行**：try-finally块，确保无论如何都会注销baostock。
  - **第260-262行**：获取预期最新数据日期并打印。
  - **第264行**：获取股票列表。
  - **第267-272行**：加载已有数据与完成状态。
  - **第275-340行**：增量更新模式，仅更新已有数据的股票。
  - **第343-396行**：全量下载模式，下载所有股票数据。
  - **第398行**：注销baostock。

### 第402-407行：主程序入口

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A股日K数据下载")
    parser.add_argument("-f", "--full", action="store_true", help="全量下载模式，重新下载所有股票数据")
    args = parser.parse_args()
    main(full=args.full)
```
- **第402-407行**：主程序入口，使用argparse解析命令行参数，支持全量下载模式。