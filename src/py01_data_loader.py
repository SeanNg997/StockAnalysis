"""
py01_data_loader.py — 数据加载与清洗模块
==========================================
职责：
1. 加载原始CSV日K数据
2. 过滤仅保留沪深主板股票（代码以sh.60或sz.00开头）
3. 剔除ST/*ST股票、停牌日、新股上市前5日
4. 缺失值处理
5. 流动性过滤（日均成交额 < 500万剔除）
6. 输出清洗后的pickle文件加速后续加载
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLEAN_PKL = os.path.join(BASE_DIR, 'data', 'mainboard_clean.pkl')


def load_raw_data() -> pd.DataFrame:
    """加载所有月度CSV数据并合并"""
    import glob as _glob
    print("[1/6] 加载月度CSV数据...")
    files = sorted(_glob.glob(os.path.join(DATA_DIR, 'Stock_dailyK_*.csv')))
    if not files:
        raise FileNotFoundError(f"未找到月度CSV文件 (Stock_dailyK_*.csv) in {DATA_DIR}")
    dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  读取 {len(files)} 个月度文件，原始数据: {df.shape[0]:,} 行, {df['代码'].nunique()} 只股票")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    return df


def filter_mainboard(df: pd.DataFrame) -> pd.DataFrame:
    """仅保留沪深主板股票（sh.60开头或sz.00开头）"""
    print("[2/6] 过滤主板股票...")
    mask = df['代码'].str.startswith('sh.60') | df['代码'].str.startswith('sz.00')
    df = df[mask].copy()
    print(f"  主板股票: {df['代码'].nunique()} 只, {df.shape[0]:,} 行")
    return df


def remove_st(df: pd.DataFrame) -> pd.DataFrame:
    """剔除ST/*ST股票（基于名称列）"""
    print("[3/6] 剔除ST股票...")
    n_before = df['代码'].nunique()
    # ST标记在名称中：ST、*ST、S*ST等
    st_mask = df['名称'].str.contains(r'\*?ST', case=True, regex=True, na=False)
    df = df[~st_mask].copy()
    n_after = df['代码'].nunique()
    print(f"  剔除含ST记录后: {n_after} 只股票 (移除了 {n_before - n_after} 只纯ST股)")
    return df


def remove_suspended_and_new(df: pd.DataFrame) -> pd.DataFrame:
    """
    剔除停牌日和新股上市前5个交易日
    停牌判断：volume=0 或 open/high/low/close全相等且volume=0
    """
    print("[4/6] 剔除停牌日和新股上市前5日...")
    n_before = len(df)

    # 剔除成交量为0的停牌日
    df = df[df['volume'] > 0].copy()

    # 剔除涨跌幅缺失的行
    df = df.dropna(subset=['pctChg'])

    # 按股票排序，剔除每只股票最早的5个交易日（新股效应）
    df = df.sort_values(['代码', 'date']).reset_index(drop=True)
    df['_rank'] = df.groupby('代码').cumcount()
    df = df[df['_rank'] >= 5].drop(columns=['_rank'])

    print(f"  剔除后: {len(df):,} 行 (移除 {n_before - len(df):,} 行)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值处理"""
    print("[5/6] 处理缺失值...")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                    'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']

    # 组内前值填充
    df = df.sort_values(['代码', 'date'])
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df.groupby('代码')[col].ffill()

    # 仍然缺失的用0填充（主要是基本面指标）
    df[numeric_cols] = df[numeric_cols].fillna(0)

    missing = df[numeric_cols].isna().sum().sum()
    print(f"  处理后残余缺失值: {missing}")
    return df


def filter_liquidity(df: pd.DataFrame, min_avg_amount: float = 500e4) -> pd.DataFrame:
    """
    过滤流动性不足的股票
    标准：20日平均成交额 < min_avg_amount (默认500万元)
    """
    print("[6/6] 流动性过滤...")
    df = df.sort_values(['代码', 'date'])

    # 计算20日滚动平均成交额
    df['avg_amount_20'] = df.groupby('代码')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )

    n_before = len(df)
    df = df[df['avg_amount_20'] >= min_avg_amount].copy()
    df = df.drop(columns=['avg_amount_20'])

    # 过滤后确保每只股票至少有120个交易日
    stock_counts = df.groupby('代码').size()
    valid_stocks = stock_counts[stock_counts >= 120].index
    df = df[df['代码'].isin(valid_stocks)].copy()

    print(f"  流动性过滤后: {df['代码'].nunique()} 只股票, {len(df):,} 行")
    return df


def run_pipeline(end_date=None) -> pd.DataFrame:
    """执行完整数据清洗流水线

    Args:
        end_date: 数据截止日期（含），格式 'YYYY-MM-DD'。None 表示使用全部数据。
    """
    # 缓存命中检查：若 pkl 已存在且最新日期与目标日期一致，直接复用
    if end_date is not None and os.path.exists(CLEAN_PKL):
        try:
            cached = pd.read_pickle(CLEAN_PKL)
            cached_max = cached['date'].max()
            if pd.Timestamp(end_date) == cached_max:
                print(f"✅ [缓存命中] mainboard_clean.pkl 已是 {end_date}，跳过重新生成")
                return cached
        except Exception:
            pass  # 读取失败则继续正常流程

    df = load_raw_data()
    df = filter_mainboard(df)
    df = remove_st(df)
    df = remove_suspended_and_new(df)
    df = handle_missing(df)
    df = filter_liquidity(df)

    # 最终排序与索引重置
    df = df.sort_values(['date', '代码']).reset_index(drop=True)

    # 日期转换
    df['date'] = pd.to_datetime(df['date'])

    # 截断到指定日期
    if end_date is not None:
        df = df[df['date'] <= pd.Timestamp(end_date)].copy()
        print(f"  [date filter] 数据截断至 {end_date}")

    # 保存pickle
    df.to_pickle(CLEAN_PKL)
    print(f"\n✅ 清洗完成! 保存至 {CLEAN_PKL}")
    print(f"  最终数据: {df.shape[0]:,} 行, {df['代码'].nunique()} 只股票")
    print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

    return df


if __name__ == '__main__':
    import sys as _sys
    _end_date = None
    _args = _sys.argv[1:]
    if '--date' in _args:
        _idx = _args.index('--date')
        _end_date = _args[_idx + 1]
    run_pipeline(end_date=_end_date)
