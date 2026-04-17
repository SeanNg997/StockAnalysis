"""数据加载与清洗 — 加载CSV、剔除ST/停牌/新股、流动性过滤"""

import pandas as pd
import numpy as np
import os
import warnings
import gc
from config import CONFIG
warnings.filterwarnings('ignore')

BASE_DIR = CONFIG['paths']['BASE_DIR']
DATA_DIR = CONFIG['paths']['DATA_DIR']
STOCK_LIST_CSV = CONFIG['paths']['STOCK_LIST_CSV']
TRADE_DAYS_TXT = CONFIG['paths']['TRADE_DAYS_TXT']
CLEAN_PKL = CONFIG['paths']['CLEAN_PKL']
MARKET_PKL = CONFIG['paths']['MARKET_PKL']


def load_raw_data() -> pd.DataFrame:
    """加载所有月度CSV合并返回"""
    import glob as _glob
    print("[1/3] 加载月度CSV数据...")
    files = sorted(_glob.glob(os.path.join(DATA_DIR, 'Stock_dailyK_*.csv')))
    if not files:
        raise FileNotFoundError(f"未找到月度CSV文件 (Stock_dailyK_*.csv) in {DATA_DIR}")
    
    dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    print(f"  共读取到 {len(files)} 个月度文件（共 {df['code'].nunique()} 只股票）")
    print(f"  交易日范围: {df['date'].min()} ~ {df['date'].max()}")
    return df


def build_market_status(df: pd.DataFrame, end_date=None):
    """生成市场状态快照（不做过滤），用于执行日验证"""
    cols = ['code', 'date', 'isST', 'isTrading', 'open', 'close', 'volume', 'amount']
    market_df = df[cols].copy()
    market_df['date'] = pd.to_datetime(market_df['date'])

    if end_date is not None:
        market_df = market_df[market_df['date'] <= pd.Timestamp(end_date)]

    os.makedirs(os.path.dirname(MARKET_PKL), exist_ok=True)
    market_df.to_pickle(MARKET_PKL)
    return market_df


def add_is_new_flag(df: pd.DataFrame) -> pd.DataFrame:
    """添加isNew字段，新股上市前30个交易日标记为1，其他为0"""
    print("[2/3] 添加isNew字段...")

    stock_list_df = pd.read_csv(STOCK_LIST_CSV, encoding='utf-8-sig')
    stock_list_df['list_date'] = pd.to_datetime(stock_list_df['list_date'])
    trade_days = pd.read_csv(TRADE_DAYS_TXT, header=None, names=['date'])
    trade_days['date'] = pd.to_datetime(trade_days['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Merge stock_list_df with industry columns
    df = df.merge(stock_list_df[['code', 'list_date', 'industry', 'industryClassification']], on='code', how='left')

    # 向量化计算 cutoff_date
    trade_dates = trade_days['date'].values
    valid_mask = stock_list_df['list_date'].notna()
    indices = trade_dates.searchsorted(stock_list_df.loc[valid_mask, 'list_date'].values, side='left')
    # 第30个交易日的索引 = idx + 29
    valid_indices = indices + 29
    has_enough = valid_indices < len(trade_dates)
    cutoff_dates = np.full(len(stock_list_df), None, dtype='datetime64[ns]')
    valid_rows = stock_list_df.index[valid_mask]
    cutoff_dates[valid_rows[has_enough]] = trade_dates[valid_indices[has_enough]]

    stock_list_df['cutoff_date'] = cutoff_dates
    df = df.merge(stock_list_df[['code', 'cutoff_date']], on='code', how='left')
    
    # 添加isNew字段：前30个交易日标记为1，其他为0
    df['isNew'] = 0
    mask = df['cutoff_date'].notna() & (df['date'] <= df['cutoff_date'])
    df.loc[mask, 'isNew'] = 1
    
    df = df.drop(columns=['list_date', 'cutoff_date'])

    print(f"  isNew=1 的行数：{df['isNew'].sum():,} 行")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值处理：连续缺失小于等于2天用前值，否则全部设置为0"""
    print("[3/3] 处理缺失值...")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                    'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']

    df = df.sort_values(['code', 'date'])

    before_missing = df[numeric_cols].isna().sum().sum()
    
    # 对每个股票分组处理
    for col in numeric_cols:
        # 保存原始数据的副本
        original_col = df[col].copy()
        
        # 计算连续缺失天数
        def calculate_consecutive_missing(group):
            missing_mask = group.isna()
            consecutive_missing = missing_mask.astype(int).groupby((~missing_mask).cumsum()).cumsum()
            return consecutive_missing
        
        # 计算每个股票的连续缺失天数
        consecutive_missing = df.groupby('code')[col].transform(calculate_consecutive_missing)
        
        # 前值填充所有缺失值
        df[col] = df.groupby('code')[col].transform(lambda x: x.ffill())
        
        # 找出所有连续缺失超过2天的缺失值组
        # 对每个股票分组处理
        for code, group in df.groupby('code'):
            # 找出该股票的缺失值位置
            missing_indices = group[original_col.isna()].index
            if len(missing_indices) == 0:
                continue
            
            # 计算连续缺失的起始和结束位置
            consecutive_groups = []
            current_start = missing_indices[0]
            current_end = missing_indices[0]
            
            for i in range(1, len(missing_indices)):
                if missing_indices[i] == missing_indices[i-1] + 1:
                    current_end = missing_indices[i]
                else:
                    consecutive_groups.append((current_start, current_end))
                    current_start = missing_indices[i]
                    current_end = missing_indices[i]
            consecutive_groups.append((current_start, current_end))
            
            # 对于连续缺失超过2天的组，将整个组设置为0
            for start, end in consecutive_groups:
                group_length = end - start + 1
                if group_length > 2:
                    df.loc[start:end, col] = 0

    after_missing = df[numeric_cols].isna().sum().sum()
    filled = before_missing - after_missing
    
    # 处理可能的剩余缺失值（如股票的第一个值）
    df[numeric_cols] = df[numeric_cols].fillna(0)
    final_missing = df[numeric_cols].isna().sum().sum()

    print(f"  原始缺失值: {before_missing:,}")
    print(f"  处理后缺失值: {final_missing}")
    return df


def run_pipeline(end_date=None) -> pd.DataFrame:
    """执行完整数据清洗流水线"""
    if os.path.exists(CLEAN_PKL):
        try:
            cached = pd.read_pickle(CLEAN_PKL)
            cached_max = cached['date'].max()

            if end_date is not None:
                if cached_max >= pd.Timestamp(end_date):
                    cached = cached[cached['date'] <= pd.Timestamp(end_date)]
                    if os.path.exists(MARKET_PKL):
                        mkt_cached = pd.read_pickle(MARKET_PKL)
                        if mkt_cached['date'].max() >= pd.Timestamp(end_date):
                            print(f"✅ [缓存命中] mainboard_clean.pkl 已包含 {end_date}，跳过重新生成")
                            return cached
            else:
                import glob as _glob
                csv_files = sorted(_glob.glob(os.path.join(DATA_DIR, 'Stock_dailyK_*.csv')))
                if csv_files:
                    latest_csv = pd.read_csv(csv_files[-1], encoding='utf-8-sig')
                    csv_max_date = pd.to_datetime(latest_csv['date'].max())
                    if cached_max == csv_max_date:
                        if os.path.exists(MARKET_PKL):
                            mkt_cached = pd.read_pickle(MARKET_PKL)
                            if mkt_cached['date'].max() == csv_max_date:
                                print(f"✅ [缓存命中] mainboard_clean.pkl 已是最新 ({cached_max.date()})，跳过重新生成")
                                return cached
        except Exception as e:
            print(f"  [缓存读取失败: {e}]，继续重新生成...")
            pass

    df = load_raw_data()
    build_market_status(df, end_date=end_date)
    df = add_is_new_flag(df) 
    df = handle_missing(df)
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    if end_date is not None:
        df = df[df['date'] <= pd.Timestamp(end_date)].copy()
        print(f"  [date filter] 数据截断至 {end_date}")

    os.makedirs(os.path.dirname(CLEAN_PKL), exist_ok=True)
    df.to_pickle(CLEAN_PKL)
    print(f"\n✅ 市场状态快照已保存至 {MARKET_PKL}")
    print(f"✅ 清洗完成! 保存至 {CLEAN_PKL}")
    print(f"  最终数据: {df.shape[0]:,} 行, {df['code'].nunique()} 只股票")
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
