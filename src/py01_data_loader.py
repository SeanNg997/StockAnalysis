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
    print("[1/6] 加载月度CSV数据...")
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
    print(f"  ✅ 市场状态快照已保存至 {MARKET_PKL}")
    print(f"     {market_df.shape[0]:,} 行, {market_df['code'].nunique()} 只股票")
    return market_df



def remove_st(df: pd.DataFrame) -> pd.DataFrame:
    """剔除ST/*ST状态的日记录（基于逐日isST字段，避免前瞻偏差）"""
    print("[2/6] 剔除ST/*ST股票（仅剔除ST状态的日记录）...")
    st_mask = df['isST'] == 1
    df = df[~st_mask].copy()
    print(f"  剔除后：{len(df):,} 行")
    return df


def remove_suspended(df: pd.DataFrame) -> pd.DataFrame:
    """剔除停牌日"""
    print("[3/6] 剔除停牌日...")
    df = df[df['isTrading'] == 1].copy()
    print(f"  剔除后：{len(df):,} 行")

    return df


def remove_new(df: pd.DataFrame) -> pd.DataFrame:
    """剔除新股上市前30个交易日"""
    print("[4/6] 剔除新股上市的前30个交易日...")

    stock_list_df = pd.read_csv(STOCK_LIST_CSV, encoding='utf-8-sig')
    stock_list_df['list_date'] = pd.to_datetime(stock_list_df['list_date'])
    trade_days = pd.read_csv(TRADE_DAYS_TXT, header=None, names=['date'])
    trade_days['date'] = pd.to_datetime(trade_days['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(stock_list_df[['code', 'list_date']], on='code', how='left')

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
    mask = df['cutoff_date'].notna() & (df['date'] > df['cutoff_date'])
    df = df[mask].copy()
    df = df.drop(columns=['list_date', 'cutoff_date'])

    print(f"  剔除后：{len(df):,} 行")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值处理：组内前值填充 + 零填充"""
    print("[5/6] 处理缺失值...")
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                    'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']

    df = df.sort_values(['code', 'date'])
    df[numeric_cols] = df.groupby('code')[numeric_cols].transform(lambda x: x.ffill())

    df[numeric_cols] = df[numeric_cols].fillna(0)

    missing = df[numeric_cols].isna().sum().sum()
    print(f"  处理后残余缺失值: {missing}")
    return df


def filter_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """过滤流动性不足的股票"""
    print("[6/6] 流动性过滤...")
    df = df.sort_values(['code', 'date'])

    def rolling_mean(x):
        return x.rolling(20, min_periods=10).mean()
    
    df['avg_amount_20'] = df.groupby('code')['amount'].transform(rolling_mean)

    min_avg_amount = CONFIG['data_loader']['MIN_AVG_AMOUNT']
    df = df[df['avg_amount_20'] >= min_avg_amount].copy()
    df = df.drop(columns=['avg_amount_20'])

    min_trading_days = CONFIG['data_loader']['MIN_TRADING_DAYS']
    stock_counts = df.groupby('code').size()
    valid_stocks = stock_counts[stock_counts >= min_trading_days].index
    df = df[df['code'].isin(valid_stocks)].copy()

    print(f"  剔除后：{len(df):,} 行")
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
    df = remove_st(df)
    df = remove_suspended(df)
    df = remove_new(df) 
    df = handle_missing(df)
    df = filter_liquidity(df)

    df = df.sort_values(['date', 'code']).reset_index(drop=True)

    df['date'] = pd.to_datetime(df['date'])

    if end_date is not None:
        df = df[df['date'] <= pd.Timestamp(end_date)].copy()
        print(f"  [date filter] 数据截断至 {end_date}")

    os.makedirs(os.path.dirname(CLEAN_PKL), exist_ok=True)
    df.to_pickle(CLEAN_PKL)
    print(f"\n✅ 清洗完成! 保存至 {CLEAN_PKL}")
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
