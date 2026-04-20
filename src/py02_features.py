"""特征工程 — 技术指标 / 动量 / 截面特征 / 标签生成"""

import pandas as pd
import numpy as np
import os
import gc
import warnings

from config import CONFIG

warnings.filterwarnings('ignore')

BASE_DIR = CONFIG['paths']['BASE_DIR']
CLEAN_PKL = CONFIG['paths']['CLEAN_PKL']
FEATURE_PKL = CONFIG['paths']['FEATURE_PKL']
TRADE_DAYS_TXT = CONFIG['paths']['TRADE_DAYS_TXT']

def compute_features_for_stock(g: pd.DataFrame, trade_day_idx: dict = None) -> pd.DataFrame:
    """对单只股票计算所有特征（输入须按date升序）"""
    close = g['close'].values.astype(np.float32)
    open_ = g['open'].values.astype(np.float32)
    high = g['high'].values.astype(np.float32)
    low = g['low'].values.astype(np.float32)
    volume = g['volume'].values.astype(np.float32)
    amount = g['amount'].values.astype(np.float32)
    turn = g['turn'].values.astype(np.float32)
    is_trading = g['isTrading'].values.astype(int)
    n = len(g)

    feats = {}
    
    # 计算连续停牌天数
    consecutive_suspend = np.zeros(n, dtype=int)
    current_suspend = 0
    for i in range(n):
        if is_trading[i] != 1:
            current_suspend += 1
        else:
            current_suspend = 0
        consecutive_suspend[i] = current_suspend
    feats['consecutive_suspend'] = consecutive_suspend
    
    # 计算最近5天是否有停牌
    recent_5d_suspend = np.zeros(n, dtype=int)
    for i in range(n):
        start = max(0, i - 4)
        has_suspend = np.any(is_trading[start:i+1] != 1)
        recent_5d_suspend[i] = 1 if has_suspend else 0
    feats['recent_5d_suspend'] = recent_5d_suspend

    # 1. 价格动量类
    for period in [1, 3, 5, 10, 20, 60]:
        ret = np.empty(n)
        ret[:period] = np.nan
        ret[period:] = close[period:] / close[:-period] - 1.0
        feats[f'ret_{period}d'] = ret

    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(close[1:] / close[:-1])
    feats['log_ret_1d'] = log_ret

    # 2. 均线系统
    for w in [5, 10, 20, 60]:
        ma = pd.Series(close).rolling(w, min_periods=1).mean().values
        feats[f'ma_{w}'] = ma
        feats[f'ma_bias_{w}'] = (close - ma) / (ma + 1e-10)

    feats['ma_bull'] = (
        (feats['ma_5'] > feats['ma_10']).astype(float) +
        (feats['ma_10'] > feats['ma_20']).astype(float) +
        (feats['ma_20'] > feats['ma_60']).astype(float)
    ) / 3.0

    # 3. MACD
    def ewma(x, span):
        alpha = 2 / (span + 1)
        n = len(x)
        result = np.zeros(n)
        result[0] = x[0]
        for i in range(1, n):
            result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
        return result

    ema12 = ewma(close, 12)
    ema26 = ewma(close, 26)
    dif = ema12 - ema26
    dea = ewma(dif, 9)
    macd_hist = 2.0 * (dif - dea)
    feats['macd_dif'] = dif / (close + 1e-10)  # 归一化
    feats['macd_dea'] = dea / (close + 1e-10)
    feats['macd_hist'] = macd_hist / (close + 1e-10)

    # 4. RSI
    # Wilder's smoothing
    def wilder_ema(x, period):
        alpha = 1.0 / period
        result = np.zeros(len(x))
        result[0] = x[0]
        for i in range(1, len(x)):
            result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
        return result

    _rsi_delta = np.zeros(n)
    _rsi_delta[1:] = close[1:] - close[:-1]
    _rsi_gain = np.where(_rsi_delta > 0, _rsi_delta, 0.0)
    _rsi_loss = np.where(_rsi_delta < 0, -_rsi_delta, 0.0)

    for period in [6, 12, 24]:
        avg_gain = wilder_ema(_rsi_gain, period)
        avg_loss = wilder_ema(_rsi_loss, period)
        rs = avg_gain / (avg_loss + 1e-10)
        feats[f'rsi_{period}'] = 100.0 - 100.0 / (1.0 + rs)

    # 5. 布林带
    bb_period = 20
    bb_ma = pd.Series(close).rolling(bb_period, min_periods=1).mean().values
    bb_std = pd.Series(close).rolling(bb_period, min_periods=2).std().fillna(0).values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    feats['bb_pctb'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    feats['bb_width'] = (bb_upper - bb_lower) / (bb_ma + 1e-10)

    # 6. ATR
    hl = high - low
    tr = np.empty(n)
    tr[0] = hl[0]
    tr[1:] = np.maximum(hl[1:], np.maximum(
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ))
    atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    feats['atr14_ratio'] = atr14 / (close + 1e-10)

    # 7. KDJ
    kdj_period = 9
    low_min = pd.Series(low).rolling(kdj_period, min_periods=1).min().values
    high_max = pd.Series(high).rolling(kdj_period, min_periods=1).max().values
    rsv = (close - low_min) / (high_max - low_min + 1e-10) * 100.0
    k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
    d = pd.Series(k).ewm(com=2, adjust=False).mean().values
    j = 3 * k - 2 * d
    feats['kdj_k'] = k
    feats['kdj_d'] = d
    feats['kdj_j'] = j

    # 8. 成交量特征
    vol_s = pd.Series(volume)
    vol_ma5 = vol_s.rolling(5, min_periods=1).mean().values
    feats['vol_ratio'] = volume / (vol_ma5 + 1e-10)

    _obv_sign = np.sign(np.diff(close, prepend=close[0]))
    obv = np.cumsum(_obv_sign * volume)
    obv_ma = pd.Series(obv).rolling(10, min_periods=1).mean().values
    feats['obv_diff'] = (obv - obv_ma) / (np.abs(obv_ma) + 1e-10)

    turn_s = pd.Series(turn)
    turn_ma5 = turn_s.rolling(5, min_periods=1).mean().values
    feats['turn_ratio'] = turn / (turn_ma5 + 1e-10)
    feats['turn_ma5'] = turn_ma5

    amt_s = pd.Series(amount)
    amt_ma20 = amt_s.rolling(20, min_periods=1).mean().values
    feats['amt_ratio'] = amount / (amt_ma20 + 1e-10)

    # 9. 波动率与风险
    log_ret_s = pd.Series(feats['log_ret_1d'])
    for w in [5, 10, 20]:
        feats[f'volatility_{w}d'] = log_ret_s.rolling(w, min_periods=2).std().values

    pos_ret = log_ret_s.clip(lower=0)
    neg_ret = log_ret_s.clip(upper=0)
    feats['upside_vol_20'] = pos_ret.rolling(20, min_periods=2).std().values
    feats['downside_vol_20'] = neg_ret.rolling(20, min_periods=2).std().values

    close_s20 = pd.Series(close)
    rolling_max = close_s20.rolling(20, min_periods=1).max().values
    feats['drawdown_20d'] = (close - rolling_max) / (rolling_max + 1e-10)

    # 10. 价格位置
    for w in [10, 20, 60]:
        h = pd.Series(high).rolling(w, min_periods=1).max().values
        l = pd.Series(low).rolling(w, min_periods=1).min().values
        feats[f'price_pos_{w}d'] = (close - l) / (h - l + 1e-10)

    # 11. 基本面变化率
    for col_name, col_vals in [('peTTM', g['peTTM'].values),
                                ('pbMRQ', g['pbMRQ'].values),
                                ('psTTM', g['psTTM'].values)]:
        col_s = pd.Series(col_vals.astype(np.float64))
        ma20 = col_s.rolling(20, min_periods=1).mean().values
        feats[f'{col_name}_chg'] = (col_vals - ma20) / (np.abs(ma20) + 1e-10)

    # 12. 额外alpha因子
    ret_5d = feats['ret_5d']
    vol_5d = feats['volatility_5d']
    feats['reversal_5d'] = -ret_5d  # 短期反转
    safe_vol = np.where(np.isnan(vol_5d) | (vol_5d < 1e-10), 1e-10, vol_5d)
    feats['risk_adj_mom_20d'] = feats['ret_20d'] / safe_vol

    amt_5ma = pd.Series(amount).rolling(5, min_periods=1).mean().values
    amt_20ma = pd.Series(amount).rolling(20, min_periods=1).mean().values
    feats['amt_trend'] = amt_5ma / (amt_20ma + 1e-10)

    ret_1d = feats['ret_1d']
    ret_1d_s = pd.Series(ret_1d)
    feats['momentum_acc'] = ret_1d_s.rolling(5, min_periods=2).mean().values - \
                            ret_1d_s.rolling(20, min_periods=5).mean().values

    price_trend = pd.Series(close).pct_change(5).values
    vol_trend = pd.Series(volume).pct_change(5).values
    feats['vol_price_diverge'] = np.where(
        np.isnan(price_trend) | np.isnan(vol_trend), np.nan,
        price_trend - vol_trend
    )

    total_range = high - low + 1e-10
    feats['upper_shadow'] = (high - np.maximum(close, open_)) / total_range
    feats['lower_shadow'] = (np.minimum(close, open_) - low) / total_range

    # 13. 标签：T+1开盘买入 → 持有HOLD_DAYS天 → T+1+HOLD_DAYS开盘卖出
    HOLD_DAYS = CONFIG['features']['HOLD_DAYS']  # 持有交易日数（T+1买入 → T+1+HOLD_DAYS卖出）
    sell_offset = HOLD_DAYS + 1  # 卖出价在 T+1+HOLD_DAYS 位置

    open_next_1 = np.roll(open_, -1)
    open_next_1[-1] = np.nan

    open_next_n = np.roll(open_, -sell_offset)
    open_next_n[-sell_offset:] = np.nan
    if n <= sell_offset:
        open_next_n[:] = np.nan

    buy_cost = CONFIG['features']['BUY_COST']
    sell_cost = CONFIG['features']['SELL_COST']
    raw_label = (open_next_n * sell_cost) / (open_next_1 * buy_cost) - 1.0

    # 验证标签交易日连续性
    if trade_day_idx is not None:
        dates = g['date'].values
        for i in range(n - sell_offset):
            buy_date = dates[i + 1]
            sell_date = dates[i + sell_offset]
            buy_idx = trade_day_idx.get(buy_date)
            sell_idx = trade_day_idx.get(sell_date)
            if buy_idx is None or sell_idx is None or (sell_idx - buy_idx) != HOLD_DAYS:
                raw_label[i] = np.nan

    # Winsorize极端标签
    label_min = CONFIG['features']['LABEL_WINSORIZE_MIN']
    label_max = CONFIG['features']['LABEL_WINSORIZE_MAX']
    raw_label = np.clip(raw_label, label_min, label_max)
    feats['label'] = raw_label

    feat_df = pd.DataFrame(feats, index=g.index)
    return feat_df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """对所有股票批量计算特征"""
    print("开始特征工程...")
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    trade_days = pd.read_csv(TRADE_DAYS_TXT, header=None, names=['date'])
    trade_days['date'] = pd.to_datetime(trade_days['date'])
    trade_day_idx = {d: i for i, d in enumerate(trade_days['date'].values)}

    print("  计算个股技术指标...")

    stock_groups = df.groupby('code')
    feat_dfs = []

    for _, group in stock_groups:
        stock_feats = compute_features_for_stock(group, trade_day_idx=trade_day_idx)
        feat_dfs.append(stock_feats)

    feat_all = pd.concat(feat_dfs)
    # 只取特征列（不在原始 df 中的列）
    new_cols = [c for c in feat_all.columns if c not in df.columns]
    df = df.join(feat_all[new_cols], how='left')

    # 截面特征
    print("  计算市场截面特征...")
    # 市场整体截面特征
    daily = df.groupby('date').agg(
        mkt_ret_mean=('ret_1d', 'mean'),
        mkt_ret_std=('ret_1d', 'std'),
        mkt_advance_ratio=('ret_1d', lambda x: (x > 0).mean()),
    ).reset_index()

    daily = daily.sort_values('date')
    daily['mkt_mom_5d'] = daily['mkt_ret_mean'].rolling(5, min_periods=1).sum()

    df = df.merge(daily, on='date', how='left')

    df['excess_ret_1d'] = df['ret_1d'] - df['mkt_ret_mean']
    
    # 行业截面特征（industry）
    print("  计算行业截面特征...")
    industry_daily = df.groupby(['date', 'industry']).agg(
        industry_ret_mean=('ret_1d', 'mean'),
        industry_ret_std=('ret_1d', 'std'),
        industry_advance_ratio=('ret_1d', lambda x: (x > 0).mean()),
        industry_count=('code', 'count')
    ).reset_index()
    
    # 计算行业动量
    industry_daily = industry_daily.sort_values(['industry', 'date'])
    industry_daily['industry_mom_5d'] = industry_daily.groupby('industry')['industry_ret_mean'].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    
    # 合并行业截面特征
    df = df.merge(industry_daily, on=['date', 'industry'], how='left')
    
    # 计算行业超额收益
    df['excess_ret_industry'] = df['ret_1d'] - df['industry_ret_mean']
    
    # 行业分类截面特征（industryClassification）
    industry_class_daily = df.groupby(['date', 'industryClassification']).agg(
        industry_class_ret_mean=('ret_1d', 'mean'),
        industry_class_ret_std=('ret_1d', 'std'),
        industry_class_advance_ratio=('ret_1d', lambda x: (x > 0).mean()),
        industry_class_count=('code', 'count')
    ).reset_index()
    
    # 计算行业分类动量
    industry_class_daily = industry_class_daily.sort_values(['industryClassification', 'date'])
    industry_class_daily['industry_class_mom_5d'] = industry_class_daily.groupby('industryClassification')['industry_class_ret_mean'].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    
    # 合并行业分类截面特征
    df = df.merge(industry_class_daily, on=['date', 'industryClassification'], how='left')
    
    # 计算行业分类超额收益
    df['excess_ret_industry_class'] = df['ret_1d'] - df['industry_class_ret_mean']

    # 基本面截面排名（0值为缺失值填充，排名前先将0替换为NaN避免扭曲截面）
    print("  计算截面排名...")
    for col in ['peTTM', 'pbMRQ', 'psTTM']:
        df[f'{col}_rank'] = df.groupby('date')[col].transform(
            lambda x: x.replace(0, np.nan).rank(pct=True)
        )

    # 动量截面排名
    for period in [5, 10, 20]:
        df[f'ret_{period}d_rank'] = df.groupby('date')[f'ret_{period}d'].rank(pct=True)

    # 清理前60日 NaN
    print("  清理NaN行...")
    df = df.sort_values(['code', 'date'])
    df['_rank'] = df.groupby('code').cumcount()
    df = df[df['_rank'] >= 60].drop(columns=['_rank'])
    df = df.reset_index(drop=True)

    feat_cols = get_feature_columns(df)
    float_cols = [col for col in feat_cols if df[col].dtype == np.float64]
    if float_cols:
        df[float_cols] = df[float_cols].astype(np.float32)
    gc.collect()

    print(f"\n✅ 特征工程完成! {df.shape[0]:,} 行, {df.shape[1]} 列")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取所有特征列名"""
    exclude = {'code', 'name', 'date', 'open', 'high', 'low', 'close',
               'volume', 'amount', 'turn', 'pctChg',
               'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'label',
               'isST', 'isTrading'}
    return [c for c in df.columns if c not in exclude]


def run_pipeline(end_date=None):
    """执行特征工程流水线"""
    if os.path.exists(FEATURE_PKL):
        try:
            cached = pd.read_pickle(FEATURE_PKL)
            cached_max = cached['date'].max()
            
            if end_date is not None:
                # 指定了截止日期：检查缓存是否已包含该日期
                if pd.Timestamp(end_date) == cached_max:
                    print(f"✅ [缓存命中] features.pkl 已是 {end_date}，跳过重新计算")
                    return cached
            else:
                # 未指定截止日期：检查缓存是否与清洗后数据一致
                if os.path.exists(CLEAN_PKL):
                    clean_df = pd.read_pickle(CLEAN_PKL)
                    clean_max = clean_df['date'].max()
                    if cached_max == clean_max:
                        print(f"✅ [缓存命中] features.pkl 已是最新 ({cached_max.date()})，跳过重新计算")
                        return cached
        except Exception as e:
            print(f"  [缓存读取失败: {e}]，继续重新计算...")
            pass  # 读取失败则继续正常流程

    df = pd.read_pickle(CLEAN_PKL)

    if end_date is not None:
        df = df[df['date'] <= pd.Timestamp(end_date)].copy()
        print(f"  [date filter] 数据截断至 {end_date}")

    df = compute_all_features(df)

    os.makedirs(os.path.dirname(FEATURE_PKL), exist_ok=True)
    df.to_pickle(FEATURE_PKL)
    print(f"保存至 {FEATURE_PKL}")
    print(f"特征列数: {len(get_feature_columns(df))}")
    print(f"特征列: {get_feature_columns(df)}")

    return df


if __name__ == '__main__':
    import sys as _sys
    _end_date = None
    _args = _sys.argv[1:]
    if '--date' in _args:
        _idx = _args.index('--date')
        _end_date = _args[_idx + 1]
    run_pipeline(end_date=_end_date)
