"""
py02_features.py — 特征工程模块
================================
职责：
1. 计算技术指标（MA/EMA/MACD/RSI/Bollinger/ATR/KDJ/OBV等）
2. 动量与价格位置特征
3. 成交量特征
4. 基本面截面排名特征
5. 波动率与风险特征
6. 市场环境特征（截面）
7. 生成标签：T+1日开盘买入 → T+1+HOLD_DAYS日开盘卖出的收益率（T日为决策日）

严格避免未来信息泄露：所有特征仅使用T日及之前数据。
"""

import pandas as pd
import numpy as np
import os
import gc
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PKL = os.path.join(BASE_DIR, 'data', 'mainboard_clean.pkl')
FEATURE_PKL = os.path.join(BASE_DIR, 'data', 'features.pkl')


# ============ 辅助函数 ============

def _ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window, min_periods=1).mean()


# ============ 特征计算函数（全部在groupby内按股票计算） ============

def compute_features_for_stock(g: pd.DataFrame) -> pd.DataFrame:
    """
    对单只股票的DataFrame计算所有特征。
    输入g已按date升序排列。
    """
    close = g['close'].values.astype(np.float64)
    open_ = g['open'].values.astype(np.float64)
    high = g['high'].values.astype(np.float64)
    low = g['low'].values.astype(np.float64)
    volume = g['volume'].values.astype(np.float64)
    amount = g['amount'].values.astype(np.float64)
    turn = g['turn'].values.astype(np.float64)
    n = len(g)

    feats = {}

    # ===== 1. 价格动量类 =====
    for period in [1, 3, 5, 10, 20, 60]:
        ret = np.empty(n)
        ret[:period] = np.nan
        ret[period:] = close[period:] / close[:-period] - 1.0
        feats[f'ret_{period}d'] = ret

    # 对数收益率（1日）
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(close[1:] / close[:-1])
    feats['log_ret_1d'] = log_ret

    # ===== 2. 均线系统 =====
    close_s = pd.Series(close)
    for w in [5, 10, 20, 60]:
        ma = close_s.rolling(w, min_periods=1).mean().values
        feats[f'ma_{w}'] = ma
        feats[f'ma_bias_{w}'] = (close - ma) / (ma + 1e-10)

    # 均线多头排列信号: MA5 > MA10 > MA20 > MA60
    feats['ma_bull'] = (
        (feats['ma_5'] > feats['ma_10']).astype(float) +
        (feats['ma_10'] > feats['ma_20']).astype(float) +
        (feats['ma_20'] > feats['ma_60']).astype(float)
    ) / 3.0

    # ===== 3. MACD =====
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    macd_hist = 2.0 * (dif - dea)
    feats['macd_dif'] = dif / (close + 1e-10)  # 归一化
    feats['macd_dea'] = dea / (close + 1e-10)
    feats['macd_hist'] = macd_hist / (close + 1e-10)

    # ===== 4. RSI =====
    for period in [6, 12, 24]:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        feats[f'rsi_{period}'] = 100.0 - 100.0 / (1.0 + rs)

    # ===== 5. 布林带 =====
    bb_period = 20
    bb_ma = pd.Series(close).rolling(bb_period, min_periods=1).mean().values
    bb_std = pd.Series(close).rolling(bb_period, min_periods=1).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    feats['bb_pctb'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    feats['bb_width'] = (bb_upper - bb_lower) / (bb_ma + 1e-10)

    # ===== 6. ATR =====
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    feats['atr14_ratio'] = atr14 / (close + 1e-10)

    # ===== 7. KDJ =====
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

    # ===== 8. 成交量特征 =====
    vol_s = pd.Series(volume)
    vol_ma5 = vol_s.rolling(5, min_periods=1).mean().values
    feats['vol_ratio'] = volume / (vol_ma5 + 1e-10)

    # OBV
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    obv_ma = pd.Series(obv).rolling(10, min_periods=1).mean().values
    feats['obv_diff'] = (obv - obv_ma) / (np.abs(obv_ma) + 1e-10)

    # 换手率均值和突变
    turn_s = pd.Series(turn)
    turn_ma5 = turn_s.rolling(5, min_periods=1).mean().values
    feats['turn_ratio'] = turn / (turn_ma5 + 1e-10)
    feats['turn_ma5'] = turn_ma5

    # 成交额（归一化）
    amt_s = pd.Series(amount)
    amt_ma20 = amt_s.rolling(20, min_periods=1).mean().values
    feats['amt_ratio'] = amount / (amt_ma20 + 1e-10)

    # ===== 9. 波动率与风险 =====
    log_ret_s = pd.Series(feats['log_ret_1d'])
    for w in [5, 10, 20]:
        feats[f'volatility_{w}d'] = log_ret_s.rolling(w, min_periods=2).std().values

    # 上行/下行波动率
    pos_ret = log_ret_s.clip(lower=0)
    neg_ret = log_ret_s.clip(upper=0)
    feats['upside_vol_20'] = pos_ret.rolling(20, min_periods=2).std().values
    feats['downside_vol_20'] = neg_ret.rolling(20, min_periods=2).std().values

    # 20日最大回撤
    close_s20 = pd.Series(close)
    rolling_max = close_s20.rolling(20, min_periods=1).max().values
    feats['drawdown_20d'] = (close - rolling_max) / (rolling_max + 1e-10)

    # ===== 10. 价格位置 =====
    for w in [10, 20, 60]:
        h = pd.Series(high).rolling(w, min_periods=1).max().values
        l = pd.Series(low).rolling(w, min_periods=1).min().values
        feats[f'price_pos_{w}d'] = (close - l) / (h - l + 1e-10)

    # ===== 11. 基本面变化率 =====
    for col_name, col_vals in [('peTTM', g['peTTM'].values),
                                ('pbMRQ', g['pbMRQ'].values),
                                ('psTTM', g['psTTM'].values)]:
        col_s = pd.Series(col_vals.astype(np.float64))
        ma20 = col_s.rolling(20, min_periods=1).mean().values
        feats[f'{col_name}_chg'] = (col_vals - ma20) / (np.abs(ma20) + 1e-10)

    # ===== 12. 额外alpha因子 =====
    # 反转因子：短期超跌反弹
    if n >= 5:
        ret_5d = feats['ret_5d']
        vol_5d = feats['volatility_5d']
        feats['reversal_5d'] = -ret_5d  # 短期反转
        # 波动率调整动量
        safe_vol = np.where(np.isnan(vol_5d) | (vol_5d < 1e-10), 1e-10, vol_5d)
        feats['risk_adj_mom_20d'] = feats['ret_20d'] / safe_vol

    # 成交额变化趋势
    amt_5ma = pd.Series(amount).rolling(5, min_periods=1).mean().values
    amt_20ma = pd.Series(amount).rolling(20, min_periods=1).mean().values
    feats['amt_trend'] = amt_5ma / (amt_20ma + 1e-10)

    # 价格加速度（动量变化率）
    ret_1d = feats['ret_1d']
    ret_1d_s = pd.Series(ret_1d)
    feats['momentum_acc'] = ret_1d_s.rolling(5, min_periods=2).mean().values - \
                            ret_1d_s.rolling(20, min_periods=5).mean().values

    # 量价背离：价格上涨但成交量萎缩
    price_trend = pd.Series(close).pct_change(5).values
    vol_trend = pd.Series(volume).pct_change(5).values
    feats['vol_price_diverge'] = np.where(
        np.isnan(price_trend) | np.isnan(vol_trend), np.nan,
        price_trend - vol_trend
    )

    # 上影线比例（卖压指标）
    body = np.abs(close - open_)
    total_range = high - low + 1e-10
    feats['upper_shadow'] = (high - np.maximum(close, open_)) / total_range
    feats['lower_shadow'] = (np.minimum(close, open_) - low) / total_range

    # ===== 13. 标签：T+1开盘买入 → 持有HOLD_DAYS天 → T+1+HOLD_DAYS开盘卖出 =====
    HOLD_DAYS = 5  # 持有5个交易日（T+1买入 → T+6卖出）
    sell_offset = HOLD_DAYS + 1  # 卖出价在 T+1+HOLD_DAYS = T+6 位置

    open_next1 = np.empty(n)
    open_next1[:-1] = open_[1:]
    open_next1[-1] = np.nan

    open_next_n = np.empty(n)
    if n > sell_offset:
        open_next_n[:n-sell_offset] = open_[sell_offset:]
        open_next_n[n-sell_offset:] = np.nan
    else:
        open_next_n[:] = np.nan

    # 扣除交易成本: 买入手续费0.0085%, 卖出手续费0.0085%+印花税0.05%
    buy_cost = 1.0 + 0.000085
    sell_cost = 1.0 - 0.000585
    raw_label = (open_next_n * sell_cost) / (open_next1 * buy_cost) - 1.0

    # Winsorize极端标签值（截断到±30%）
    # 主板5日复利极端收益约50%，±30%保留绝大多数信号同时抑制极端噪声。
    # 原±10%过于保守，会压制高动量标的的训练信号。
    raw_label = np.clip(raw_label, -0.30, 0.30)
    feats['label'] = raw_label

    # 将所有特征转为DataFrame
    feat_df = pd.DataFrame(feats, index=g.index)
    return feat_df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """对所有股票批量计算特征"""
    print("开始特征工程...")
    df = df.sort_values(['代码', 'date']).reset_index(drop=True)

    # 分组计算（使用高效的apply）
    print("  计算个股技术指标...")

    # 分批处理避免内存问题
    stocks = df['代码'].unique()
    n_stocks = len(stocks)
    batch_size = 100
    feat_cols_initialized = False

    for i in range(0, n_stocks, batch_size):
        batch_stocks = stocks[i:i+batch_size]
        batch_df = df[df['代码'].isin(batch_stocks)]
        batch_feats = batch_df.groupby('代码', group_keys=False).apply(
            compute_features_for_stock
        )

        # 首批时初始化特征列
        if not feat_cols_initialized:
            for col in batch_feats.columns:
                df[col] = np.nan
            feat_cols_initialized = True

        # 直接写入原 DataFrame，避免 concat 双倍内存
        df.loc[batch_feats.index, batch_feats.columns] = batch_feats

        del batch_df, batch_feats
        gc.collect()
        print(f"  进度: {min(i+batch_size, n_stocks)}/{n_stocks} 只股票")

    # ===== 截面特征（市场环境） =====
    print("  计算截面特征...")
    daily = df.groupby('date').agg(
        mkt_ret_mean=('ret_1d', 'mean'),
        mkt_ret_std=('ret_1d', 'std'),
        mkt_advance_ratio=('ret_1d', lambda x: (x > 0).mean()),
    ).reset_index()

    # 市场动量（5日滚动）
    daily = daily.sort_values('date')
    daily['mkt_mom_5d'] = daily['mkt_ret_mean'].rolling(5, min_periods=1).sum()

    df = df.merge(daily, on='date', how='left')

    # 个股相对市场超额收益
    df['excess_ret_1d'] = df['ret_1d'] - df['mkt_ret_mean']

    # ===== 基本面截面排名（百分位） =====
    print("  计算截面排名...")
    for col in ['peTTM', 'pbMRQ', 'psTTM']:
        df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)

    # ===== 动量截面排名 =====
    for period in [5, 10, 20]:
        df[f'ret_{period}d_rank'] = df.groupby('date')[f'ret_{period}d'].rank(pct=True)

    # 清理：删除特征计算初期的NaN行（前60日）
    print("  清理NaN行...")
    df = df.sort_values(['代码', 'date'])
    df['_rank'] = df.groupby('代码').cumcount()
    df = df[df['_rank'] >= 60].drop(columns=['_rank'])
    df = df.reset_index(drop=True)

    # 特征列转 float32 节省内存
    feat_cols = get_feature_columns(df)
    for col in feat_cols:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    gc.collect()

    print(f"✅ 特征工程完成! {df.shape[0]:,} 行, {df.shape[1]} 列")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取所有特征列名（排除标识列、标签列等）"""
    exclude = {'代码', '名称', 'date', 'open', 'high', 'low', 'close',
               'volume', 'amount', 'turn', 'pctChg',
               'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'label'}
    return [c for c in df.columns if c not in exclude]


def run_pipeline(end_date=None):
    """执行特征工程流水线

    Args:
        end_date: 数据截止日期（含），格式 'YYYY-MM-DD'。None 表示使用全部数据。
    """
    df = pd.read_pickle(CLEAN_PKL)

    # 截断到指定日期（在特征计算前截断，确保不使用未来数据）
    if end_date is not None:
        df = df[df['date'] <= pd.Timestamp(end_date)].copy()
        print(f"  [date filter] 数据截断至 {end_date}")

    df = compute_all_features(df)

    # 保存
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
