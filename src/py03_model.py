"""
py03_model.py — LightGBM模型模块
==================================
职责：
1. Walk-forward滚动训练LightGBM回归模型
2. Ensemble多模型（不同随机种子）估计不确定性
3. 输出每日每只股票的预测收益率和置信度
4. 保存预测结果供回测使用
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
import gc

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PKL = os.path.join(BASE_DIR, 'data', 'features.pkl')
PREDICT_PKL = os.path.join(BASE_DIR, 'data', 'predictions.pkl')

# ============ 配置 ============
N_ENSEMBLE = 5          # Ensemble模型数量（不同随机种子）
TRAIN_YEARS = 3         # 训练窗口年数
RETRAIN_DAYS = 22       # 每22个交易日（约1个月）重新训练
BACKTEST_START = '2023-01-01'  # 回测起始日期

LGB_PARAMS = {
    'objective': 'huber',          # Huber损失，对异常值更鲁棒
    'alpha': 0.9,                  # Huber delta参数
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,              # 降低复杂度防过拟合 (63→31)
    'max_depth': 5,                # 限制树深度 (7→5)
    'learning_rate': 0.03,         # 更小学习率，更稳定 (0.05→0.03)
    'feature_fraction': 0.6,       # 更强随机性 (0.7→0.6)
    'bagging_fraction': 0.7,       # 更强随机性 (0.8→0.7)
    'bagging_freq': 5,
    'min_child_samples': 200,      # 更保守的叶节点 (100→200)
    'lambda_l1': 1.0,              # 更强L1正则 (0.1→1.0)
    'lambda_l2': 5.0,              # 更强L2正则 (1.0→5.0)
    'min_gain_to_split': 0.01,     # 分裂最小增益，防止无意义分裂
    'verbose': -1,
    'n_jobs': -1,
}


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取特征列"""
    exclude = {'代码', '名称', 'date', 'open', 'high', 'low', 'close',
               'volume', 'amount', 'turn', 'pctChg',
               'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'label',
               'ma_5', 'ma_10', 'ma_20', 'ma_60'}
    return [c for c in df.columns if c not in exclude]


def train_and_predict(df: pd.DataFrame, end_date=None) -> pd.DataFrame:
    """
    Walk-forward滚动训练与预测
    返回包含预测结果的DataFrame

    Args:
        df: 特征数据（已截断到 end_date）
        end_date: 若指定，则只预测该日（单日快速模式）；否则从 BACKTEST_START 全量预测
    """
    feature_cols = get_feature_columns(df)
    print(f"使用 {len(feature_cols)} 个特征: {feature_cols[:10]}...")

    # 获取所有交易日
    all_dates = sorted(df['date'].unique())

    if end_date is not None:
        # 单日模式：只预测 end_date 当天，用之前 TRAIN_YEARS 年数据训练
        target_date = pd.Timestamp(end_date)
        if target_date not in all_dates:
            # 取 <= end_date 的最新交易日
            candidates = [d for d in all_dates if d <= target_date]
            if not candidates:
                raise ValueError(f"end_date {end_date} 之前没有可用数据")
            target_date = candidates[-1]
            print(f"  [end_date 非交易日，取最近交易日 {target_date.date()}]")

        target_idx = all_dates.index(target_date)
        train_window = TRAIN_YEARS * 252
        train_start_idx = max(0, target_idx - train_window)
        train_dates = all_dates[train_start_idx:target_idx]

        if len(train_dates) < 252:
            raise ValueError(f"训练数据不足（{len(train_dates)} 天 < 252 天），请使用更早的 BACKTEST_START 或更长的历史数据")

        val_split = int(len(train_dates) * 0.9)
        train_date_set = set(train_dates[:val_split])
        val_date_set = set(train_dates[val_split:])

        train_mask = df['date'].isin(train_date_set)
        val_mask = df['date'].isin(val_date_set)
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, 'label']
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, 'label']

        valid_train = y_train.notna()
        X_train, y_train = X_train[valid_train], y_train[valid_train]
        valid_val = y_val.notna()
        X_val, y_val = X_val[valid_val], y_val[valid_val]

        print(f"单日预测模式: 目标日 {target_date.date()}")
        print(f"训练集: {len(X_train):,} 行, 验证集: {len(X_val):,} 行")

        models = []
        for seed in range(N_ENSEMBLE):
            params = LGB_PARAMS.copy()
            params['seed'] = seed * 42
            params['feature_fraction_seed'] = seed * 42
            params['bagging_seed'] = seed * 42
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            model = lgb.train(
                params, dtrain, num_boost_round=800,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            models.append(model)
            print(f"  模型 {seed+1}/{N_ENSEMBLE} 完成, best_iter={model.best_iteration}")

        del X_train, y_train, X_val, y_val
        gc.collect()

        day_mask = df['date'] == target_date
        day_data = df.loc[day_mask]
        X_pred = day_data[feature_cols]
        preds = np.array([m.predict(X_pred) for m in models])
        pred_mean = preds.mean(axis=0)
        pred_std = preds.std(axis=0)
        confidence = 1.0 / (1.0 + pred_std * 100)

        pred_records = []
        for j, idx in enumerate(day_data.index):
            pred_records.append({
                'idx': idx,
                'date': target_date,
                '代码': day_data.iloc[j]['代码'],
                'pred_return': pred_mean[j],
                'pred_std': pred_std[j],
                'confidence': confidence[j],
            })

        print(f"\n预测完成! 共 {len(pred_records):,} 条记录")
        return pd.DataFrame(pred_records)

    # 全量 walk-forward 模式
    bt_start = pd.Timestamp(BACKTEST_START)

    # 找到回测起始日期在all_dates中的位置
    bt_start_idx = 0
    for i, d in enumerate(all_dates):
        if d >= bt_start:
            bt_start_idx = i
            break

    # 训练窗口长度（约3年）
    train_window = TRAIN_YEARS * 252

    print(f"回测起始: {all_dates[bt_start_idx].date()}")
    print(f"总交易日: {len(all_dates) - bt_start_idx} 天")

    # 存储预测结果
    pred_records = []

    # 当前模型（ensemble）
    models = None
    last_train_idx = -999  # 上次训练的日期索引

    for day_idx in range(bt_start_idx, len(all_dates)):
        current_date = all_dates[day_idx]

        # 判断是否需要重新训练
        if day_idx - last_train_idx >= RETRAIN_DAYS or models is None:
            # 训练数据：当前日期之前的train_window个交易日
            train_start_idx = max(0, day_idx - train_window)
            train_dates = all_dates[train_start_idx:day_idx]

            if len(train_dates) < 252:  # 至少1年数据才训练
                continue

            # 划分训练集和验证集（最后10%作为验证）
            val_split = int(len(train_dates) * 0.9)
            train_date_set = set(train_dates[:val_split])
            val_date_set = set(train_dates[val_split:])

            train_mask = df['date'].isin(train_date_set)
            val_mask = df['date'].isin(val_date_set)

            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, 'label']
            X_val = df.loc[val_mask, feature_cols]
            y_val = df.loc[val_mask, 'label']

            # 去掉标签为NaN的行
            valid_train = y_train.notna()
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]
            valid_val = y_val.notna()
            X_val = X_val[valid_val]
            y_val = y_val[valid_val]

            # 训练Ensemble
            models = []
            for seed in range(N_ENSEMBLE):
                params = LGB_PARAMS.copy()
                params['seed'] = seed * 42
                params['feature_fraction_seed'] = seed * 42
                params['bagging_seed'] = seed * 42

                dtrain = lgb.Dataset(X_train, label=y_train)
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=800,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                models.append(model)

            last_train_idx = day_idx
            if day_idx % (RETRAIN_DAYS * 3) == bt_start_idx % (RETRAIN_DAYS * 3):
                print(f"  [{current_date.date()}] 模型重训练完成, "
                      f"best_iter={[m.best_iteration for m in models]}")

            # 清理大对象
            del X_train, y_train, X_val, y_val
            gc.collect()

        # 预测当天所有股票
        day_mask = df['date'] == current_date
        day_data = df.loc[day_mask]

        if len(day_data) == 0:
            continue

        X_pred = day_data[feature_cols]

        # Ensemble预测
        preds = np.array([m.predict(X_pred) for m in models])
        pred_mean = preds.mean(axis=0)
        pred_std = preds.std(axis=0)

        # 置信度：std越小置信度越高（归一化到0-1）
        # 用1/(1+std*100)映射，std接近0时置信度接近1
        confidence = 1.0 / (1.0 + pred_std * 100)

        # 保存预测结果
        for j, idx in enumerate(day_data.index):
            pred_records.append({
                'idx': idx,
                'date': current_date,
                '代码': day_data.iloc[j]['代码'],
                'pred_return': pred_mean[j],
                'pred_std': pred_std[j],
                'confidence': confidence[j],
            })

    print(f"\n预测完成! 共 {len(pred_records):,} 条记录")

    # 转为DataFrame
    pred_df = pd.DataFrame(pred_records)
    return pred_df


def run_pipeline(end_date=None):
    """执行模型训练与预测流水线

    Args:
        end_date: 预测截止日期（含），格式 'YYYY-MM-DD'。
                  None 表示从 BACKTEST_START 到最新日期全量 walk-forward。
                  指定日期时只预测该日（用该日之前 TRAIN_YEARS 年数据训练）。
    """
    print("加载特征数据...")
    df = pd.read_pickle(FEATURE_PKL)
    print(f"数据: {df.shape[0]:,} 行, {df['代码'].nunique()} 只股票")

    # 替换inf为NaN再填0
    feature_cols = get_feature_columns(df)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 截断到指定日期（features.pkl 可能已截断，这里做双重保证）
    if end_date is not None:
        df = df[df['date'] <= pd.Timestamp(end_date)].copy()
        print(f"  [date filter] 数据截断至 {end_date}")

    pred_df = train_and_predict(df, end_date=end_date)

    # 保存预测结果
    pred_df.to_pickle(PREDICT_PKL)
    print(f"保存至 {PREDICT_PKL}")

    # 打印统计
    print(f"\n预测统计:")
    print(f"  日期范围: {pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
    print(f"  股票数: {pred_df['代码'].nunique()}")
    print(f"  预测收益率均值: {pred_df['pred_return'].mean():.6f}")
    print(f"  预测收益率标准差: {pred_df['pred_return'].std():.6f}")

    return pred_df


if __name__ == '__main__':
    import sys as _sys
    _end_date = None
    _args = _sys.argv[1:]
    if '--date' in _args:
        _idx = _args.index('--date')
        _end_date = _args[_idx + 1]
    run_pipeline(end_date=_end_date)
