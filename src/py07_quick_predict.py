"""
py03_quick_predict.py — 每日快速预测模块
=========================================
职责：
1. 加载特征数据，使用最近3年数据训练一个Ensemble模型
2. 对最新交易日做预测
3. 输出预测结果供py05使用

与py03_model.py的区别：
- py03 执行完整的walk-forward滚动训练+回测（耗时长，用于验证策略）
- 本脚本只训练最新模型窗口，做当日预测（几分钟搞定，用于每日决策）
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PKL = os.path.join(BASE_DIR, 'data', 'features.pkl')
PREDICT_PKL = os.path.join(BASE_DIR, 'data', 'predictions.pkl')

N_ENSEMBLE = 5
TRAIN_YEARS = 3

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 7,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 100,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'verbose': -1,
    'n_jobs': -1,
}


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {'代码', '名称', 'date', 'open', 'high', 'low', 'close',
               'volume', 'amount', 'turn', 'pctChg',
               'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'label',
               'ma_5', 'ma_10', 'ma_20', 'ma_60'}
    return [c for c in df.columns if c not in exclude]


def quick_predict():
    """快速训练最新模型窗口并预测"""
    print("加载特征数据...")
    df = pd.read_pickle(FEATURE_PKL)
    feature_cols = get_feature_columns(df)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    all_dates = sorted(df['date'].unique())
    latest_date = all_dates[-1]
    print(f"最新数据日期: {latest_date.date()}")
    print(f"使用 {len(feature_cols)} 个特征")

    # 训练数据：最近3年（排除最新一天）
    train_window = TRAIN_YEARS * 252
    train_start_idx = max(0, len(all_dates) - 1 - train_window)
    train_dates = all_dates[train_start_idx:-1]  # 排除最后一天

    # 划分训练集和验证集
    val_split = int(len(train_dates) * 0.9)
    train_date_set = set(train_dates[:val_split])
    val_date_set = set(train_dates[val_split:])

    train_mask = df['date'].isin(train_date_set)
    val_mask = df['date'].isin(val_date_set)

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'label']
    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, 'label']

    # 去掉标签NaN
    valid_train = y_train.notna()
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    valid_val = y_val.notna()
    X_val, y_val = X_val[valid_val], y_val[valid_val]

    print(f"训练集: {len(X_train):,} 行, 验证集: {len(X_val):,} 行")

    # 训练 Ensemble
    print("训练Ensemble模型...")
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
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        models.append(model)
        print(f"  模型 {seed+1}/{N_ENSEMBLE} 完成, best_iter={model.best_iteration}")

    # 预测最新一天
    day_mask = df['date'] == latest_date
    day_data = df.loc[day_mask]
    X_pred = day_data[feature_cols]

    preds = np.array([m.predict(X_pred) for m in models])
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    confidence = 1.0 / (1.0 + pred_std * 100)

    # 构建预测结果（兼容py05的格式）
    pred_records = []
    for j, idx in enumerate(day_data.index):
        pred_records.append({
            'idx': idx,
            'date': latest_date,
            '代码': day_data.iloc[j]['代码'],
            'pred_return': pred_mean[j],
            'pred_std': pred_std[j],
            'confidence': confidence[j],
        })

    pred_df = pd.DataFrame(pred_records)

    # 如果已有历史预测，合并（保留历史用于回测参考）
    if os.path.exists(PREDICT_PKL):
        old_pred = pd.read_pickle(PREDICT_PKL)
        # 移除旧的最新日期预测（如果有），替换为新的
        old_pred = old_pred[old_pred['date'] != latest_date]
        pred_df = pd.concat([old_pred, pred_df], ignore_index=True)

    pred_df.to_pickle(PREDICT_PKL)

    n_positive = (pred_df[pred_df['date'] == latest_date]['pred_return'] > 0).sum()
    n_total = len(pred_df[pred_df['date'] == latest_date])
    print(f"\n预测完成! {latest_date.date()}: {n_total} 只股票, {n_positive} 只预测正收益")
    print(f"保存至 {PREDICT_PKL}")


if __name__ == '__main__':
    quick_predict()
