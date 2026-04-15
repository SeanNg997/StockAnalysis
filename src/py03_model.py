"""LightGBM Ensemble 滚动训练与预测"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
import gc
from concurrent.futures import ThreadPoolExecutor

from config import CONFIG

warnings.filterwarnings('ignore')

BASE_DIR = CONFIG['paths']['BASE_DIR']
FEATURE_PKL = CONFIG['paths']['FEATURE_PKL']
PREDICT_PKL = CONFIG['paths']['PREDICT_PKL']

N_ENSEMBLE = CONFIG['model']['N_ENSEMBLE']
TRAIN_YEARS = CONFIG['model']['TRAIN_YEARS']
RETRAIN_DAYS = CONFIG['model']['RETRAIN_DAYS']
BACKTEST_YEARS = CONFIG['model']['BACKTEST_YEARS']
HOLD_DAYS = CONFIG['model']['HOLD_DAYS']
# 训练集末尾 HOLD_DAYS 条 label 依赖未来价格，必须剔除

LGB_PARAMS = CONFIG['model']['LGB_PARAMS'].copy()
LGB_PARAMS['n_jobs'] = max(1, (os.cpu_count() or 4) // N_ENSEMBLE)


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取特征列（排除标识列和标签列）"""
    exclude = {'code', 'name', 'date', 'open', 'high', 'low', 'close',
               'volume', 'amount', 'turn', 'pctChg',
               'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'label',
               'isST', 'isTrading',
               'ma_5', 'ma_10', 'ma_20', 'ma_60'}
    return [c for c in df.columns if c not in exclude]


# 退化检测阈值：原始label尺度下，预测值std低于此值视为退化
_DEGENERATE_STD_THRESH = 0.0002
# 标准化空间下的退化阈值（标准化后y_std≈1，std<0.05说明模型几乎没学到东西）
_DEGENERATE_STD_THRESH_NORM = 0.05


def _is_degenerate(preds: np.ndarray, normalized: bool = False) -> bool:
    """检测模型输出是否退化为近似常数"""
    thresh = _DEGENERATE_STD_THRESH_NORM if normalized else _DEGENERATE_STD_THRESH
    return float(np.std(preds)) < thresh


def _train_single_model(seed, params, X_train, y_train, X_val, y_val,
                        retry_on_degenerate: bool = True):
    """训练单个LightGBM模型（用于并行执行）"""
    _PRIME_SEEDS = [7, 13, 31, 97, 127, 211, 307, 401, 503, 607]

    def _build_and_train(p, round_override=None):
        p = p.copy()
        num_rounds = round_override or p.pop('num_boost_round', 800)
        p['seed'] = _PRIME_SEEDS[seed % len(_PRIME_SEEDS)]
        p['feature_fraction_seed'] = _PRIME_SEEDS[(seed + 2) % len(_PRIME_SEEDS)]
        p['bagging_seed'] = _PRIME_SEEDS[(seed + 4) % len(_PRIME_SEEDS)]
        dtrain = lgb.Dataset(X_train, label=y_train)
        # y_train 已标准化（均值0, std≈1），y_val 同步用训练集均值/std变换保持量级一致
        # 注意：y_val 传入时已是标准化后的值（外部调用前已做 (y_val - y_mean) / y_std）
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        return lgb.train(
            p, dtrain, num_boost_round=num_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

    model = _build_and_train(params)

    # 退化检测：在标准化空间下判断（y已标准化，正常预测std应在0.05以上）
    val_preds = model.predict(X_val)
    best_iter = model.best_iteration if hasattr(model, 'best_iteration') else params.get('num_boost_round', 800)
    if retry_on_degenerate and (
        _is_degenerate(val_preds, normalized=True) or best_iter < 20
    ):
        # early stopping 触发太早（可能因验证集和训练集有分布偏移），改用固定轮数 + 宽松正则
        relaxed = params.copy()
        relaxed['lambda_l1'] = 0.1
        relaxed['lambda_l2'] = 0.1
        relaxed['min_gain_to_split'] = 0.0
        # learning_rate提升3倍，fixed_rounds按比例缩减，等效训练量保持一致
        lr_scale = 3
        relaxed['learning_rate'] = min(params.get('learning_rate', 0.01) * lr_scale, 0.05)
        relaxed['min_child_samples'] = max(params.get('min_child_samples', 200) // 4, 20)
        # 按 lr 缩放比例折算等效轮数（lr提升3x → rounds缩为1/3），但最少300轮
        fixed_rounds = max(params.get('num_boost_round', 3000) // lr_scale, 300)
        p_fixed = relaxed.copy()
        p_fixed.pop('num_boost_round', None)
        dtrain = lgb.Dataset(X_train, label=y_train)
        p_fixed['seed'] = _PRIME_SEEDS[seed % len(_PRIME_SEEDS)]
        p_fixed['feature_fraction_seed'] = _PRIME_SEEDS[(seed + 2) % len(_PRIME_SEEDS)]
        p_fixed['bagging_seed'] = _PRIME_SEEDS[(seed + 4) % len(_PRIME_SEEDS)]
        model2 = lgb.train(p_fixed, dtrain, num_boost_round=fixed_rounds)
        if not _is_degenerate(model2.predict(X_val), normalized=True):
            model = model2
            # 为固定轮数训练的模型添加 best_iteration 属性
            if not hasattr(model, 'best_iteration'):
                model.best_iteration = fixed_rounds

    return seed, model


def _retrain_degenerate_dates(df, feature_cols, all_dates, degenerate_dates,
                               train_window, existing_records):
    """对退化日期用宽松参数重新训练并补充预测"""
    relaxed_params = LGB_PARAMS.copy()
    relaxed_params['lambda_l1'] = 0.1
    relaxed_params['lambda_l2'] = 0.1
    relaxed_params['min_gain_to_split'] = 0.0
    relaxed_params['learning_rate'] = 0.03
    relaxed_params['min_child_samples'] = 50

    sorted_degen = sorted(degenerate_dates)
    new_records = []
    last_retrain_date = None
    retrain_models = None

    for current_date in sorted_degen:
        day_idx = all_dates.index(current_date)
        train_start_idx = max(0, day_idx - train_window)
        safe_end_idx = max(0, day_idx - HOLD_DAYS)
        train_dates = all_dates[train_start_idx:safe_end_idx]

        if len(train_dates) < 252:
            continue

        # 每隔 RETRAIN_DAYS 重训一次，或首次
        need_retrain = (
            retrain_models is None or
            last_retrain_date is None or
            (current_date - last_retrain_date).days > RETRAIN_DAYS * 2
        )

        if need_retrain:
            val_split = int(len(train_dates) * 0.9)
            val_start = min(val_split + HOLD_DAYS, len(train_dates))
            train_date_set = set(train_dates[:val_split])
            val_date_set = set(train_dates[val_start:])
            if len(val_date_set) < 10:
                val_date_set = set(train_dates[val_split:])

            train_mask = df['date'].isin(train_date_set)
            val_mask = df['date'].isin(val_date_set)
            X_tr = df.loc[train_mask, feature_cols]
            y_tr = df.loc[train_mask, 'label']
            X_v = df.loc[val_mask, feature_cols]
            y_v = df.loc[val_mask, 'label']

            valid_tr = y_tr.notna()
            X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
            valid_v = y_v.notna()
            X_v, y_v = X_v[valid_v], y_v[valid_v]

            # 截面标准化
            rt_mean = float(y_tr.mean())
            rt_std = float(y_tr.std()) + 1e-8
            X_tr_arr = X_tr.values
            y_tr_arr = ((y_tr - rt_mean) / rt_std).values
            retrain_label_mean = rt_mean
            retrain_label_std = rt_std

            # 固定轮数并行训练（不用 early stopping，避免验证集分布偏移导致过早停止）
            p_fixed = relaxed_params.copy()
            p_fixed.pop('num_boost_round', None)
            _PRIME_SEEDS = [7, 13, 31, 97, 127, 211, 307, 401, 503, 607]

            def _train_fixed(s):
                p_s = p_fixed.copy()
                p_s['seed'] = _PRIME_SEEDS[s % len(_PRIME_SEEDS)]
                p_s['feature_fraction_seed'] = _PRIME_SEEDS[(s + 2) % len(_PRIME_SEEDS)]
                p_s['bagging_seed'] = _PRIME_SEEDS[(s + 4) % len(_PRIME_SEEDS)]
                # 每个线程独立创建 Dataset，避免多线程共享 LightGBM 对象
                d = lgb.Dataset(X_tr_arr, label=y_tr_arr, free_raw_data=False)
                m = lgb.train(p_s, d, num_boost_round=500)
                # 固定轮数训练没有 best_iteration，手动添加
                if not hasattr(m, 'best_iteration'):
                    m.best_iteration = 500
                return s, m

            with ThreadPoolExecutor(max_workers=N_ENSEMBLE) as executor:
                futures = [executor.submit(_train_fixed, s) for s in range(N_ENSEMBLE)]
                models_dict = {s: m for s, m in (f.result() for f in futures)}
            retrain_models = [models_dict[s] for s in range(N_ENSEMBLE)]
            last_retrain_date = current_date
            print(f"  [{current_date.date()}] 退化补训完成")

        day_mask = df['date'] == current_date
        day_data = df.loc[day_mask]
        if len(day_data) == 0:
            continue

        X_pred = day_data[feature_cols]
        preds = np.array([m.predict(X_pred) for m in retrain_models])
        pred_mean = preds.mean(axis=0) * retrain_label_std + retrain_label_mean
        pred_std = preds.std(axis=0) * retrain_label_std

        if _is_degenerate(pred_mean):
            print(f"  [{current_date.date()}] 警告：宽松参数重训后仍退化，跳过")
            continue

        confidence = 1.0 / (1.0 + pred_std * 100)
        for j, idx in enumerate(day_data.index):
            new_records.append({
                'idx': idx,
                'date': current_date,
                'code': day_data.iloc[j]['code'],
                'pred_return': pred_mean[j],
                'pred_std': pred_std[j],
                'confidence': confidence[j],
            })

    print(f"  退化补训完成，补充 {len(new_records):,} 条记录")
    return existing_records + new_records


def train_and_predict(df: pd.DataFrame, end_date=None) -> pd.DataFrame:
    """Walk-forward滚动训练与预测"""
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
        # 剔除训练集末尾label含未来数据的部分
        safe_end_idx = max(0, target_idx - HOLD_DAYS)
        train_dates = all_dates[train_start_idx:safe_end_idx]

        if len(train_dates) < 252:
            raise ValueError(f"训练数据不足（{len(train_dates)} 天 < 252 天），请使用更早的 BACKTEST_START 或更长的历史数据")

        val_split = int(len(train_dates) * 0.9)
        # purge gap 防止验证集 label 依赖训练集末尾未来价格
        val_start = min(val_split + HOLD_DAYS, len(train_dates))
        train_date_set = set(train_dates[:val_split])
        val_date_set = set(train_dates[val_start:])
        # 若 purge gap 过大导致验证集过小，回退到无 gap 的最后10%
        if len(val_date_set) < 10:
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

        # 截面标准化 label
        y_mean = float(y_train.mean())
        y_std = float(y_train.std()) + 1e-8
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std

        # 转为 numpy array 供多线程共享
        X_train_arr = X_train.values
        y_train_arr = y_train_norm.values
        X_val_arr = X_val.values
        y_val_arr = y_val_norm.values
        del X_train, y_train, X_val, y_val

        # 并行训练Ensemble模型
        models_dict = {}
        with ThreadPoolExecutor(max_workers=N_ENSEMBLE) as executor:
            futures = [
                executor.submit(_train_single_model, seed, LGB_PARAMS,
                               X_train_arr, y_train_arr, X_val_arr, y_val_arr)
                for seed in range(N_ENSEMBLE)
            ]
            for future in futures:
                seed, model = future.result()
                models_dict[seed] = model
                print(f"  模型 {seed+1}/{N_ENSEMBLE} 完成, best_iter={model.best_iteration}")

        models = [models_dict[seed] for seed in range(N_ENSEMBLE)]
        gc.collect()

        day_mask = df['date'] == target_date
        day_data = df.loc[day_mask]
        X_pred = day_data[feature_cols]
        preds = np.array([m.predict(X_pred) for m in models])
        pred_mean = preds.mean(axis=0) * y_std + y_mean  # 逆变换
        pred_std = preds.std(axis=0) * y_std
        confidence = 1.0 / (1.0 + pred_std * 100)

        pred_records = []
        for j, idx in enumerate(day_data.index):
            pred_records.append({
                'idx': idx,
                'date': target_date,
                'code': day_data.iloc[j]['code'],
                'pred_return': pred_mean[j],
                'pred_std': pred_std[j],
                'confidence': confidence[j],
            })

        print(f"\n预测完成! 共 {len(pred_records):,} 条记录")
        return pd.DataFrame(pred_records)

    # 全量 walk-forward 模式
    bt_start = max(all_dates) - pd.DateOffset(years=BACKTEST_YEARS)

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
    degenerate_dates = set()  # 退化日期，循环后补训

    # 当前模型（ensemble）
    models = None
    last_train_idx = -999  # 上次训练的日期索引

    for day_idx in range(bt_start_idx, len(all_dates)):
        current_date = all_dates[day_idx]

        # 判断是否需要重新训练
        if day_idx - last_train_idx >= RETRAIN_DAYS or models is None:
            # 训练数据：当前日期之前的train_window个交易日
            train_start_idx = max(0, day_idx - train_window)
            # 剔除训练集末尾label含未来数据的部分
            safe_end_idx = max(0, day_idx - HOLD_DAYS)
            train_dates = all_dates[train_start_idx:safe_end_idx]

            if len(train_dates) < 252:  # 至少1年数据才训练
                continue

            # 划分训练集和验证集（最后10%作为验证，中间加 purge gap）
            val_split = int(len(train_dates) * 0.9)
            val_start = min(val_split + HOLD_DAYS, len(train_dates))
            train_date_set = set(train_dates[:val_split])
            val_date_set = set(train_dates[val_start:])
            # 若 purge gap 过大导致验证集过小，回退到无 gap 的最后10%
            if len(val_date_set) < 10:
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

            # 截面标准化 label：消除市场整体涨跌偏移，保留个股相对强弱信号
            # 使用训练集的均值/std做标准化（val集同步变换）
            y_mean = float(y_train.mean())
            y_std = float(y_train.std()) + 1e-8
            y_train_norm = (y_train - y_mean) / y_std
            y_val_norm = (y_val - y_mean) / y_std

            # 转为 numpy array 供多线程共享
            X_train_arr = X_train.values
            y_train_arr = y_train_norm.values
            X_val_arr = X_val.values
            y_val_arr = y_val_norm.values
            # 保存逆变换参数（预测结果需还原到原始尺度）
            label_mean = y_mean
            label_std = y_std

            # 并行训练Ensemble模型
            models_dict = {}
            with ThreadPoolExecutor(max_workers=N_ENSEMBLE) as executor:
                futures = [
                    executor.submit(_train_single_model, seed, LGB_PARAMS,
                                   X_train_arr, y_train_arr, X_val_arr, y_val_arr)
                    for seed in range(N_ENSEMBLE)
                ]
                for future in futures:
                    seed, model = future.result()
                    models_dict[seed] = model

            models = [models_dict[seed] for seed in range(N_ENSEMBLE)]

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

        # Ensemble预测（标准化空间）
        preds = np.array([m.predict(X_pred) for m in models])
        pred_mean_norm = preds.mean(axis=0)
        pred_std_norm = preds.std(axis=0)

        # 还原到原始 label 尺度
        pred_mean = pred_mean_norm * label_std + label_mean
        pred_std = pred_std_norm * label_std

        # 退化检测：若ensemble均值std极低，跳过本日（标记为需重训）
        if _is_degenerate(pred_mean):
            degenerate_dates.add(current_date)
            continue

        confidence = 1.0 / (1.0 + pred_std * 100)

        # 保存预测结果
        for j, idx in enumerate(day_data.index):
            pred_records.append({
                'idx': idx,
                'date': current_date,
                'code': day_data.iloc[j]['code'],
                'pred_return': pred_mean[j],
                'pred_std': pred_std[j],
                'confidence': confidence[j],
            })

    print(f"\n预测完成! 共 {len(pred_records):,} 条记录")

    # 退化日期补训：对跳过的退化日期用宽松参数重新预测
    if degenerate_dates:
        print(f"\n检测到 {len(degenerate_dates)} 个退化日期，使用宽松参数补训...")
        pred_records = _retrain_degenerate_dates(
            df, feature_cols, all_dates, degenerate_dates,
            train_window, pred_records
        )

    # 转为DataFrame
    pred_df = pd.DataFrame(pred_records)
    return pred_df


def run_pipeline(end_date=None):
    """执行模型训练与预测流水线"""
    print("加载特征数据...")
    df = pd.read_pickle(FEATURE_PKL)
    print(f"数据: {df.shape[0]:,} 行, {df['code'].nunique()} 只股票")

    # 缓存命中检查
    if end_date is None and os.path.exists(PREDICT_PKL):
        try:
            feat_max = df['date'].max()
            old_pred = pd.read_pickle(PREDICT_PKL)
            pred_max = old_pred['date'].max()
            if feat_max == pred_max:
                # 检测退化日期：pred_return std 极低的日期
                daily_std = old_pred.groupby('date')['pred_return'].std()
                bad_dates = daily_std[daily_std < _DEGENERATE_STD_THRESH].index
                if len(bad_dates) == 0:
                    print(f"✅ [缓存命中] predictions.pkl 已是最新 ({pred_max.date()})，跳过全量重训练")
                    return old_pred
                else:
                    print(f"⚠️  [缓存命中但存在退化] 发现 {len(bad_dates)} 个退化日期，删除后补训...")
                    old_pred = old_pred[~old_pred['date'].isin(bad_dates)]
                    old_pred.to_pickle(PREDICT_PKL)
                    # 只对退化日期重训（通过 end_date 机制触发全量后合并）
                    # 直接走全量 walk-forward，让退化检测机制在训练中处理
        except Exception:
            pass  # 读取失败则继续正常流程

    # inf → NaN（LightGBM 原生处理 NaN）
    feature_cols = get_feature_columns(df)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    if end_date is not None:
        df = df[df['date'] <= pd.Timestamp(end_date)].copy()
        print(f"  [date filter] 数据截断至 {end_date}")

    pred_df = train_and_predict(df, end_date=end_date)

    # 保存预测结果
    if end_date is not None and os.path.exists(PREDICT_PKL):
        old_pred = pd.read_pickle(PREDICT_PKL)
        old_pred = old_pred[~old_pred['date'].isin(pred_df['date'].unique())]
        pred_df = pd.concat([old_pred, pred_df], ignore_index=True)
        pred_df = pred_df.sort_values(['date', 'code']).reset_index(drop=True)
    os.makedirs(os.path.dirname(PREDICT_PKL), exist_ok=True)
    pred_df.to_pickle(PREDICT_PKL)
    print(f"保存至 {PREDICT_PKL}")

    print(f"\n预测统计:")
    print(f"  日期范围: {pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
    print(f"  股票数: {pred_df['code'].nunique()}")
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
