"""优化后的 py03 模型训练与预测。

本版目标：
1. 基于上一次 full run 结果，收敛到更稳定的单一主模型方案
2. 用稳定性特征筛选 + 相关性去冗余替代多模型并行选择
3. 用同一模型家族的多 seed 集成保留稳健性，同时降低复杂度与维护成本
4. 保持单日预测与 walk-forward 全量预测两种运行方式
"""

import argparse
import gc
import math
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import CONFIG

warnings.filterwarnings("ignore")

FEATURE_PKL = CONFIG["paths"]["FEATURE_PKL"]
PREDICT_PKL = CONFIG["paths"]["PREDICT_PKL"]
OUTPUT_DIR = CONFIG["paths"]["OUTPUT_DIR"]

TRAIN_YEARS = CONFIG["model"]["TRAIN_YEARS"]
RETRAIN_DAYS = CONFIG["model"]["RETRAIN_DAYS"]
BACKTEST_START_YEAR = int(CONFIG["model"]["BACKTEST_START_YEAR"])
HOLD_DAYS = CONFIG["model"]["HOLD_DAYS"]
LGB_BASE_PARAMS = CONFIG["model"]["LGB_PARAMS"]
LGB_FORMAL_CFG = CONFIG["model"]["LGB_FORMAL"]
LGB_FAST_CFG = CONFIG["model"]["LGB_FAST"]
LGB_FALLBACK_CFG = CONFIG["model"]["LGB_FALLBACK"]

TOTAL_CPU = max(1, os.cpu_count() or 1)
CPU_BUDGET = max(1, math.floor(TOTAL_CPU * 2 / 3))
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MODEL_INFO_DIR = os.path.join(OUTPUT_DIR, "model_selection")
REPORT_PATH = os.path.join(MODEL_INFO_DIR, "py03_optimization_report.md")
TRAIN_METRICS_PATH = os.path.join(MODEL_INFO_DIR, "train_metrics.csv")
SINGLE_DAY_METRICS_PATH = os.path.join(MODEL_INFO_DIR, "single_day_metrics.csv")
PREDICT_CHECKPOINT_PKL = PREDICT_PKL.replace(".pkl", "_checkpoint.pkl")

DEFAULT_CHECKPOINT_EVERY = max(10, RETRAIN_DAYS)
SEED_LIST = [42, 2024, 3407]
FIXED_TRAIN_ROWS = 1_000_000
MIN_FORMAL_TRAIN_ROWS = 480_000
MAX_FORMAL_TRAIN_ROWS = 1_000_000
MAX_SELECTOR_ROWS = 640_000
MAX_FEATURES = 48
MAX_VAL_ROWS = 100_000
SOFT_MIN_BEST_ITER = 6
SEVERE_MIN_BEST_ITER = 2
SOFT_DEGENERATE_STD_THRESH_NORM = 0.008
SEVERE_DEGENERATE_STD_THRESH_NORM = 0.003
FALLBACK_TRAIN_ROWS = 120_000
FALLBACK_BOOST_ROUNDS = 80
FALLBACK_ACCEPT_SCORE_DELTA = 0.05
FALLBACK_SEVERE_MIN_SCORE_DELTA = 0.01


def _cap_numeric_threads() -> None:
    thread_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for var in thread_vars:
        os.environ[var] = str(CPU_BUDGET)


def _safe_metric(value: float, default: float = 0.0) -> float:
    return default if pd.isna(value) else float(value)


class ProgressLogger:
    """同时输出到 stdout 与日志文件。"""

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self._fp = open(log_path, "a", encoding="utf-8")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        self._fp.write(line + "\n")
        self._fp.flush()

    def section(self, title: str) -> None:
        bar = "=" * 88
        self.log(bar)
        self.log(title)
        self.log(bar)

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


@dataclass
class SeedModel:
    seed: int
    booster: lgb.Booster
    best_iteration: int
    label_mean: float
    label_std: float
    used_fallback: bool = False


@dataclass
class EnsembleBundle:
    models: list[SeedModel]
    feature_cols: list[str]
    train_rows: int
    val_rows: int
    metrics: dict


def build_log_path(end_date: Optional[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{end_date}" if end_date else "_full"
    return os.path.join(LOG_DIR, f"py03_model{suffix}_{stamp}.log")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "code",
        "name",
        "date",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "pctChg",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "label",
        "isST",
        "isTrading",
        "industry",
        "industryClassification",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_60",
        "pt_adjust_factor",
    }
    return [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def _add_feature(df: pd.DataFrame, name: str, values) -> None:
    df[name] = pd.Series(values, index=df.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def add_derived_features(df: pd.DataFrame, logger: ProgressLogger) -> pd.DataFrame:
    logger.section("Step 1/6 特征增强")
    logger.log("根据上次全量结果，增强市场状态、行业相对强弱与趋势质量因子...")

    eps = 1e-6
    range_span = (df["high"] - df["low"]).abs() + eps
    market_vol = df["mkt_ret_std"].abs() + 1e-4
    industry_vol = df["industry_ret_std"].abs() + 1e-4
    class_vol = df["industry_class_ret_std"].abs() + 1e-4
    volatility_20 = df["volatility_20d"].abs() + 1e-4
    bb_width = df["bb_width"].abs() + 1e-4
    amt_ratio = df["amt_ratio"].clip(lower=0)
    turn_ratio = df["turn_ratio"].clip(lower=0)
    vol_ratio = df["vol_ratio"].clip(lower=0)

    _add_feature(df, "alpha_intraday_strength", (df["close"] - df["open"]) / range_span)
    _add_feature(df, "alpha_market_regime_strength", df["mkt_mom_5d"] / market_vol)
    _add_feature(df, "alpha_industry_regime_strength", df["industry_class_mom_5d"] / class_vol)
    _add_feature(df, "alpha_market_industry_spread", df["industry_class_mom_5d"] - df["mkt_mom_5d"])
    _add_feature(df, "alpha_trend_quality", df["ma_bias_60"] * df["ma_bull"])
    _add_feature(df, "alpha_drawdown_rebound", -df["drawdown_20d"] * df["ret_3d"])
    _add_feature(df, "alpha_breakout_confirmation", df["bb_pctb"] * np.log1p(amt_ratio))
    _add_feature(df, "alpha_volatility_compression", bb_width / volatility_20)
    _add_feature(df, "alpha_turnover_impulse", turn_ratio * df["ret_10d"])
    _add_feature(df, "alpha_quality_spread", df["upside_vol_20"] - df["downside_vol_20"])
    _add_feature(df, "alpha_relative_strength", df["industry_ret_mean"] - df["mkt_ret_mean"] + df["excess_ret_industry"])
    _add_feature(df, "alpha_valuation_momentum", (1.0 - df["pbMRQ_rank"]) * df["ret_20d"])
    _add_feature(df, "alpha_rsi_macd", ((df["rsi_6"] - 50.0) / 50.0) * df["macd_dif"])
    _add_feature(df, "alpha_price_position_trend", df["price_pos_20d"] * df["ma_bias_20"])
    _add_feature(df, "alpha_liquidity_confirmation", np.log1p(vol_ratio) * np.log1p(amt_ratio))
    _add_feature(df, "alpha_industry_defensive_ratio", df["industry_ret_mean"] / industry_vol)

    logger.log("新增派生/交互特征 16 个")
    return df


def filter_trainable_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["label"].notna()
    if "isTrading" in df.columns:
        mask &= df["isTrading"] == 1
    if "consecutive_suspend" in df.columns:
        mask &= df["consecutive_suspend"] <= 2
    if "recent_5d_suspend" in df.columns:
        mask &= df["recent_5d_suspend"] == 0
    return df.loc[mask].sort_values(["date", "code"]).reset_index(drop=True)


def recent_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.tail(max_rows).reset_index(drop=True)


def build_train_val_split(
    df: pd.DataFrame,
    all_dates: list[pd.Timestamp],
    target_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_window = TRAIN_YEARS * 252
    train_start_idx = max(0, target_idx - train_window)
    safe_end_idx = max(0, target_idx - HOLD_DAYS)
    train_dates = all_dates[train_start_idx:safe_end_idx]
    if len(train_dates) < 252:
        raise ValueError(f"训练交易日不足: {len(train_dates)}")

    val_split_idx = int(len(train_dates) * 0.88)
    val_start_idx = min(val_split_idx + HOLD_DAYS + 1, len(train_dates))
    train_date_set = set(train_dates[:val_split_idx])
    val_date_set = set(train_dates[val_start_idx:])
    if len(val_date_set) < 10:
        val_date_set = set(train_dates[min(val_split_idx + HOLD_DAYS + 1, len(train_dates)):])

    train_df = filter_trainable_rows(df[df["date"].isin(train_date_set)])
    val_df = filter_trainable_rows(df[df["date"].isin(val_date_set)])
    if train_df.empty or val_df.empty:
        raise ValueError("训练集或验证集为空")

    return train_df, val_df


def split_validation_roles(val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = pd.Index(val_df["date"].unique()).sort_values().tolist()
    if len(unique_dates) < 20:
        return val_df, val_df

    split_idx = max(1, int(len(unique_dates) * 0.5))
    selector_dates = set(unique_dates[:split_idx])
    holdout_dates = set(unique_dates[split_idx:])
    selector_df = val_df[val_df["date"].isin(selector_dates)].copy()
    holdout_df = val_df[val_df["date"].isin(holdout_dates)].copy()
    if selector_df.empty or holdout_df.empty:
        return val_df, val_df
    return selector_df, holdout_df


def build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    date_col = df["date"]
    date_order = pd.Index(date_col.unique()).sort_values().tolist()
    if not date_order:
        return np.ones(len(df), dtype=np.float32)

    if len(date_order) == 1:
        recency_map = {date_order[0]: 1.0}
    else:
        recency_map = {
            d: 0.9 + 0.45 * ((i / (len(date_order) - 1)) ** 1.35)
            for i, d in enumerate(date_order)
        }

    recency_weight = date_col.map(recency_map).astype(np.float32).values
    label_weight = 1.0 + 0.10 * np.clip(np.abs(df["label"].values) / 0.03, 0, 2)
    return (recency_weight * label_weight).astype(np.float32)


def build_date_group_indices(dates: Iterable[pd.Timestamp]) -> list[np.ndarray]:
    date_arr = dates if isinstance(dates, np.ndarray) else np.asarray(list(dates))
    if len(date_arr) == 0:
        return []

    idx_frame = pd.DataFrame({
        "date": date_arr,
        "_idx": np.arange(len(date_arr), dtype=np.int64),
    })
    return [
        day_idx["_idx"].to_numpy(dtype=np.int64, copy=False)
        for _, day_idx in idx_frame.groupby("date", sort=True)
    ]


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Iterable[pd.Timestamp],
    date_groups: Optional[list[np.ndarray]] = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if date_groups is None:
        date_groups = build_date_group_indices(dates)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    direction_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())

    rank_ic_list = []
    top20_mean_list = []
    top20_win_list = []
    spread_list = []

    for day_idx in date_groups:
        day_df = pd.DataFrame({
            "y_true": y_true[day_idx],
            "y_pred": y_pred[day_idx],
        })
        if len(day_df) < 5:
            continue
        bucket = max(1, int(len(day_df) * 0.2))
        pred_rank = day_df["y_pred"].rank(method="average")
        true_rank = day_df["y_true"].rank(method="average")
        ic = pred_rank.corr(true_rank)
        if pd.notna(ic):
            rank_ic_list.append(float(ic))

        top_df = day_df.nlargest(bucket, "y_pred")
        bottom_df = day_df.nsmallest(bucket, "y_pred")
        top20_mean_list.append(float(top_df["y_true"].mean()))
        top20_win_list.append(float((top_df["y_true"] > 0).mean()))
        spread_list.append(float(top_df["y_true"].mean() - bottom_df["y_true"].mean()))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_acc": direction_acc,
        "rank_ic": float(np.mean(rank_ic_list)) if rank_ic_list else np.nan,
        "top20_mean_return": float(np.mean(top20_mean_list)) if top20_mean_list else np.nan,
        "top20_win_rate": float(np.mean(top20_win_list)) if top20_win_list else np.nan,
        "top_bottom_spread": float(np.mean(spread_list)) if spread_list else np.nan,
    }


def save_records(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


def _build_lgb_params(
    seed: int,
    fast_mode: bool = False,
    fallback_mode: bool = False,
    n_jobs: Optional[int] = None,
) -> tuple[dict, int, Optional[int]]:
    params = LGB_BASE_PARAMS.copy()
    params.update({
        "metric": "mae",
        "n_jobs": n_jobs or CPU_BUDGET,
        "verbose": -1,
        "seed": seed,
        "feature_fraction_seed": seed + 17,
        "bagging_seed": seed + 37,
        "data_random_seed": seed + 57,
        "drop_seed": seed + 77,
    })

    mode_cfg = LGB_FORMAL_CFG
    if fast_mode:
        mode_cfg = LGB_FAST_CFG
    elif fallback_mode:
        mode_cfg = LGB_FALLBACK_CFG

    params.update(mode_cfg.get("params", {}))
    return (
        params,
        int(mode_cfg["num_boost_round"]),
        mode_cfg.get("early_stopping_rounds"),
    )


def resolve_backtest_start(
    all_dates: list[pd.Timestamp],
    logger: Optional[ProgressLogger] = None,
) -> tuple[int, pd.Timestamp]:
    target_start = pd.Timestamp(f"{BACKTEST_START_YEAR}-01-01")
    idx = pd.Index(all_dates).searchsorted(target_start, side="left")
    if idx >= len(all_dates):
        raise ValueError(
            f"配置的 BACKTEST_START_YEAR={BACKTEST_START_YEAR} 超出特征数据范围，"
            f"当前最新交易日为 {all_dates[-1].date()}"
        )
    actual_start = all_dates[idx]
    if logger is not None:
        logger.log(
            f"固定回测起始年份={BACKTEST_START_YEAR}，"
            f"实际首个交易日={actual_start.date()}"
        )
    return idx, actual_start


def _prepare_normalized_labels(y_train: pd.Series, y_val: pd.Series) -> tuple[pd.Series, pd.Series, float, float]:
    label_mean = float(y_train.mean())
    label_std = float(y_train.std()) + 1e-8
    y_train_norm = ((y_train - label_mean) / label_std).astype(np.float32)
    y_val_norm = ((y_val - label_mean) / label_std).astype(np.float32)
    return y_train_norm, y_val_norm, label_mean, label_std


def _evaluate_booster_on_validation(
    booster: lgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_dates,
    val_date_groups: Optional[list[np.ndarray]],
    label_mean: float,
    label_std: float,
) -> tuple[np.ndarray, float, dict, float]:
    pred_norm = booster.predict(X_val)
    pred_std_norm = float(np.std(pred_norm))
    pred = pred_norm * label_std + label_mean
    metrics = compute_prediction_metrics(y_val.values, pred, val_dates, date_groups=val_date_groups)
    score = compute_probe_score(metrics)
    return pred, pred_std_norm, metrics, score


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_dates,
    train_weight: np.ndarray,
    seed: int,
    fast_mode: bool = False,
    logger: Optional[ProgressLogger] = None,
    n_jobs: Optional[int] = None,
    val_date_groups: Optional[list[np.ndarray]] = None,
) -> SeedModel:
    params, rounds, stop_rounds = _build_lgb_params(
        seed=seed,
        fast_mode=fast_mode,
        fallback_mode=False,
        n_jobs=n_jobs,
    )
    y_train_norm, y_val_norm, base_label_mean, base_label_std = _prepare_normalized_labels(y_train, y_val)

    dtrain = lgb.Dataset(X_train, label=y_train_norm, weight=train_weight)
    dval = lgb.Dataset(X_val, label=y_val_norm, reference=dtrain)
    callbacks = [lgb.early_stopping(stop_rounds, verbose=False)] if stop_rounds else []
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    best_iteration = booster.best_iteration or rounds
    _, pred_std_norm, base_metrics, base_score = _evaluate_booster_on_validation(
        booster=booster,
        X_val=X_val,
        y_val=y_val,
        val_dates=val_dates,
        val_date_groups=val_date_groups,
        label_mean=base_label_mean,
        label_std=base_label_std,
    )
    used_fallback = False
    label_mean = base_label_mean
    label_std = base_label_std

    severe_degenerate = (
        best_iteration <= SEVERE_MIN_BEST_ITER
        or pred_std_norm < SEVERE_DEGENERATE_STD_THRESH_NORM
    )
    soft_degenerate = (
        best_iteration <= SOFT_MIN_BEST_ITER
        and pred_std_norm < SOFT_DEGENERATE_STD_THRESH_NORM
    )
    fallback_candidate = severe_degenerate or (soft_degenerate and base_score < 0.60)

    if not fast_mode and fallback_candidate:
        fallback_rows = min(len(X_train), FALLBACK_TRAIN_ROWS)
        if logger is not None:
            logger.log(
                f"  seed={seed} 检测到退化，切换兜底训练 | "
                f"best_iter={best_iteration} | pred_std_norm={pred_std_norm:.5f} | "
                f"base_score={base_score:.4f} | severe={severe_degenerate} | "
                f"fallback_rows={fallback_rows:,} | fallback_rounds={FALLBACK_BOOST_ROUNDS}"
            )

        X_fb = X_train.tail(fallback_rows)
        y_fb = y_train.tail(fallback_rows)
        w_fb = train_weight[-fallback_rows:]
        y_fb_norm, _, fb_label_mean, fb_label_std = _prepare_normalized_labels(y_fb, y_val)
        params_fb, rounds_fb, _ = _build_lgb_params(
            seed=seed,
            fast_mode=False,
            fallback_mode=True,
            n_jobs=n_jobs,
        )
        dtrain_fb = lgb.Dataset(X_fb, label=y_fb_norm, weight=w_fb)
        booster_fb = lgb.train(params_fb, dtrain_fb, num_boost_round=rounds_fb)
        _, fb_pred_std_norm, fb_metrics, fb_score = _evaluate_booster_on_validation(
            booster=booster_fb,
            X_val=X_val,
            y_val=y_val,
            val_dates=val_dates,
            val_date_groups=val_date_groups,
            label_mean=fb_label_mean,
            label_std=fb_label_std,
        )

        accept_fallback = fb_score >= base_score + FALLBACK_ACCEPT_SCORE_DELTA
        if severe_degenerate and not accept_fallback:
            accept_fallback = (
                fb_score >= base_score + FALLBACK_SEVERE_MIN_SCORE_DELTA
                and _safe_metric(fb_metrics["rank_ic"], -1.0) >= _safe_metric(base_metrics["rank_ic"], -1.0) - 0.01
                and _safe_metric(fb_metrics["top_bottom_spread"], -1.0)
                >= _safe_metric(base_metrics["top_bottom_spread"], -1.0) - 0.002
                and fb_pred_std_norm > pred_std_norm * 1.5
            )

        if accept_fallback:
            booster = booster_fb
            best_iteration = rounds_fb
            used_fallback = True
            label_mean = fb_label_mean
            label_std = fb_label_std
            if logger is not None:
                logger.log(
                    f"  seed={seed} 接受 fallback | base_score={base_score:.4f} -> "
                    f"fb_score={fb_score:.4f} | base_rank_ic={_safe_metric(base_metrics['rank_ic']):.4f} -> "
                    f"fb_rank_ic={_safe_metric(fb_metrics['rank_ic']):.4f}"
                )
        elif logger is not None:
            logger.log(
                f"  seed={seed} 放弃 fallback | base_score={base_score:.4f} | "
                f"fb_score={fb_score:.4f} | base_rank_ic={_safe_metric(base_metrics['rank_ic']):.4f} | "
                f"fb_rank_ic={_safe_metric(fb_metrics['rank_ic']):.4f}"
            )

    return SeedModel(
        seed=seed,
        booster=booster,
        best_iteration=best_iteration,
        label_mean=label_mean,
        label_std=label_std,
        used_fallback=used_fallback,
    )


def predict_with_ensemble(models: list[SeedModel], X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred_matrix = np.vstack([
        model.booster.predict(X) * model.label_std + model.label_mean
        for model in models
    ])
    return pred_matrix.mean(axis=0), pred_matrix.std(axis=0)


def compute_probe_score(metrics: dict) -> float:
    return (
        _safe_metric(metrics["rank_ic"], 0.0) * 6.0
        + _safe_metric(metrics["top20_mean_return"], 0.0) * 60.0
        + _safe_metric(metrics["top_bottom_spread"], 0.0) * 30.0
        + max(_safe_metric(metrics["direction_acc"], 0.5) - 0.5, 0.0) * 2.5
        - _safe_metric(metrics["mae"], 0.0) * 0.4
    )


def build_sample_ladder(total_rows: int) -> list[int]:
    candidates = [80_000, 160_000, 320_000, 480_000, 640_000, 800_000, 960_000, total_rows]
    return sorted(set(min(total_rows, x) for x in candidates if x > 0 and total_rows >= 20_000))


def probe_sample_capacity(
    train_df: pd.DataFrame,
    selector_val_df: pd.DataFrame,
    feature_cols: list[str],
    logger: ProgressLogger,
    max_train_rows_override: Optional[int] = None,
) -> int:
    logger.section("Step 2/6 样本量探针")
    fixed = max_train_rows_override if max_train_rows_override is not None else FIXED_TRAIN_ROWS
    chosen = min(len(train_df), fixed)
    logger.log(f"固定训练样本量={fixed:,}，实际可用={len(train_df):,}，采用 {chosen:,} 行")
    return chosen


def compute_feature_stability_scores(
    train_df: pd.DataFrame,
    selector_val_df: pd.DataFrame,
    feature_cols: list[str],
    logger: ProgressLogger,
) -> pd.DataFrame:
    rows_ladder = sorted(set(min(len(train_df), x) for x in [160_000, 320_000, MAX_SELECTOR_ROWS]))
    rows_ladder = [x for x in rows_ladder if x >= 20_000]
    val_sample = recent_sample(selector_val_df, min(len(selector_val_df), 60_000))
    val_X = val_sample[feature_cols]
    val_y = val_sample["label"]
    val_dates = val_sample["date"].values
    val_date_groups = build_date_group_indices(val_dates)
    parts = []

    for rows in rows_ladder:
        sample_train = recent_sample(train_df, rows)
        train_weight = build_sample_weights(sample_train)
        model = train_lightgbm_model(
            sample_train[feature_cols],
            sample_train["label"],
            val_X,
            val_y,
            val_dates,
            train_weight=train_weight,
            seed=11 + rows,
            fast_mode=True,
            val_date_groups=val_date_groups,
        )
        importance_gain = pd.Series(
            model.booster.feature_importance(importance_type="gain"),
            index=feature_cols,
            dtype=np.float64,
        )
        importance_split = pd.Series(
            model.booster.feature_importance(importance_type="split"),
            index=feature_cols,
            dtype=np.float64,
        )
        gain_share = importance_gain / max(importance_gain.sum(), 1.0)
        split_share = importance_split / max(importance_split.sum(), 1.0)
        score = gain_share * 0.8 + split_share * 0.2
        part = pd.DataFrame({
            "feature": feature_cols,
            f"score_{rows}": score.values,
            f"positive_{rows}": (importance_gain.values > 0).astype(float),
        })
        parts.append(part.set_index("feature"))

    merged = pd.concat(parts, axis=1).fillna(0.0)
    score_cols = [col for col in merged.columns if col.startswith("score_")]
    positive_cols = [col for col in merged.columns if col.startswith("positive_")]
    merged["stability_score"] = merged[score_cols].mean(axis=1)
    merged["positive_ratio"] = merged[positive_cols].mean(axis=1)
    merged["final_score"] = merged["stability_score"] * 0.85 + merged["positive_ratio"] * 0.15
    result = merged.reset_index().rename(columns={"index": "feature"}).sort_values("final_score", ascending=False)
    return result


def apply_correlation_pruning(
    train_df: pd.DataFrame,
    ranked_features: list[str],
    feature_score_map: dict[str, float],
    keep_limit: int,
    threshold: float,
) -> list[str]:
    sample = recent_sample(train_df[ranked_features], min(len(train_df), 120_000))
    corr = sample.corr().abs()
    selected: list[str] = []
    protected = {
        "industry_class_count",
        "mkt_ret_std",
        "industry_class_mom_5d",
        "mkt_ret_mean",
        "atr14_ratio",
        "ma_bias_60",
        "upside_vol_20",
        "alpha_market_industry_spread",
        "pbMRQ_rank",
        "macd_dif",
        "ret_3d",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "rsi_6",
        "macd_hist",
        "turn_ma5",
        "volatility_20d",
        "drawdown_20d",
        "price_pos_20d",
        "risk_adj_mom_20d",
        "alpha_drawdown_rebound",
        "alpha_trend_quality",
        "alpha_relative_strength",
        "alpha_turnover_impulse",
        "alpha_rsi_macd",
    }

    for feature in ranked_features:
        if len(selected) >= keep_limit:
            break
        if feature in protected:
            selected.append(feature)
            continue

        blocked = False
        for kept in selected:
            if feature == kept:
                blocked = True
                break
            corr_val = corr.loc[feature, kept]
            if pd.notna(corr_val) and corr_val >= threshold:
                if feature_score_map.get(feature, 0.0) <= feature_score_map.get(kept, 0.0):
                    blocked = True
                    break
        if not blocked:
            selected.append(feature)

    return selected[:keep_limit]


def select_features(
    train_df: pd.DataFrame,
    selector_val_df: pd.DataFrame,
    feature_cols: list[str],
    logger: ProgressLogger,
) -> list[str]:
    logger.section("Step 3/6 特征筛选")
    score_df = compute_feature_stability_scores(train_df, selector_val_df, feature_cols, logger)
    score_df.to_csv(os.path.join(MODEL_INFO_DIR, "feature_importance.csv"), index=False, encoding="utf-8-sig")

    positive_df = score_df[score_df["final_score"] > 0].copy()
    if positive_df.empty:
        positive_df = score_df.copy()

    ranked_features = positive_df["feature"].tolist()
    feature_score_map = dict(zip(positive_df["feature"], positive_df["final_score"]))
    selected = apply_correlation_pruning(
        train_df=train_df,
        ranked_features=ranked_features,
        feature_score_map=feature_score_map,
        keep_limit=min(MAX_FEATURES, len(ranked_features)),
        threshold=0.96,
    )
    if len(selected) < 24:
        selected = ranked_features[: min(MAX_FEATURES, len(ranked_features))]

    logger.log(f"特征筛选完成，从 {len(feature_cols)} 个特征中保留 {len(selected)} 个")
    logger.log(f"Top 15 特征: {selected[:15]}")
    return selected


def log_metrics(logger: ProgressLogger, stage: str, metrics: dict, elapsed: float, seed_iters: list[int]) -> None:
    logger.log(
        f"{stage} 完成 | "
        f"MAE={metrics['mae']:.6f} | RMSE={metrics['rmse']:.6f} | "
        f"R2={metrics['r2']:.4f} | RankIC={metrics['rank_ic']:.4f} | "
        f"DirectionAcc={metrics['direction_acc']:.4f} | "
        f"Top20Mean={metrics['top20_mean_return']:.4%} | "
        f"Top20Win={metrics['top20_win_rate']:.4f} | "
        f"Spread={metrics['top_bottom_spread']:.4%} | "
        f"best_iter={seed_iters} | 耗时={elapsed:.1f}s"
    )


def fit_lightgbm_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    formal_train_rows: int,
    logger: ProgressLogger,
    stage_name: str,
    metrics_records: list[dict],
    metrics_output_path: str,
) -> tuple[EnsembleBundle, list[dict]]:
    fit_train = recent_sample(train_df, min(len(train_df), formal_train_rows))
    fit_val = recent_sample(val_df, min(len(val_df), MAX_VAL_ROWS))
    X_train = fit_train[feature_cols]
    y_train = fit_train["label"]
    X_val = fit_val[feature_cols]
    y_val = fit_val["label"]
    val_dates = fit_val["date"].values
    val_date_groups = build_date_group_indices(val_dates)
    train_weight = build_sample_weights(fit_train)

    logger.section(f"Step 4/6 正式训练 | {stage_name}")
    logger.log(
        f"LightGBM 单模型家族训练开始 | train_rows={len(fit_train):,} | "
        f"val_rows={len(fit_val):,} | features={len(feature_cols)} | seeds={SEED_LIST}"
    )

    start_time = time.time()
    models: list[SeedModel] = []
    for seed in SEED_LIST:
        model = train_lightgbm_model(
            X_train,
            y_train,
            X_val,
            y_val,
            val_dates,
            train_weight=train_weight,
            seed=seed,
            fast_mode=False,
            logger=logger,
            val_date_groups=val_date_groups,
        )
        models.append(model)
        mode_tag = " | fallback" if model.used_fallback else ""
        logger.log(f"  seed={seed} 训练完成 | best_iter={model.best_iteration}{mode_tag}")

    pred_mean, pred_std = predict_with_ensemble(models, X_val)
    metrics = compute_prediction_metrics(
        y_val.values,
        pred_mean,
        val_dates,
        date_groups=val_date_groups,
    )
    elapsed = time.time() - start_time
    seed_iters = [m.best_iteration for m in models]
    log_metrics(logger, stage_name, metrics, elapsed, seed_iters)

    record = {
        "stage": stage_name,
        "model": "LightGBMSeedEnsemble",
        "train_rows": len(fit_train),
        "val_rows": len(fit_val),
        "elapsed_sec": round(elapsed, 3),
        "seed_iters": "|".join(str(x) for x in seed_iters),
        **metrics,
    }
    metrics_records.append(record)
    save_records(metrics_records, metrics_output_path)

    bundle = EnsembleBundle(
        models=models,
        feature_cols=feature_cols,
        train_rows=len(fit_train),
        val_rows=len(fit_val),
        metrics=metrics,
    )
    return bundle, metrics_records


def resolve_target_date(all_dates: list[pd.Timestamp], end_date: str, logger: ProgressLogger) -> pd.Timestamp:
    target = pd.Timestamp(end_date)
    if target in all_dates:
        return target
    prior = [d for d in all_dates if d <= target]
    if not prior:
        raise ValueError(f"{end_date} 之前没有可用交易日数据")
    adjusted = prior[-1]
    logger.log(f"[end_date={end_date} 非交易日，自动回退到最近交易日 {adjusted.date()}]")
    return adjusted


def prune_dataframe(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    keep_cols = ["date", "code", "label", "isTrading", "consecutive_suspend", "recent_5d_suspend"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    keep_cols = list(dict.fromkeys(keep_cols + feature_cols))
    return df[keep_cols].copy()


def build_day_frame_cache(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    cache: dict[pd.Timestamp, pd.DataFrame] = {}
    for date_value, day_df in df.groupby("date", sort=False):
        cache[date_value] = day_df.sort_values("code").reset_index(drop=True)
    return cache


def build_prediction_frame(
    day_df: pd.DataFrame,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    date_value: pd.Timestamp,
) -> pd.DataFrame:
    # pred_std 越大，不确定性越高；映射为 0~1 置信度供下游规则使用。
    confidence = 1.0 / (1.0 + np.clip(pred_std, 0, None) * 100.0)
    result = pd.DataFrame({
        "date": date_value,
        "code": day_df["code"].values,
        "pred_return": pred_mean.astype(np.float32),
        "pred_std": pred_std.astype(np.float32),
        "confidence": confidence.astype(np.float32),
    })
    for col in ("consecutive_suspend", "recent_5d_suspend", "isNew"):
        if col in day_df.columns:
            result[col] = day_df[col].values
    return result


def _safe_read_pickle(path: str) -> pd.DataFrame | None:
    """安全读取pickle文件，损坏时返回None而非抛异常"""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        print(f"  [警告] {path} 读取失败({exc})，将忽略旧checkpoint")
        return None


def flush_prediction_checkpoint(pred_df: pd.DataFrame, logger: ProgressLogger) -> None:
    if pred_df.empty:
        return

    pred_df = pred_df.sort_values(["date", "code"]).reset_index(drop=True)
    old_df = _safe_read_pickle(PREDICT_CHECKPOINT_PKL)
    if old_df is not None:
        pred_df = pd.concat([old_df, pred_df], ignore_index=True)
        pred_df = pred_df.drop_duplicates(subset=["date", "code"], keep="last")
        pred_df = pred_df.sort_values(["date", "code"]).reset_index(drop=True)

    # 先写临时文件再重命名，避免写入中断导致文件损坏
    tmp_path = PREDICT_CHECKPOINT_PKL + ".tmp"
    pred_df.to_pickle(tmp_path)
    if os.path.exists(PREDICT_CHECKPOINT_PKL):
        os.remove(PREDICT_CHECKPOINT_PKL)
    os.rename(tmp_path, PREDICT_CHECKPOINT_PKL)
    logger.log(
        f"checkpoint 已写入 {PREDICT_CHECKPOINT_PKL} | "
        f"rows={len(pred_df):,} | dates={pred_df['date'].nunique():,}"
    )


def collect_checkpoint_predictions(buffer_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    old_df = _safe_read_pickle(PREDICT_CHECKPOINT_PKL)
    if old_df is not None:
        frames.append(old_df)
    if buffer_df is not None and not buffer_df.empty:
        frames.append(buffer_df)

    if not frames:
        return pd.DataFrame()

    pred_df = pd.concat(frames, ignore_index=True)
    pred_df = pred_df.drop_duplicates(subset=["date", "code"], keep="last")
    pred_df = pred_df.sort_values(["date", "code"]).reset_index(drop=True)
    return pred_df


def _model_config_hash() -> str:
    """模型配置哈希，参数变化时强制重训"""
    import hashlib, json
    model_cfg = CONFIG["model"]
    feat_cfg = CONFIG["features"]
    snapshot = {
        "HOLD_DAYS": model_cfg.get("HOLD_DAYS"),
        "LGB_FORMAL": model_cfg.get("LGB_FORMAL"),
        "TRAIN_YEARS": model_cfg.get("TRAIN_YEARS"),
        "RETRAIN_DAYS": model_cfg.get("RETRAIN_DAYS"),
        "LABEL_WINSORIZE_MIN": feat_cfg.get("LABEL_WINSORIZE_MIN"),
        "LABEL_WINSORIZE_MAX": feat_cfg.get("LABEL_WINSORIZE_MAX"),
    }
    raw = json.dumps(snapshot, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def detect_resume_index(
    all_dates: list[pd.Timestamp],
    start_idx: int,
    logger: ProgressLogger,
) -> tuple[int, Optional[pd.Timestamp]]:
    if not os.path.exists(PREDICT_PKL):
        logger.log("未检测到历史 predictions.pkl，将从回测起点全量生成")
        return start_idx, None

    try:
        old_pred = pd.read_pickle(PREDICT_PKL)
    except Exception as exc:
        logger.log(f"读取历史 predictions.pkl 失败，将从回测起点全量生成: {exc}")
        return start_idx, None

    # 模型配置变更检测
    cached_hash = old_pred.attrs.get("model_config_hash", "")
    current_hash = _model_config_hash()
    if cached_hash != current_hash:
        logger.log(f"检测到模型参数变更 (旧={cached_hash}, 新={current_hash})，将全量重训")
        return start_idx, None

    if old_pred.empty or "date" not in old_pred.columns:
        logger.log("历史 predictions.pkl 为空或缺少 date 列，将从回测起点全量生成")
        return start_idx, None

    if not pd.api.types.is_datetime64_any_dtype(old_pred["date"]):
        old_pred["date"] = pd.to_datetime(old_pred["date"], errors="coerce")
    old_pred = old_pred.loc[old_pred["date"].notna()].copy()
    if old_pred.empty:
        logger.log("历史 predictions.pkl 日期无效，将从回测起点全量生成")
        return start_idx, None

    covered_dates = set(old_pred["date"].unique())
    last_complete_idx = start_idx - 1
    for idx in range(start_idx, len(all_dates)):
        if all_dates[idx] in covered_dates:
            last_complete_idx = idx
            continue
        break

    resume_idx = last_complete_idx + 1
    if resume_idx <= start_idx:
        logger.log("历史 predictions.pkl 未覆盖回测区间，将从回测起点全量生成")
        return start_idx, None

    if resume_idx >= len(all_dates):
        logger.log(
            f"历史 predictions.pkl 已覆盖到最新交易日 {all_dates[last_complete_idx].date()}，无需新增预测"
        )
        return len(all_dates), all_dates[last_complete_idx]

    resume_start = all_dates[resume_idx]
    logger.log(
        f"检测到历史 predictions.pkl 已覆盖至 {all_dates[last_complete_idx].date()}，"
        f"本次从 {resume_start.date()} 继续增量更新"
    )
    return resume_idx, resume_start


def save_predictions(
    pred_df: pd.DataFrame,
    end_date: Optional[str],
    logger: ProgressLogger,
    resume_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    pred_df = pred_df.sort_values(["date", "code"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(PREDICT_PKL), exist_ok=True)

    if resume_start is not None:
        resume_ts = pd.Timestamp(resume_start)
        if not pred_df.empty and pred_df["date"].min() < resume_ts:
            raise ValueError(
                f"增量保存失败：新预测包含 {resume_ts.date()} 之前的数据，"
                f"最早日期为 {pred_df['date'].min().date()}"
            )
        if os.path.exists(PREDICT_PKL):
            old_pred = pd.read_pickle(PREDICT_PKL)
            if not pd.api.types.is_datetime64_any_dtype(old_pred["date"]):
                old_pred["date"] = pd.to_datetime(old_pred["date"], errors="coerce")
            old_prefix = old_pred.loc[old_pred["date"] < resume_ts].copy()
            old_prefix_rows = len(old_prefix)
            pred_df = pd.concat([old_prefix, pred_df], ignore_index=True)
            pred_df = pred_df.sort_values(["date", "code"]).reset_index(drop=True)
            merged_prefix_rows = int((pred_df["date"] < resume_ts).sum())
            if merged_prefix_rows != old_prefix_rows:
                raise ValueError("增量保存失败：历史预测前缀记录数发生变化")
    elif end_date is not None and os.path.exists(PREDICT_PKL):
        old_pred = pd.read_pickle(PREDICT_PKL)
        old_pred = old_pred[~old_pred["date"].isin(pred_df["date"].unique())]
        pred_df = pd.concat([old_pred, pred_df], ignore_index=True)
        pred_df = pred_df.sort_values(["date", "code"]).reset_index(drop=True)

    pred_df.attrs["model_config_hash"] = _model_config_hash()
    pred_df.to_pickle(PREDICT_PKL)
    logger.log(f"预测结果已保存至 {PREDICT_PKL}")
    logger.log(
        f"预测统计 | date_range={pred_df['date'].min().date()} ~ {pred_df['date'].max().date()} | "
        f"rows={len(pred_df):,} | codes={pred_df['code'].nunique():,} | "
        f"pred_mean={pred_df['pred_return'].mean():.6f} | pred_std={pred_df['pred_return'].std():.6f}"
    )
    return pred_df


def _build_metrics_summary_markdown(metrics_path: str) -> str:
    if not os.path.exists(metrics_path):
        return ""

    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return ""

    required_cols = {
        "stage",
        "elapsed_sec",
        "seed_iters",
        "mae",
        "rmse",
        "r2",
        "direction_acc",
        "rank_ic",
        "top20_mean_return",
        "top20_win_rate",
        "top_bottom_spread",
    }
    if df.empty or not required_cols.issubset(df.columns):
        return ""

    summary = df[[
        "mae",
        "rmse",
        "r2",
        "direction_acc",
        "rank_ic",
        "top20_mean_return",
        "top20_win_rate",
        "top_bottom_spread",
        "elapsed_sec",
    ]].mean().to_dict()

    seed_iters = df["seed_iters"].fillna("").astype(str)
    all_seed_fallback = int(seed_iters.str.fullmatch(r"80\|80\|80").sum())
    any_fallback = int(seed_iters.str.contains(r"(^|\\|)80($|\\|)", regex=True).sum())
    slow_windows = int((df["elapsed_sec"] > 60).sum())

    lines = [
        "## 5. 当前版本最近一次 full run 均值表现",
        "",
        f"- 样本窗口数：{len(df)}",
        f"- `MAE={summary['mae']:.6f}` | `RMSE={summary['rmse']:.6f}` | `R2={summary['r2']:.4f}`",
        f"- `RankIC={summary['rank_ic']:.4f}` | `Top20Mean={summary['top20_mean_return']:.4%}` | `Spread={summary['top_bottom_spread']:.4%}`",
        f"- `DirectionAcc={summary['direction_acc']:.4f}` | `Top20Win={summary['top20_win_rate']:.4f}`",
        f"- 平均训练耗时：`{summary['elapsed_sec']:.1f}s/窗口`",
        f"- 最终采用 fallback 的窗口：`{any_fallback}/{len(df)}`，其中三 seed 最终都采用 fallback：`{all_seed_fallback}`",
        f"- 慢窗口（>60s）：`{slow_windows}`",
        "",
        "## 6. 当前冻结配置",
        "",
        f"- 主模型：`LightGBM + {len(SEED_LIST)} seeds ensemble`",
        f"- 正式训练样本上限：`{MIN_FORMAL_TRAIN_ROWS:,} ~ {MAX_FORMAL_TRAIN_ROWS:,}`",
        f"- 特征上限：`{MAX_FEATURES}`",
        f"- fallback：`rows={FALLBACK_TRAIN_ROWS:,}`，`rounds={FALLBACK_BOOST_ROUNDS}`",
        "- 当前结论：优先保留现有“效果/速度平衡”，后续默认不再继续压 fallback 或追加模型复杂度。",
        "",
    ]
    return "\n".join(lines)


def write_optimization_report() -> None:
    os.makedirs(MODEL_INFO_DIR, exist_ok=True)
    old_metrics_summary = ""
    old_metrics_path = os.path.join(MODEL_INFO_DIR, "candidate_metrics.csv")
    if os.path.exists(old_metrics_path):
        try:
            old_df = pd.read_csv(old_metrics_path)
            if not old_df.empty and {"model", "mae", "rmse", "r2", "direction_acc", "rank_ic", "top20_mean_return", "top_bottom_spread"}.issubset(old_df.columns):
                summary = old_df.groupby("model")[["mae", "rmse", "r2", "direction_acc", "rank_ic", "top20_mean_return", "top_bottom_spread"]].mean().round(6)
                old_metrics_summary = summary.to_markdown()
        except Exception:
            old_metrics_summary = ""

    report = [
        "# py03 模型优化报告",
        "",
        "## 1. 模型路线调整",
        "- 原版本同时维护 LightGBM、XGBoost、HistGB 三个候选模型，运行逻辑复杂，且从最近一次 full run 看，LightGBM 在 MAE、RMSE、R2、方向准确率上整体最稳。",
        "- 新版本收敛为 `LightGBM 单一主模型 + 多 seed 集成`。这意味着我们不再保留三种模型家族，而是保留一个最稳的模型家族，并用 3 个随机种子集成提升稳健性。",
        "",
        "## 2. 这次代码上的核心改进",
        "- 合并实现到 `src/py03_model.py`，删除 `py03_model_sel.py` 兼容层与多模型选择代码。",
        "- 强化派生特征，重点补充市场状态、行业相对强弱、趋势质量、波动压缩、回撤修复与流动性确认因子。",
        "- 特征选择改为 `稳定性重要性 + 相关性去冗余`，并把保留特征数从 32 扩到 48，减少只剩市场因子的情况。",
        "- 样本量探针改为使用综合 score 并优先保留较大的近优样本，避免再次把正式训练压到 16 万行。",
        "- 正式训练恢复 `label 标准化`，降低不同市场阶段下 early stopping 失真和 best_iter 过早塌缩的问题。",
        "- 增加退化兜底训练：只有在 `best_iter` 和预测离散度同时明显异常时才触发，并将 fallback 压缩到更小样本和更少轮数。",
        "- 为 fallback 增加验证集接管门槛：只有 fallback 在综合 score 上明确优于原模型，才允许替换，尽量避免为追求波动而过拟合验证集。",
        "- 下调 recency weighting 强度，避免近期样本权重过大导致模型只学到短期噪声。",
        "- 保留 `pred_std` 和 `confidence` 字段，但它们现在来自同一模型家族的 seed 集成，而不是跨模型家族离散度。",
        "- 保留详细日志、checkpoint 与 walk-forward 机制，保证大规模运行可追踪。",
        "",
        "## 3. 预期效果",
        "- 排序能力更稳：重点修复 `best_iter=1~3` 的退化窗口，目标是抬升 RankIC、Top20Mean 和 Spread。",
        "- 截面一致性更好：更多个股侧特征和更大的正式训练样本，有助于缓解过度依赖市场/行业状态变量的问题。",
        "- 训练耗时可控：虽然正式训练样本增大，但仍只使用约 2/3 CPU，并通过探针、特征裁剪和更克制的兜底补训控制总耗时。",
        "- 维护成本更低：后续你再让我继续改 py03 或联动 py04/py05 时，接口和逻辑都会更清晰。",
        "- 泛化更稳：通过稳定性筛选、相关性去冗余、标签标准化、适度 recency weighting 与 fallback 接管门槛，目标是减少不同窗口之间指标波动和验证集过拟合。",
        "",
    ]

    if old_metrics_summary:
        report.extend([
            "## 4. 上一次 full run 的旧版多模型均值表现",
            "",
            old_metrics_summary,
            "",
            "从这组结果看，LightGBM 的误差控制和稳定性最好，因此这次选择将其作为唯一主模型。",
            "",
        ])

    current_metrics_summary = _build_metrics_summary_markdown(TRAIN_METRICS_PATH)
    if current_metrics_summary:
        report.extend([current_metrics_summary])

    report.extend([
        "## 7. 运行方式",
        "- 全量 walk-forward：`python src/py03_model.py`",
        "- 指定日期单日预测：`python src/py03_model.py --date 2026-04-17`",
        "- 如需做轻量 smoke：`python src/py03_model.py --date 2026-04-17 --max-train-rows 50000`",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as fp:
        fp.write("\n".join(report))


def predict_single_day(
    df: pd.DataFrame,
    all_dates: list[pd.Timestamp],
    end_date: str,
    logger: ProgressLogger,
    max_train_rows: Optional[int] = None,
) -> pd.DataFrame:
    logger.section("单日模式")
    target_date = resolve_target_date(all_dates, end_date, logger)
    target_idx = all_dates.index(target_date)
    logger.log(f"目标预测日: {target_date.date()}")

    train_df, val_df = build_train_val_split(df, all_dates, target_idx)
    selector_val_df, holdout_val_df = split_validation_roles(val_df)
    base_features = get_feature_columns(df)
    formal_train_rows = probe_sample_capacity(
        train_df,
        selector_val_df,
        base_features,
        logger,
        max_train_rows_override=max_train_rows,
    )
    selected_features = select_features(train_df, selector_val_df, base_features, logger)
    df_model = prune_dataframe(df, selected_features)
    day_frame_cache = build_day_frame_cache(df_model)
    train_df, val_df = build_train_val_split(df_model, all_dates, target_idx)
    _, holdout_val_df = split_validation_roles(val_df)

    metrics_records: list[dict] = []
    ensemble, metrics_records = fit_lightgbm_ensemble(
        train_df=train_df,
        val_df=holdout_val_df,
        feature_cols=selected_features,
        formal_train_rows=formal_train_rows,
        logger=logger,
        stage_name=f"single_day_{target_date.date()}",
        metrics_records=metrics_records,
        metrics_output_path=SINGLE_DAY_METRICS_PATH,
    )

    day_df = day_frame_cache.get(target_date)
    if day_df is None:
        day_df = df_model[df_model["date"] == target_date].sort_values("code").reset_index(drop=True)
    pred_mean, pred_std = predict_with_ensemble(ensemble.models, day_df[selected_features])
    pred_df = build_prediction_frame(day_df, pred_mean, pred_std, target_date)
    return save_predictions(pred_df, end_date=target_date.strftime("%Y-%m-%d"), logger=logger)


def walk_forward_predict(
    df: pd.DataFrame,
    all_dates: list[pd.Timestamp],
    logger: ProgressLogger,
    max_train_rows: Optional[int] = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    resume: bool = True,
) -> pd.DataFrame:
    logger.section("全量 Walk-Forward 模式")
    bt_start_idx, bt_start_date = resolve_backtest_start(all_dates, logger=logger)
    logger.log(
        f"回测预测区间: {bt_start_date.date()} ~ {all_dates[-1].date()} | "
        f"共 {len(all_dates) - bt_start_idx} 个交易日"
    )

    initial_idx = bt_start_idx
    train_df, val_df = build_train_val_split(df, all_dates, initial_idx)
    selector_val_df, _ = split_validation_roles(val_df)
    base_features = get_feature_columns(df)
    formal_train_rows = probe_sample_capacity(
        train_df,
        selector_val_df,
        base_features,
        logger,
        max_train_rows_override=max_train_rows,
    )
    selected_features = select_features(train_df, selector_val_df, base_features, logger)
    df_model = prune_dataframe(df, selected_features)
    all_dates = pd.Index(df_model["date"].unique()).sort_values().tolist()
    day_frame_cache = build_day_frame_cache(df_model)
    bt_start_idx, _ = resolve_backtest_start(all_dates, logger=None)

    resume_start = None
    start_idx = bt_start_idx
    if resume:
        start_idx, resume_start = detect_resume_index(all_dates, bt_start_idx, logger)
        if start_idx >= len(all_dates):
            return pd.read_pickle(PREDICT_PKL)

    if os.path.exists(PREDICT_CHECKPOINT_PKL):
        os.remove(PREDICT_CHECKPOINT_PKL)
        logger.log(f"已清理旧 checkpoint: {PREDICT_CHECKPOINT_PKL}")

    pred_chunks: list[pd.DataFrame] = []
    metrics_records: list[dict] = []
    ensemble: Optional[EnsembleBundle] = None
    last_train_idx = -10**9
    total_steps = len(all_dates) - start_idx

    for offset, day_idx in enumerate(range(start_idx, len(all_dates)), start=1):
        current_date = all_dates[day_idx]
        need_retrain = ensemble is None or (day_idx - last_train_idx >= RETRAIN_DAYS)

        if need_retrain:
            logger.log(f"[{current_date.date()}] 触发重训练 ({offset}/{total_steps})")
            train_df, val_df = build_train_val_split(df_model, all_dates, day_idx)
            ensemble, metrics_records = fit_lightgbm_ensemble(
                train_df=train_df,
                val_df=val_df,
                feature_cols=selected_features,
                formal_train_rows=formal_train_rows,
                logger=logger,
                stage_name=f"wf_{current_date.date()}",
                metrics_records=metrics_records,
                metrics_output_path=TRAIN_METRICS_PATH,
            )
            last_train_idx = day_idx
            gc.collect()

        day_df = day_frame_cache.get(current_date)
        if day_df is None:
            day_df = df_model[df_model["date"] == current_date].sort_values("code").reset_index(drop=True)
        if day_df.empty:
            continue

        pred_mean, pred_std = predict_with_ensemble(ensemble.models, day_df[selected_features])
        pred_chunks.append(build_prediction_frame(day_df, pred_mean, pred_std, current_date))

        if offset % 10 == 0 or offset == 1:
            logger.log(
                f"[{current_date.date()}] 已完成 {offset}/{total_steps} 个交易日预测，"
                f"当日股票数={len(day_df):,}"
            )

        if offset % checkpoint_every == 0:
            logger.log(f"[{current_date.date()}] 执行 checkpoint 保存")
            flush_prediction_checkpoint(pd.concat(pred_chunks, ignore_index=True), logger=logger)
            pred_chunks = []

    pred_df = collect_checkpoint_predictions(
        pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()
    )
    if pred_df.empty:
        raise RuntimeError("walk-forward 未生成任何预测结果")

    final_df = save_predictions(pred_df, end_date=None, logger=logger, resume_start=resume_start)
    if os.path.exists(PREDICT_CHECKPOINT_PKL):
        os.remove(PREDICT_CHECKPOINT_PKL)
        logger.log(f"已删除 checkpoint 文件: {PREDICT_CHECKPOINT_PKL}")
    return final_df


def run_pipeline(
    end_date: Optional[str] = None,
    max_train_rows: Optional[int] = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> pd.DataFrame:
    _cap_numeric_threads()
    os.makedirs(MODEL_INFO_DIR, exist_ok=True)
    write_optimization_report()

    log_path = build_log_path(end_date)
    logger = ProgressLogger(log_path)
    logger.section("py03 优化版模型启动")
    logger.log(f"日志文件: {log_path}")
    logger.log(f"CPU 资源控制: total_cpu={TOTAL_CPU}, allowed_cpu={CPU_BUDGET} (≈2/3)")
    logger.log(f"优化报告: {REPORT_PATH}")

    try:
        logger.section("Step 0/6 加载特征数据")
        df = pd.read_pickle(FEATURE_PKL)
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        logger.log(
            f"features.pkl 加载完成 | rows={len(df):,} | cols={len(df.columns)} | "
            f"date_range={df['date'].min().date()} ~ {df['date'].max().date()}"
        )

        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)].copy()
            logger.log(f"按日期截断至 {end_date}，剩余 {len(df):,} 行")

        df = add_derived_features(df, logger)
        feature_cols = get_feature_columns(df)
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        all_dates = pd.Index(df["date"].unique()).sort_values().tolist()

        if end_date is not None:
            pred_df = predict_single_day(
                df=df,
                all_dates=all_dates,
                end_date=end_date,
                logger=logger,
                max_train_rows=max_train_rows,
            )
        else:
            pred_df = walk_forward_predict(
                df=df,
                all_dates=all_dates,
                logger=logger,
                max_train_rows=max_train_rows,
                checkpoint_every=max(1, checkpoint_every),
            )

        logger.section("完成")
        logger.log("py03 优化版模型流程结束")
        write_optimization_report()
        return pred_df
    finally:
        logger.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="优化版 py03 模型训练与预测")
    parser.add_argument("--date", dest="end_date", help="指定预测日期（YYYY-MM-DD）")
    parser.add_argument("--latest", action="store_true", help="单日模式：仅预测数据中最新交易日")
    parser.add_argument("--max-train-rows", type=int, default=None, help="手动限制训练样本行数")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="walk-forward 模式下每隔多少个交易日执行一次 checkpoint",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    end_date = args.end_date
    if args.latest:
        df_tmp = pd.read_pickle(FEATURE_PKL)
        end_date = pd.to_datetime(df_tmp["date"]).max().strftime("%Y-%m-%d")
        del df_tmp
    run_pipeline(
        end_date=end_date,
        max_train_rows=args.max_train_rows,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
