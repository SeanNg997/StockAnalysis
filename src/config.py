"""项目全局配置"""

CONFIG = {
    'data_fetch': {
        'START_DATE': '2019-12-01',
        'SAVE_EVERY': 200,
        'MARKET_DATA_READY_HOUR': 18,
        'ADJUST_FLAG': '3',  # 1:后复权 2:前复权 3:不复权
    },

    'features': {
        'HOLD_DAYS': 4,            # T+1买入 → T+5卖出
        'BUY_COST': 1.0 + 0.000085,
        'SELL_COST': 1.0 - 0.000585,
        'LABEL_WINSORIZE_MIN': -0.30,
        'LABEL_WINSORIZE_MAX': 0.30,
    },

    'model': {
        'TRAIN_YEARS': 3,
        'RETRAIN_DAYS': 22,
        'BACKTEST_START_YEAR': 2023,
        'HOLD_DAYS': 4,            # 须与 features.HOLD_DAYS 一致
        'LGB_PARAMS': {
            'objective': 'Huber',
            'alpha': 0.7,
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'deterministic': True,
            'force_col_wise': True,
            'verbose': -1,
        },
        'LGB_FORMAL': {
            'params': {
                'num_leaves': 39,
                'max_depth': 6,
                'learning_rate': 0.012,
                'feature_fraction': 0.78,
                'bagging_fraction': 0.82,
                'bagging_freq': 4,
                'min_child_samples': 220,
                'lambda_l1': 0.3,
                'lambda_l2': 1.2,
                'min_gain_to_split': 0.0,
            },
            'num_boost_round': 2600,
            'early_stopping_rounds': 180,
        },
        'LGB_FAST': {
            'params': {
                'num_leaves': 31,
                'max_depth': 5,
                'learning_rate': 0.03,
                'feature_fraction': 0.76,
                'bagging_fraction': 0.80,
                'bagging_freq': 4,
                'min_child_samples': 180,
                'lambda_l1': 0.2,
                'lambda_l2': 0.8,
                'min_gain_to_split': 0.0,
            },
            'num_boost_round': 360,
            'early_stopping_rounds': 60,
        },
        'LGB_FALLBACK': {
            'params': {
                'num_leaves': 31,
                'max_depth': 5,
                'learning_rate': 0.05,
                'feature_fraction': 0.88,
                'bagging_fraction': 0.88,
                'bagging_freq': 4,
                'min_child_samples': 48,
                'lambda_l1': 0.1,
                'lambda_l2': 0.2,
                'min_gain_to_split': 0.0,
            },
            'num_boost_round': 100,
            'early_stopping_rounds': None,
        },
    },

    'backtest': {
        'COMMISSION_RATE': 0.000085,
        'MIN_COMMISSION': 1.0,
        'STAMP_TAX': 0.0005,
        'MAX_POSITIONS': 15,
        'MIN_PRED_RETURN': 0.003,
        'MIN_CONFIDENCE': 0.50,
        'HOLD_DAYS': 9,
        'STOP_LOSS': -0.08,
        'TAKE_PROFIT': 0.15,
        'TRAILING_STOP_ACTIVATE': 0.05,   # 利润超过5%时激活追踪止盈
        'TRAILING_STOP_DRAWDOWN': 0.07,   # 从最高利润回落7%时触发卖出
        'INITIAL_CAPITAL': 100_000,
        'MAX_DAILY_BUY': 5,
        'MARKET_REGIME_LOOKBACK': 10,
        'MIN_EXEC_AMOUNT': 100e4,  # 执行日最低成交额(100万)
        'MAX_OPEN_TRADE_AMOUNT_RATIO': 0.02,  # 单票最多按最近5日平均成交额的2%成交
        'ALLOW_ST_BUY': True,                 # 是否允许买入 ST（实盘可按券商权限改为 False）
        'SPECIAL_LIMIT_GAP_TOL': 0.03,        # 开盘涨跌幅超过常规涨跌停+容差，视为特殊规则日（IPO/复牌等）
        'CORP_ACTION_TOL': 0.005,             # preclose 与前收偏离阈值，fallback 只在明显除权缺口时触发
        'MAX_DELIST_HOLD_DAYS': 5, # 持仓股停牌超过此天数视为退市，强制清仓
        'MIN_STOCK_PRICE': 3.0,    # 最低股价(元)，过滤低质量股
        'MIN_PRICE_DAYS': 5,       # 连续N个交易日低于MIN_STOCK_PRICE视为高风险
        'MIN_PRICE_CONSECUTIVE': True,  # True=连续N天，False=最近N天内任意N天
        'ENABLE_TREND_RISK_FILTER': True,  # 过滤无企稳信号的深度下跌票
        'TREND_RISK_RET_20D': -0.20,
        'TREND_RISK_RET_60D': -0.35,
        'TREND_RISK_DRAWDOWN_20D': -0.25,
        'TREND_RISK_MA_BIAS_20': -0.12,
        'TREND_STABILIZE_RET_5D': -0.02,
        'TREND_SCORE_PENALTY': 0.25,
    },

    'paths': {
        'BASE_DIR': None,  # 运行时自动设置
        'DATA_DIR': 'data',
        'OUTPUT_DIR': 'output',
        'BACKTEST_OUTPUT_DIR': 'output/backtest',
        'STOCK_LIST_CSV': 'output/tmp/mainboard_stocks.csv',
        'TRADE_DAYS_TXT': 'output/tmp/trade_days.txt',
        'CLEAN_PKL': 'output/tmp/mainboard_clean.pkl',
        'FEATURE_PKL': 'output/tmp/features.pkl',
        'PREDICT_PKL': 'output/tmp/predictions.pkl',
        'MARKET_PKL': 'output/tmp/market_status.pkl',
        'BACKTEST_MARKET_PKL': 'output/tmp/market_status_backtest.pkl',
        'ADJUST_FACTOR_PKL': 'data/adjust_factors.pkl',
        'DIVIDEND_PKL': 'data/dividend_events.pkl',
        'DATA_META_JSON': 'data/.dataset_meta.json',
    }
}

import os
CONFIG['paths']['BASE_DIR'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for key, value in CONFIG['paths'].items():
    if key != 'BASE_DIR' and isinstance(value, str) and not os.path.isabs(value):
        CONFIG['paths'][key] = os.path.join(CONFIG['paths']['BASE_DIR'], value)
