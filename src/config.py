"""项目全局配置"""

CONFIG = {
    'data_fetch': {
        'START_DATE': '2019-12-01',
        'SAVE_EVERY': 200,
        'MARKET_DATA_READY_HOUR': 18,
        'ADJUST_FLAG': '2',  # 1:后复权 2:前复权 3:不复权
    },

    'data_loader': {
        'MIN_AVG_AMOUNT': 500e4,   # 最低20日均成交额(元)
        'MIN_TRADING_DAYS': 120,
    },

    'features': {
        'HOLD_DAYS': 4,            # T+1买入 → T+5卖出
        'BUY_COST': 1.0 + 0.000085,
        'SELL_COST': 1.0 - 0.000585,
        'LABEL_WINSORIZE_MIN': -0.30,
        'LABEL_WINSORIZE_MAX': 0.30,
    },

    'model': {
        'N_ENSEMBLE': 6,
        'TRAIN_YEARS': 3,
        'RETRAIN_DAYS': 22,
        'BACKTEST_YEARS': 3,
        'HOLD_DAYS': 4,            # 须与 features.HOLD_DAYS 一致
        'LGB_PARAMS': {
            'objective': 'Huber',
            'alpha': 0.7,
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 5,
            'num_boost_round': 3000,
            'learning_rate': 0.01,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 200,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'min_gain_to_split': 0.01,
            'verbose': -1,
        },
    },

    'backtest': {
        'COMMISSION_RATE': 0.000085,
        'MIN_COMMISSION': 1.0,
        'STAMP_TAX': 0.0005,
        'MAX_POSITIONS': 5,
        'MIN_PRED_RETURN': 0.002,
        'MIN_CONFIDENCE': 0.5,
        'HOLD_DAYS': 4,
        'STOP_LOSS': -0.05,
        'TAKE_PROFIT': 0.08,
        'INITIAL_CAPITAL': 100_000,
        'MAX_DAILY_BUY': 2,
        'MARKET_REGIME_LOOKBACK': 10,
        'MIN_EXEC_AMOUNT': 100e4,  # 执行日最低成交额(100万)
        'MIN_STOCK_PRICE': 2.0,    # 最低股价(元)，过滤退市/垃圾股
        'MAX_DELIST_HOLD_DAYS': 5, # 持仓股停牌超过此天数视为退市，强制清仓
        'MIN_PRICE_DAYS': 5,       # 连续N个交易日低于MIN_STOCK_PRICE视为高风险
        'MIN_PRICE_CONSECUTIVE': True,  # True=连续N天，False=最近N天内任意N天
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
    }
}

import os
CONFIG['paths']['BASE_DIR'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for key, value in CONFIG['paths'].items():
    if key != 'BASE_DIR' and isinstance(value, str) and not os.path.isabs(value):
        CONFIG['paths'][key] = os.path.join(CONFIG['paths']['BASE_DIR'], value)
