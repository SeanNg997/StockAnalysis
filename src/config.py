"""
配置文件 - 外置化所有硬编码参数
================================

本文件包含项目中所有可配置的参数，避免硬编码在各个模块中。

使用方法：
1. 在需要使用配置的模块中导入：from config import CONFIG
2. 访问配置参数：CONFIG['section']['parameter']
"""

CONFIG = {
    # 数据获取配置
    'data_fetch': {
        'START_DATE': '2019-12-01',  # 数据开始日期
        'SAVE_EVERY': 200,  # 每下载多少只股票保存一次
        'MARKET_DATA_READY_HOUR': 18,  # 市场数据就绪时间（小时）
        'ADJUST_FLAG': '2', # 1：后复权；2：前复权；3：不复权
    },
    
    # 数据清洗配置
    'data_loader': {
        'MIN_AVG_AMOUNT': 500e4,  # 最低平均成交额（元）
        'MIN_TRADING_DAYS': 120,  # 每只股票最少交易日数
    },
    
    # 特征工程配置
    'features': {
        'HOLD_DAYS': 4,  # 持有天数（T+1买入 → T+5卖出）
        'BUY_COST': 1.0 + 0.000085,  # 买入成本（1 + 佣金率）
        'SELL_COST': 1.0 - 0.000585,  # 卖出成本（1 - 佣金率 - 印花税率）
        'LABEL_WINSORIZE_MIN': -0.30,  # 标签截断最小值
        'LABEL_WINSORIZE_MAX': 0.30,  # 标签截断最大值
    },
    
    # 模型配置
    'model': {
        'N_ENSEMBLE': 6,  # Ensemble模型数量
        'TRAIN_YEARS': 3,  # 训练窗口年数
        'RETRAIN_DAYS': 22,  # 每多少个交易日重新训练
        'BACKTEST_YEARS': 3,    # 回测窗口年数（从最新数据往前推）
        'HOLD_DAYS': 4,  # 持有天数（与features中的保持一致）
        'LGB_PARAMS': {
            'objective': 'Huber',  # Huber损失，对异常值更鲁棒
            'alpha': 0.7,  # Huber delta参数
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,  # 降低复杂度防过拟合
            'max_depth': 5,  # 限制树深度
            'num_boost_round': 3000,  # 最大boosting轮数
            'learning_rate': 0.01,  # 学习率
            'feature_fraction': 0.6,  # 特征采样比例
            'bagging_fraction': 0.7,  # 数据采样比例
            'bagging_freq': 5,
            'min_child_samples': 200,  # 叶节点最小样本数
            'lambda_l1': 1.0,  # L1正则
            'lambda_l2': 1.0,  # L2正则
            'min_gain_to_split': 0.01,  # 分裂最小增益
            'verbose': -1,
        },
    },
    
    # 回测配置
    'backtest': {
        'COMMISSION_RATE': 0.000085,  # 手续费费率
        'MIN_COMMISSION': 1.0,  # 每笔最低佣金
        'STAMP_TAX': 0.0005,  # 印花税（卖出时收取）
        'MAX_POSITIONS': 5,  # 最大持仓数
        'MIN_PRED_RETURN': 0.002,  # 最低预测收益率阈值
        'MIN_CONFIDENCE': 0.5,  # 最低置信度阈值
        'HOLD_DAYS': 4,  # 持有天数
        'STOP_LOSS': -0.05,  # 止损线
        'TAKE_PROFIT': 0.08,  # 止盈线
        'INITIAL_CAPITAL': 100_000,  # 初始资金
        'MAX_DAILY_BUY': 2,  # 每天最多买入数量
        'MARKET_REGIME_LOOKBACK': 10,  # 市场择时回看天数
    },
    
    # 路径配置
    'paths': {
        'BASE_DIR': None,  # 会在运行时自动设置
        'DATA_DIR': 'data',
        'OUTPUT_DIR': 'output',
        'BACKTEST_OUTPUT_DIR': 'output/backtest',
        'CLEAN_PKL': 'output/tmp/mainboard_clean.pkl',
        'FEATURE_PKL': 'output/tmp/features.pkl',
        'PREDICT_PKL': 'output/tmp/predictions.pkl',
    }
}

# 自动设置BASE_DIR
import os
CONFIG['paths']['BASE_DIR'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 构建完整路径
for key, value in CONFIG['paths'].items():
    if key != 'BASE_DIR' and isinstance(value, str) and not os.path.isabs(value):
        CONFIG['paths'][key] = os.path.join(CONFIG['paths']['BASE_DIR'], value)
