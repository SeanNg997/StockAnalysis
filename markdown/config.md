# config.py 代码解释

## 文件概览

本文件是项目的配置文件，包含了所有可配置的参数，避免硬编码在各个模块中。

## 逐行代码解释

### 第1-10行：文件头部注释

```python
"""
配置文件 - 外置化所有硬编码参数
================================

本文件包含项目中所有可配置的参数，避免硬编码在各个模块中。

使用方法：
1. 在需要使用配置的模块中导入：from config import CONFIG
2. 访问配置参数：CONFIG['section']['parameter']
"""
```
- **第1-10行**：文件头部的文档字符串，说明该文件的作用是作为配置文件，外置化所有硬编码参数，并提供了使用方法。

### 第12-89行：CONFIG 字典定义

```python
CONFIG = {
    # 数据获取配置
    'data_fetch': {
        'START_DATE': '2019-12-01',  # 数据开始日期
        'SAVE_EVERY': 200,  # 每下载多少只股票保存一次
        'MARKET_DATA_READY_HOUR': 18,  # 市场数据就绪时间（小时）
        'ADJUST_FLAG': '2',
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
```
- **第12行**：定义了一个名为 CONFIG 的字典，用于存储所有配置参数。
- **第14-19行**：`data_fetch` 部分，包含数据获取相关的配置：
  - **第15行**：`START_DATE`：数据开始日期，设置为 '2019-12-01'。
  - **第16行**：`SAVE_EVERY`：每下载多少只股票保存一次，设置为 200。
  - **第17行**：`MARKET_DATA_READY_HOUR`：市场数据就绪时间（小时），设置为 18。
  - **第18行**：`ADJUST_FLAG`：调整标志，设置为 '2'。
- **第22-25行**：`data_loader` 部分，包含数据清洗相关的配置：
  - **第23行**：`MIN_AVG_AMOUNT`：最低平均成交额（元），设置为 500e4（500万元）。
  - **第24行**：`MIN_TRADING_DAYS`：每只股票最少交易日数，设置为 120。
- **第28-34行**：`features` 部分，包含特征工程相关的配置：
  - **第29行**：`HOLD_DAYS`：持有天数，设置为 4（T+1买入 → T+5卖出）。
  - **第30行**：`BUY_COST`：买入成本（1 + 佣金率），设置为 1.0 + 0.000085。
  - **第31行**：`SELL_COST`：卖出成本（1 - 佣金率 - 印花税率），设置为 1.0 - 0.000585。
  - **第32行**：`LABEL_WINSORIZE_MIN`：标签截断最小值，设置为 -0.30。
  - **第33行**：`LABEL_WINSORIZE_MAX`：标签截断最大值，设置为 0.30。
- **第37-61行**：`model` 部分，包含模型相关的配置：
  - **第38行**：`N_ENSEMBLE`：Ensemble模型数量，设置为 6。
  - **第39行**：`TRAIN_YEARS`：训练窗口年数，设置为 3。
  - **第40行**：`RETRAIN_DAYS`：每多少个交易日重新训练，设置为 22。
  - **第41行**：`BACKTEST_YEARS`：回测窗口年数（从最新数据往前推），设置为 3。
  - **第42行**：`HOLD_DAYS`：持有天数（与features中的保持一致），设置为 4。
  - **第43-60行**：`LGB_PARAMS`：LightGBM模型的参数配置：
    - **第44行**：`objective`：目标函数，设置为 'Huber'，对异常值更鲁棒。
    - **第45行**：`alpha`：Huber delta参数，设置为 0.7。
    - **第46行**：`metric`：评估指标，设置为 'mae'（平均绝对误差）。
    - **第47行**：`boosting_type`：提升类型，设置为 'gbdt'（梯度提升决策树）。
    - **第48行**：`num_leaves`：叶子节点数，设置为 31，降低复杂度防过拟合。
    - **第49行**：`max_depth`：树的最大深度，设置为 5，限制树深度。
    - **第50行**：`num_boost_round`：最大boosting轮数，设置为 3000。
    - **第51行**：`learning_rate`：学习率，设置为 0.01。
    - **第52行**：`feature_fraction`：特征采样比例，设置为 0.6。
    - **第53行**：`bagging_fraction`：数据采样比例，设置为 0.7。
    - **第54行**：`bagging_freq`：bagging的频率，设置为 5。
    - **第55行**：`min_child_samples`：叶节点最小样本数，设置为 200。
    - **第56行**：`lambda_l1`：L1正则化参数，设置为 1.0。
    - **第57行**：`lambda_l2`：L2正则化参数，设置为 1.0。
    - **第58行**：`min_gain_to_split`：分裂最小增益，设置为 0.01。
    - **第59行**：`verbose`： verbose模式，设置为 -1（不输出详细信息）。
- **第64-77行**：`backtest` 部分，包含回测相关的配置：
  - **第65行**：`COMMISSION_RATE`：手续费费率，设置为 0.000085。
  - **第66行**：`MIN_COMMISSION`：每笔最低佣金，设置为 1.0。
  - **第67行**：`STAMP_TAX`：印花税（卖出时收取），设置为 0.0005。
  - **第68行**：`MAX_POSITIONS`：最大持仓数，设置为 5。
  - **第69行**：`MIN_PRED_RETURN`：最低预测收益率阈值，设置为 0.002。
  - **第70行**：`MIN_CONFIDENCE`：最低置信度阈值，设置为 0.5。
  - **第71行**：`HOLD_DAYS`：持有天数，设置为 4。
  - **第72行**：`STOP_LOSS`：止损线，设置为 -0.05（亏损5%）。
  - **第73行**：`TAKE_PROFIT`：止盈线，设置为 0.08（盈利8%）。
  - **第74行**：`INITIAL_CAPITAL`：初始资金，设置为 100_000（10万元）。
  - **第75行**：`MAX_DAILY_BUY`：每天最多买入数量，设置为 2。
  - **第76行**：`MARKET_REGIME_LOOKBACK`：市场择时回看天数，设置为 10。
- **第80-88行**：`paths` 部分，包含路径相关的配置：
  - **第81行**：`BASE_DIR`：基础目录，设置为 None，会在运行时自动设置。
  - **第82行**：`DATA_DIR`：数据目录，设置为 'data'。
  - **第83行**：`OUTPUT_DIR`：输出目录，设置为 'output'。
  - **第84行**：`BACKTEST_OUTPUT_DIR`：回测输出目录，设置为 'output/backtest'。
  - **第85行**：`CLEAN_PKL`：清洗后的数据文件路径，设置为 'output/tmp/mainboard_clean.pkl'。
  - **第86行**：`FEATURE_PKL`：特征数据文件路径，设置为 'output/tmp/features.pkl'。
  - **第87行**：`PREDICT_PKL`：预测结果文件路径，设置为 'output/tmp/predictions.pkl'。

### 第91-98行：路径处理

```python
# 自动设置BASE_DIR
import os
CONFIG['paths']['BASE_DIR'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 构建完整路径
for key, value in CONFIG['paths'].items():
    if key != 'BASE_DIR' and isinstance(value, str) and not os.path.isabs(value):
        CONFIG['paths'][key] = os.path.join(CONFIG['paths']['BASE_DIR'], value)
```
- **第92行**：导入 os 模块，用于路径操作。
- **第93行**：自动设置 BASE_DIR，通过 `os.path.abspath(__file__)` 获取当前文件的绝对路径，然后通过 `os.path.dirname()` 两次获取上两级目录，即项目的根目录。
- **第96-98行**：遍历 paths 字典中的所有键值对，对于非 BASE_DIR 且值为字符串且不是绝对路径的键，将其值转换为基于 BASE_DIR 的完整路径。
  - **第97行**：条件判断：键不是 'BASE_DIR'，值是字符串类型，且不是绝对路径。
  - **第98行**：使用 `os.path.join()` 将 BASE_DIR 与当前值拼接，形成完整路径。