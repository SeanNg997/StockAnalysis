# StockAnalysis — A股量化交易策略系统

基于 LightGBM Ensemble + Walk-Forward 滚动训练的 A 股量化选股系统，支持每日自动化运行与本地网页控制台。

## 1 策略特性

- **模型**：LightGBM Ensemble（3个随机种子集成），Huber 损失函数
- **持有周期**：T+1 买入，持有4个交易日后卖出（T+5 开盘）
- **选股范围**：沪深主板（sh.60* / sz.00*），过滤停牌/新股/低流动性/低价风险（执行层允许 ST）
- **风控**：止损 -5%，止盈 +8%，每日最多建仓5只，最大持仓5只

## 2 快速开始

```bash
pip install -r requirements.txt

# 首次：全量下载数据（约数小时，下载本项目中的data文件夹后可以省去大部分时间）
python src/py00_fetch_stock_data.py

# 完整训练流水线（Walk-Forward + 回测 + 可视化）
./scripts/run_backtest.sh

# 日常：增量更新 + 快速预测（约7分钟）
./scripts/run_strategy.sh

# 启动本地网页控制台（按钮执行 + 实时日志 + 回测曲线）
./scripts/run_web_console.sh
```

启动网页控制台后，浏览器访问 `http://127.0.0.1:8000` 即可操作。

## 3 项目结构

```
src/
├── py00_fetch_stock_data.py   数据获取（baostock，全量/增量/断点续传）
├── py01_data_clean.py         数据清洗（主板过滤、流动性筛选、市场状态快照）
├── py02_features.py           特征工程（60+技术/动量/截面特征）
├── py03_model.py              LightGBM Ensemble 训练与预测（3-seed集成）
├── py04_today.py              今日交易决策（生成TOP5买入建议）
├── py05_backtest.py           回测引擎（T+1、涨跌停、止损止盈、交易成本）
└── py06_report.py             可视化报告（净值、回撤、热力图）
scripts/
├── run_strategy.sh            日常快速预测（增量更新 + 单日预测 + 策略报告）
├── run_backtest.sh            完整重训练（Walk-Forward + 回测 + 可视化）
└── run_web_console.sh         启动本地网页控制台
webapp/
├── server.py                  FastAPI 控制台后端（启动任务 / 转发日志 / 推送回测曲线）
└── static/                    控制台静态页面（单页仪表盘）
```

## 4 网页控制台

- 为 `py00` 到 `py06` 和两个 shell 流水线提供独立按钮
- 网页内实时显示脚本 `print` / 错误输出
- 运行 `py05_backtest.py` 或 `run_backtest.sh` 时，实时刷新收益曲线
- 自动读取最新一次 `output/backtest/backtest_daily.csv` 作为历史曲线基线

## 风险提示

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。回测结果不代表未来收益。
