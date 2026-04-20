# StockAnalysis — A股量化交易策略系统

基于 LightGBM Ensemble + Walk-Forward 滚动训练的 A 股量化选股系统，支持每日自动化运行与本地网页控制台。

## 1 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# （仅首次）全量下载数据（下载本项目中的 data 文件夹后可以省去大部分时间）
python src/py00_fetch_stock_data.py --full

# 启动本地网页控制台
# macOS / Linux（执行命令，或双击 .sh）
./scripts/run_web_console.sh
# Windows（执行命令，或双击 .bat）
scripts\run_web_console.bat
```

启动网页控制台后，浏览器访问 `http://127.0.0.1:8000` 即可操作。

## 2 日常使用

- **获取下个交易日的决策建议**：点击 **一键任务** 中的 **快速策略流水线** 。如果当前有持仓，需要在控制台右下角输入持仓信息并 **保存** 后再运行。
- **对近三年进行回测评估**：点击 **一键任务** 中的 **完整回测流水线** 。

## 3 命令行用法

```bash
# 增量更新数据
python src/py00_fetch_stock_data.py

# 数据清洗 → 特征工程 → 单日预测 → 生成策略
python src/py01_data_clean.py
python src/py02_features.py
python src/py03_model.py --date 2026-04-17
python src/py04_today.py

# 指定持仓文件生成策略（也可把 portfolio.json 放在 output/ 下自动检测）
python src/py04_today.py --portfolio output/portfolio.json

# 查看单只股票决策
python src/py04_today.py 002202

# Walk-Forward 全量训练 + 回测 + 报告
python src/py03_model.py
python src/py05_backtest.py
python src/py06_report.py
```

## 4 项目结构

```
src/
├── config.py                  全局配置（路径、模型参数、回测参数、交易规则阈值）
├── trading_rules.py           交易规则引擎（票池筛选、卖出决策、评分排序、仓位管理）
├── py00_fetch_stock_data.py   数据获取（baostock，全量/增量/断点续传）
├── py01_data_clean.py         数据清洗（主板过滤、流动性筛选、市场状态快照）
├── py02_features.py           特征工程（60+ 技术/动量/截面特征）
├── py03_model.py              LightGBM 3-seed Ensemble 训练与预测（Walk-Forward / 单日）
├── py04_today.py              今日交易决策（全市场策略 + 持仓分析 + 操作建议）
├── py05_backtest.py           回测引擎（T+1、涨跌停、止损止盈、交易成本）
└── py06_report.py             可视化报告（净值曲线、回撤、月度热力图）
scripts/
├── run_web_console.sh         macOS / Linux 启动本地网页控制台
├── run_web_console.bat        Windows 启动本地网页控制台
└── run_web_console.ps1        Windows 网页控制台启动脚本（PowerShell）
webapp/
├── server.py                  FastAPI 控制台后端（启动任务、日志推送、回测曲线）
└── static/                    控制台静态页面（单页仪表盘）
```

## 5 数据流

```
py00 数据获取 → py01 清洗 → py02 特征 → py03 训练/预测 → py04 策略决策
                                                      └→ py05 回测 → py06 报告
```

## 风险提示

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。回测结果不代表未来收益。
