# StockAnalysis — A股量化交易策略系统

基于 LightGBM Ensemble + Walk-Forward 滚动训练的 A 股量化选股系统，支持每日自动化运行与本地网页控制台。

当前版本已经将价格体系改为：

- `py00` 持久化 `baostock` 的`不复权`原始行情
- `py02` 在原始行情上内部构造`点时稳定研究价`，用于特征和标签
- `py05` 回测执行继续使用原始开盘价，并在持仓层显式处理分红送转

这样做的目标是避免“未来新增分红后，历史前复权数据被整体重写，导致历史回测漂移”。

## 1 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# （首次升级到当前版本时务必先运行一次）
# 程序会自动识别旧版前复权 CSV，并切换到原始行情重建
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
- **对固定起始年份以来的历史进行回测评估**：点击 **一键任务** 中的 **完整回测流水线** 。
- **首次升级后的注意事项**：必须先让 `py00` 重新生成原始行情 CSV，旧版前复权 CSV 不能直接复用。
- **网页控制台兼容性**：现有网页控制台仍可直接用于当前版本，不需要结构性改版；它会按 `config.py` 里的 `model.BACKTEST_START_YEAR` 展示和刷新最新回测结果。

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
py00 原始行情/复权因子/分红事件
    → py01 清洗(raw)
    → py02 点时稳定研究价 + 特征/标签
    → py03 训练/预测
    → py04 策略决策
    └→ py05 原始价回测 + 分红送转处理
        └→ py06 报告
```

## 6 稳定回测说明

- 历史 CSV 现在以`不复权`价格为准，不再依赖会被未来分红重写的供应商前复权序列。
- 模型看到的 `open/high/low/close` 是项目内部从 `close / preclose / pctChg` 递推出的点时稳定研究价。
- 回测成交价仍然使用真实可交易的原始 `open`。
- `config.py` 中的 `model.BACKTEST_START_YEAR` 控制回测起点，例如设为 `2023` 就会从 `2023` 年首个交易日开始回测。
- 持仓遇到除权除息时，会优先按 `baostock.query_dividend_data` 的事件做现金分红和送转处理。
- 如果事件表缺失，但执行日 `preclose` 与前一日 raw close 存在明显除权差异，回测会退回到 `preclose` 缺口推导的合成调整，避免历史净值被打断。
- 网页控制台当前不会单独提供“回测起始年份”的输入框；如需调整回测起点，请修改 `config.py` 后重新运行相关流水线。

## 风险提示

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。回测结果不代表未来收益。
