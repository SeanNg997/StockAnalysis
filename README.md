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
# 1) 增量更新数据
python src/py00_fetch_stock_data.py

# 2) 数据清洗与特征工程
python src/py01_data_clean.py
python src/py02_features.py

# 3) 单日预测（用于查看某个交易日的模型输出，不会跑完整回测）
python src/py03_model.py --date 2026-04-17

# 4) 生成下个交易日策略建议
python src/py04_today.py

# 指定持仓文件生成策略（也可把 portfolio.json 放在 output/ 下自动检测）
python src/py04_today.py --portfolio output/portfolio.json

# 查看单只股票决策
python src/py04_today.py 002202

# 5) Walk-Forward 训练/预测
# 第一次运行：会从 BACKTEST_START_YEAR 开始全量生成 output/tmp/predictions.pkl
python src/py03_model.py

# 后续再次运行：如果 predictions.pkl 已存在，会自动识别最后已完成日期，
# 只从下一交易日继续增量生成后面的预测，不会覆盖旧日期前缀
python src/py03_model.py

# 6) 回测
# 第一次运行：全量回测，并生成 backtest_state.json
python src/py05_backtest.py

# 后续再次运行：
# - 若共同区间预测未变化，会读取 backtest_state.json 中的现金/持仓状态，
#   从下一交易日继续增量回测；
# - 若共同区间预测发生变化，会直接报错，要求先全量重跑训练/回测
python src/py05_backtest.py

# 7) 生成报告
python src/py06_report.py
```

### 3.1 推荐的完整使用流程

#### 场景 A：首次全量运行

```bash
python src/py00_fetch_stock_data.py --full
python src/py01_data_clean.py
python src/py02_features.py
python src/py03_model.py
python src/py05_backtest.py
python src/py06_report.py
```

适用于：
- 首次使用项目
- 切换到当前“原始行情 + 点时研究价”版本后第一次重建
- 需要从头生成完整预测与完整回测结果

#### 场景 B：日常增量更新

```bash
python src/py00_fetch_stock_data.py
python src/py01_data_clean.py
python src/py02_features.py
python src/py03_model.py
python src/py05_backtest.py
python src/py06_report.py
```

适用于：
- 已经跑过一次全量
- 新增了后续几个交易日的数据
- 希望只补后续日期，而不是从头重跑整个训练和回测

#### 场景 C：只看某天预测，不做完整回测

```bash
python src/py01_data_clean.py
python src/py02_features.py
python src/py03_model.py --date 2026-04-17
python src/py04_today.py --portfolio output/portfolio.json
```

适用于：
- 只查看指定交易日的模型预测
- 只生成下一交易日的操作建议
- 不需要完整历史回测

### 3.2 增量训练 / 增量回测的工作方式

#### py03 增量训练

`src/py03_model.py` 当前支持自动续跑：

- 第一次运行时，会从 `config.py` 中 `model.BACKTEST_START_YEAR` 指定的起点开始，完整跑一遍 Walk-Forward，输出：
  - `output/tmp/predictions.pkl`
  - `output/tmp/predictions_checkpoint.pkl`（中间 checkpoint，结束后会自动清理）
- 第二次及之后运行时，如果 `output/tmp/predictions.pkl` 已存在：
  - 程序会自动识别历史预测已覆盖到哪一天；
  - 从下一交易日继续生成后续预测；
  - 保存时只拼接新日期，不覆盖旧日期前缀。

注意：
- 这不是 LightGBM 模型参数级别的“继续训练”；
- 而是“目标日期级别的增量更新”：旧日期预测保留，只为新增日期重新做 Walk-Forward 训练与预测。

#### py05 增量回测

`src/py05_backtest.py` 当前支持真正基于历史持仓状态续跑：

- 第一次运行时：
  - 从回测起点开始完整回放；
  - 输出：
    - `output/backtest/backtest_daily.csv`
    - `output/backtest/trade_log.csv`
    - `output/backtest/position_log.csv`
    - `output/backtest/corp_action_log.csv`
    - `output/backtest/backtest_metrics.md`
    - `output/backtest/backtest_state.json`
- 后续运行时：
  - 会先检查共同区间预测是否与上次一致；
  - 如果一致，则读取 `backtest_state.json` 中保存的：
    - 当前现金
    - 当前持仓
    - 每个持仓的成本、持有天数、累计分红等状态
  - 然后从下一交易日继续回放新增区间；
  - 最后把旧结果前缀保留，把新增区间结果拼接到后面。

这意味着增量回测不是“默认空仓重新买 5 只”，而是基于上次回测结束时的真实持仓继续推进。

### 3.3 什么时候必须全量重跑

出现以下情况时，不建议直接走增量：

1. 修改了模型参数、交易规则、回测参数或特征逻辑；
2. 历史行情、市场状态、分红送转数据发生修正；
3. `py05_backtest.py` 检测到共同区间预测发生变化；
4. 想确认新的代码版本对整个历史区间的完整影响。

此时建议删除旧产物后重新全量运行：

```bash
python src/py00_fetch_stock_data.py --full   # 只有需要重拉全量原始行情时才执行
python src/py01_data_clean.py
python src/py02_features.py
python src/py03_model.py
python src/py05_backtest.py
python src/py06_report.py
```

### 3.4 关键输出文件说明

#### 训练 / 预测输出

- `output/tmp/features.pkl`：特征数据
- `output/tmp/predictions.pkl`：Walk-Forward 预测结果
- `output/tmp/predictions_checkpoint.pkl`：训练中间 checkpoint
- `output/model_selection/`：训练日志、指标、特征选择结果

#### 回测输出

- `output/backtest/backtest_daily.csv`：每日资产与持仓数
- `output/backtest/trade_log.csv`：交易日志
- `output/backtest/position_log.csv`：每日持仓快照
- `output/backtest/corp_action_log.csv`：分红送转处理日志
- `output/backtest/backtest_metrics.md`：回测指标摘要
- `output/backtest/backtest_state.json`：增量回测续跑状态

#### `backtest_state.json` 的用途

这个文件是增量回测的关键状态文件，用于保存上次回测结束时的：
- 现金
- 当前持仓
- 持仓成本
- 持仓股数
- 持有天数
- 累计现金分红
- 最近一个决策日 / 执行日

如果这个文件丢失，程序就无法从历史持仓状态继续补跑，只能重新做全量回测。

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
