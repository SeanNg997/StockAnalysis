# StockAnalysis 项目详细文档

> A股量化交易策略系统 — 基于 LightGBM Ensemble + Walk-Forward 训练

---

## 目录

1. [项目架构](#1-项目架构)
2. [数据流程](#2-数据流程)
3. [模块详解](#3-模块详解)
   - [py00 — 数据获取](#py00--数据获取)
   - [py01 — 数据清洗](#py01--数据清洗)
   - [py02 — 特征工程](#py02--特征工程)
   - [py03 — 模型训练](#py03--模型训练)
   - [py04 — 今日决策](#py04--今日决策)
   - [py05 — 回测引擎](#py05--回测引擎)
   - [py06 — 可视化报告](#py06--可视化报告)
4. [Shell 脚本](#4-shell-脚本)
5. [GitHub Actions 自动化](#5-github-actions-自动化)
6. [特征工程详解](#6-特征工程详解)
7. [模型设计详解](#7-模型设计详解)
8. [回测设计详解](#8-回测设计详解)
9. [交易规则与成本](#9-交易规则与成本)
10. [输出文件说明](#10-输出文件说明)
11. [依赖与环境](#11-依赖与环境)
12. [常见问题排查](#12-常见问题排查)

---

## 1. 项目架构

```
StockAnalysis/
├── src/                                    # 核心 Python 模块
│   ├── py00_fetch_stock_data.py            #   数据获取（baostock）
│   ├── py01_data_loader.py                 #   数据清洗与过滤
│   ├── py02_features.py                    #   特征工程（60+特征）
│   ├── py03_model.py                       #   LightGBM 训练与预测（并行）
│   ├── py04_today.py                       #   今日交易决策
│   ├── py05_backtest.py                    #   回测引擎
│   └── py06_report.py                      #   可视化报告
├── scripts/
│   └── run_strategy.sh                        #   日常快速预测脚本
│   └── run_backtest.sh                        #   完整重训练脚本
├── data/                                   # 数据目录（.gitignore，通过 Actions cache 持久化）
│   ├── Stock_dailyK_YYYYMM.csv            #   月度K线数据（~140个文件）
│   ├── mainboard_clean.pkl                 #   清洗后的主板数据
│   ├── features.pkl                        #   特征工程数据集
│   ├── predictions.pkl                     #   模型预测结果
│   ├── .last_date.txt                      #   最后更新日期
│   └── .download_status.txt               #   下载进度追踪
├── output/                                 # 输出目录
│   ├── trading_strategy.md                   #   当日策略报告
│   ├── trading_strategy.html                 #   当日策略报告（HTML，供邮件使用）
│   ├── backtest/                           #   回测输出目录
│   │   ├── backtest_daily.csv              #     每日回测结果
│   │   ├── trade_log.csv                   #     交易流水
│   │   ├── backtest_metrics.md             #     回测绩效指标
│   │   ├── equity_curve.png                #     净值曲线图
│   │   ├── drawdown_curve.png              #     最大回撤图
│   │   ├── daily_positions.png             #     每日持仓图
│   │   ├── monthly_returns.png             #     月度收益热力图
│   │   └── feature_importance.png          #     特征重要性图
│   └── history/                            #   历史策略存档
├── .github/workflows/
│   └── daily_report.yml                    #   GitHub Actions 自动化工作流
├── requirements.txt                        #   Python 依赖
└── README.md                               #   项目简介
```

---

## 2. 数据流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         完整数据流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  py00_fetch_stock_data.py                                           │
│    ├─ 数据源：baostock API（免费，覆盖近6年A股日K线）                │
│    ├─ 股票范围：沪深主板（sh.60* / sz.00*），约5000只               │
│    ├─ 支持全量下载（首次）和增量更新（日常）                         │
│    └─ 输出：data/Stock_dailyK_YYYYMM.csv（月度拆分）                │
│                          ↓                                          │
│  py01_data_loader.py                                                │
│    ├─ 加载所有月度 CSV，合并为统一 DataFrame                         │
│    ├─ 过滤：仅保留主板（sh.60* / sz.00*）                            │
│    ├─ 剔除：ST股、停牌股、新股（上市5日内）                           │
│    ├─ 流动性筛选：20日均成交额 ≥ 500万元                             │
│    ├─ 缺失值处理：按股票前向填充（ffill）                            │
│    ├─ 最小历史要求：每只股票至少有120个交易日                         │
│    └─ 输出：data/mainboard_clean.pkl                                │
│                          ↓                                          │
│  py02_features.py                                                   │
│    ├─ 计算60+技术面、动量、截面特征                                   │
│    ├─ 标签：T+1开盘买入→T+6开盘卖出，扣除手续费+印花税（HOLD_DAYS=5）  │
│    └─ 输出：data/features.pkl                                       │
│                          ↓                                          │
│  py03_model.py                                                      │
│    ├─ Walk-Forward 滚动训练（每22个交易日重训一次）                   │
│    ├─ 或 单日快速预测（以指定日期前3年数据训练，预测当天）              │
│    ├─ 模型：LightGBM Ensemble（6个不同随机种子并行训练）              │
│    └─ 输出：data/predictions.pkl（含pred_return、pred_std、confidence）│
│                          ↓                    ↓                     │
│  py04_today.py                      py05_backtest.py                │
│    ├─ 读取最新预测结果                ├─ 模拟真实交易                  │
│    ├─ 过滤：pred_return>0.2%         ├─ T+1执行、涨跌停检测            │
│    │   且confidence>0.5              ├─ 止损-5%、止盈+8%               │
│    ├─ 输出TOP5买入建议                ├─ 最多5只持仓，每日限购2只        │
│    └─ 输出trading_strategy.md          └─ 输出回测报告+交易流水           │
│                          ↓                                          │
│  py06_report.py                                                     │
│    └─ 生成可视化图表（净值、回撤、持仓、热力图、特征重要性）            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 模块详解

### py00 — 数据获取

**文件**：`src/py00_fetch_stock_data.py`

**核心功能**：

从 [baostock](http://baostock.com) 下载 A 股日 K 线数据，支持全量下载与断点续传增量更新。

**主要函数**：

| 函数                                               | 说明                                       |
| -------------------------------------------------- | ------------------------------------------ |
| `get_stock_list()`                               | 获取所有主板股票代码（sh.60* / sz.00*）    |
| `get_expected_latest_date()`                     | 根据当前时间和交易日历确定最新可用数据日期 |
| `fetch_daily_full(symbol, name)`                 | 全量下载指定股票完整历史日K线              |
| `fetch_daily_increment(symbol, name, from_date)` | 增量下载：从指定日期到今日                 |
| `load_existing_csv()`                            | 加载所有已下载的月度 CSV 文件              |
| `save_to_csv(df)`                                | 将数据按月拆分保存为 CSV（全量覆盖）       |
| `save_incremental_months(new_df)`                | 增量更新月度 CSV 文件（仅写入变化的月份）  |

**数据字段**（原始 CSV 列）：

| 字段          | 说明                     |
| ------------- | ------------------------ |
| `date`      | 交易日期（YYYY-MM-DD）   |
| `代码`      | 股票代码（如 sh.600000） |
| `名称`      | 股票名称                 |
| `open`      | 开盘价                   |
| `high`      | 最高价                   |
| `low`       | 最低价                   |
| `close`     | 收盘价                   |
| `volume`    | 成交量（手）             |
| `amount`    | 成交额（元）             |
| `turn`      | 换手率（%）              |
| `pctChg`    | 涨跌幅（%）              |
| `peTTM`     | 滚动市盈率               |
| `pbMRQ`     | 市净率                   |
| `psTTM`     | 滚动市销率               |
| `pcfNcfTTM` | 市现率                   |

**数据就绪时���逻辑**：

`get_expected_latest_date()` 通过查询 baostock 交易日历判断当前最新可用数据：

- 当前时间 ≥ 18:00 且今天是交易日 → 预期最新数据为今天
- 当前时间 < 18:00 或今天是非交易日 → 预期最新数据为上一个交易日

**运行方式**：

```bash
# 增量更新（默认模式，日常使用）
python src/py00_fetch_stock_data.py

# 全量下载（首次，约30-60分钟）
python src/py00_fetch_stock_data.py --full

# 测试模式（仅下载前10只股票）
python src/py00_fetch_stock_data.py -n 10
```

**特性**：

- **断点续传**：利用 `.download_status.txt` 记录已下载的股票，中断后可继续
- **月度拆分**：按 `YYYYMM` 拆分为独立 CSV，便于增量更新
- **进度自动保存**：每处理 200 只股票自动保存一次

---

### py01 — 数据清洗

**文件**：`src/py01_data_loader.py`

**处理流程**：

1. 加载所有 `data/Stock_dailyK_*.csv` 文件
2. 合并为统一 DataFrame，解析日期
3. 过滤：保留 `sh.60*` 和 `sz.00*` 开头的主板股票
4. 剔除名称中含 ST / *ST 的股票
5. 剔除成交量为 0 的停牌日和涨跌幅缺失的行
6. 剔除每只股票上市后前5个交易日（避免新股效应）
7. 按股票前向填充缺失行情数据
8. 流动性筛选：计算20日滚动平均成交额，剔除低于500万元的记录
9. 最小历史约束：剔除历史天数不足120日的股票
10. 保存为 `data/mainboard_clean.pkl`

**缓存命中**：若 pkl 已存在且最新日期与目标日期一致，直接复用，跳过重新生成。

**运行方式**：

```bash
# 全量处理
python src/py01_data_loader.py

# 截止到指定日期（用于历史回测或单日预测）
python src/py01_data_loader.py --date 2025-03-14
```

---

### py02 — 特征工程

**文件**：`src/py02_features.py`

**输入**：`data/mainboard_clean.pkl`

**输出**：`data/features.pkl`

**预测标签**：

```python
# T日决策 → T+1开盘买入 → T+6开盘卖出（持有5个交易日）
label = open[t+6] / open[t+1] - 1 - buy_commission - sell_commission - stamp_tax
```

详细的特征说明见[第6节 特征工程详解](#6-特征工程详解)。

**运行方式**：

```bash
# 全量特征
python src/py02_features.py

# 截止到指定日期
python src/py02_features.py --date 2025-03-14
```

---

### py03 — 模型训练

**文件**：`src/py03_model.py`

**输入**：`data/features.pkl`

**输出**：`data/predictions.pkl`

详细的模型设计见[第7节 模型设计详解](#7-模型设计详解)。

**运行方式**：

```bash
# Walk-Forward 全量训练（用于回测，需数分钟到数十分钟）
python src/py03_model.py

# 单日快速预测（用于当日决策，约1-3分钟）
python src/py03_model.py --date 2025-03-14
```

**predictions.pkl 字段**：

| 字段            | 说明                                  |
| --------------- | ------------------------------------- |
| `date`        | 预测日期                              |
| `代码`        | 股票代码                              |
| `pred_return` | 预测 T+1→T+6 净收益率                |
| `pred_std`    | 6模型预测标准差（不确定性）           |
| `confidence`  | 置信度分数（1 / (1 + pred_std×100)） |

---

### py04 — 今日决策

**文件**：`src/py04_today.py`

**核心功能**：

读取 `predictions.pkl` 和 `features.pkl`，生成当日交易建议。

**选股过滤条件**：

- `pred_return > 0.2%` — 预测净收益率必须为正且超过门槛
- `confidence > 0.5` — 置信度达标（5模型分歧小）

**两种模式**：

**1. 全市场报告** (`generate_today_strategy`)

- 输出 TOP5 买入推荐（含股票名称、代码、预测收益率、置信度等）
- 包含市场整体概览（涨跌家数、平均预测收益、行情热度）
- 输出完整 TOP20 排名表格

**2. 单股报告** (`generate_stock_report(code)`)

- 该股票当前关键指标（PE、PB、换手率等）
- 5日预测趋势
- 与市场对比（相对强弱）
- 投资建议（强烈买入 / 买入 / 谨慎 / 观察 / 回避）

**运行方式**：

```bash
# 全市场报告
python src/py04_today.py

# 单只股票报告
python src/py04_today.py 600000

# 指定历史日期
python src/py04_today.py --date 2025-03-14

# 历史日期 + 单只股票
python src/py04_today.py --date 2025-03-14 600000
```

**输出**：

- `output/trading_strategy.md` — 全市场策略报告（Markdown 格式）
- `output/trading_strategy_{code}.md` — 单股报告
- `output/history/strategy_{date}.md` — 历史策略自动存档

---

### py05 — 回测引擎

**文件**：`src/py05_backtest.py`

详细设计见[第8节 回测设计详解](#8-回测设计详解)。

**策略参数**：

| 参数                | 值         | 说明                         |
| ------------------- | ---------- | ---------------------------- |
| `INITIAL_CAPITAL` | 100,000 元 | 初始资金 10 万               |
| `MAX_POSITIONS`   | 5          | 最大持仓股票数               |
| `MAX_DAILY_BUY`   | 2          | 每天最多买入只数（分批建仓） |
| `HOLD_DAYS`       | 5          | 目标持有天数                 |
| `MIN_PRED_RETURN` | 0.2%       | 最低预测收益率门槛           |
| `MIN_CONFIDENCE`  | 0.5        | 最低置信度门槛               |
| `STOP_LOSS`       | -5%        | 止损线                       |
| `TAKE_PROFIT`     | +8%        | 止盈线                       |

**运行方式**：

```bash
python src/py05_backtest.py
```

**输出文件**：

- `output/backtest/backtest_daily.csv` — 每日投资组合净值、现金、持仓数量、交易次数
- `output/backtest/trade_log.csv` — 完整交易流水（买入/卖出价格、数量、手续费、P&L）
- `output/backtest/backtest_metrics.md` — 回测绩效摘要

---

### py06 — 可视化报告

**文件**：`src/py06_report.py`

**生成的图表**：

| 图表       | 文件名                                     | 说明                               |
| ---------- | ------------------------------------------ | ---------------------------------- |
| 净值曲线   | `output/backtest/equity_curve.png`       | 策略净值 vs 基准（持有现金）对比   |
| 最大回撤   | `output/backtest/drawdown_curve.png`     | 从历史高点的回撤百分比             |
| 每日持仓   | `output/backtest/daily_positions.png`    | 持仓数量 + 每日买卖笔数            |
| 月度热力图 | `output/backtest/monthly_returns.png`    | 按月/年分布的收益率（RdYlGn 色系） |
| 特征重要性 | `output/backtest/feature_importance.png` | LightGBM Gain 排名，TOP25 特征     |

**运行方式**：

```bash
python src/py06_report.py
```

---

## 4. Shell 脚本

### `scripts/run_strategy.sh` — 日常快速预测

**典型耗时**：约30秒至3分钟

```bash
./scripts/run_strategy.sh                            # 当日全市场报告
./scripts/run_strategy.sh --date 2025-03-14          # 历史日期报告
```

**内部流程**：

```
Step 1: 增量数据更新（若指定 --date 则跳过）
Step 2: 数据清洗（截止到指定日期，缓存命中则跳过）
Step 3: 特征工程（截止到指定日期，缓存命中则跳过）
Step 4: 单日快速预测（以指定日期前3年数据训练）
Step 5: 生成策略报告
```

**缓存检测**：脚本会检查 pkl 文件的最大日期是否与目标日期一致，命中则自动跳过清洗/特征计算步骤。

---

### `scripts/run_backtest.sh` — 完整重训练

**典型耗时**：约5-30分钟（Walk-Forward 全量训练）

```bash
./scripts/run_backtest.sh
```

**内部流程**：

```
Step 1: 增量数据更新
Step 2: 全量数据清洗
Step 3: 全量特征工程
Step 4: Walk-Forward 滚动训练（每22个交易日重训一次）
Step 5: 回测模拟 + 生成回测报告
Step 6: 生成可视化图表
```

建议**每周或每月**运行一次，以更新模型的历史预测基准。

---

## 5. GitHub Actions 自动化

**文件**：`.github/workflows/daily_report.yml`

### 触发时间

- **定时**：UTC 11:30（北京时间 19:30），`cron: '30 11 * * 1-5'`（周一至周五）
- **手动**：支持 `workflow_dispatch` 在任意时间手动触发

### 超时

90 分钟

### 工作流步骤

```
1. Checkout 仓库代码
2. 设置 Python 3.11 环境
3. 缓存 pip 依赖包（加速后续安装）
4. 安装依赖：pip install -r requirements.txt
5. 恢复数据缓存（pkl 文件 + 月度 CSV）
6. 检测是否为交易日（周末跳过）
7. 判断是首次运行（无 CSV 缓存）还是增量更新
   - 无缓存：全量下载历史数据
   - 有缓存：仅增量更新最新交易日
8. 数据清洗（--date 当前日期）
9. 特征工程（--date 当前日期）
10. 单日快速预测（--date 当前日期）
11. 生成策略报告（Markdown）
12. 将策略报告转换为 HTML 邮件格式
13. 通过 dawidd6/action-send-mail 发送邮件（若配置了 Secrets）
```

### 缓存策略

使用 `actions/cache` 缓存以下文件（跨运行持久化）：

| 文件                          | 缓存目的                       |
| ----------------------------- | ------------------------------ |
| `data/mainboard_clean.pkl`  | 避免每次重新清洗全量数据       |
| `data/features.pkl`         | 避免每次重新计算特征           |
| `data/predictions.pkl`      | 保留历史预测供复盘对比         |
| `data/.last_date.txt`       | 记录最后更新日期，用于增量检测 |
| `data/Stock_dailyK_*.csv`   | 月度K线文件缓存，节省网络流量  |
| `data/.download_status.txt` | 下载进度追踪，支持断点续传     |

### 邮件推送配置

在 **GitHub → Settings → Secrets and variables → Actions** 中配置：

| Secret 名           | 说明                   | 示例值                    |
| ------------------- | ---------------------- | ------------------------- |
| `EMAIL_SERVER`    | SMTP 服务器地址        | `smtp.gmail.com`        |
| `EMAIL_PORT`      | SMTP 端口              | `587`                   |
| `EMAIL_USERNAME`  | 发件人账户             | `your@gmail.com`        |
| `EMAIL_PASSWORD`  | 应用密码（非登录密码） | `xxxx xxxx xxxx xxxx`   |
| `EMAIL_RECIPIENT` | 收件人邮箱             | `recipient@example.com` |

> **注意**：Gmail 需要开启两步验证并生成「应用专用密码」，不能使用账户密码。

### 首次启用

**推荐**：直接在 GitHub Actions 页面手动触发 workflow

- GitHub → Actions → Daily Stock Strategy Report → Run workflow
- 首次运行自动执行全量数据下载（约30-60分钟）

---

## 6. 特征工程详解

共 **60+ 个特征**，分为以下类别：

### 价格动量类

| 特征名         | 计算方式                      | 说明          |
| -------------- | ----------------------------- | ------------- |
| `ret_1d`     | `close/close.shift(1) - 1`  | 1日收益率     |
| `ret_3d`     | `close/close.shift(3) - 1`  | 3日收益率     |
| `ret_5d`     | `close/close.shift(5) - 1`  | 5日收益率     |
| `ret_10d`    | `close/close.shift(10) - 1` | 10日收益率    |
| `ret_20d`    | `close/close.shift(20) - 1` | 20日收益率    |
| `ret_60d`    | `close/close.shift(60) - 1` | 60日收益率    |
| `log_ret_1d` | `log(close/close.shift(1))` | 1日对数收益率 |

### 均线系统类

| 特征名                                                        | 说明                                                            |
| ------------------------------------------------------------- | --------------------------------------------------------------- |
| `ma_5`, `ma_10`, `ma_20`, `ma_60`                     | N日简单移动平均价格（不进入模型，仅用于计算偏离度）             |
| `ma_bias_5`, `ma_bias_10`, `ma_bias_20`, `ma_bias_60` | 价格相对MA的偏离度 = (close - MA) / MA                          |
| `ma_bull`                                                   | 多头排列程度（MA5>MA10, MA10>MA20, MA20>MA60 各占1/3，范围0~1） |

### MACD 类

| 特征名        | 说明                                     |
| ------------- | ---------------------------------------- |
| `macd_dif`  | (EMA12 - EMA26) / close（归一化）        |
| `macd_dea`  | DIF 的 9日 EMA / close（归一化）         |
| `macd_hist` | (DIF - DEA) × 2 / close（归一化MACD柱） |

### RSI 类

| 特征名     | 说明     |
| ---------- | -------- |
| `rsi_6`  | 6日 RSI  |
| `rsi_12` | 12日 RSI |
| `rsi_24` | 24日 RSI |

### 布林带类

| 特征名       | 说明                                      |
| ------------ | ----------------------------------------- |
| `bb_pctb`  | 价格在布林带中的位置（0=下轨，1=上轨）    |
| `bb_width` | 带宽 = (上轨 - 下轨) / 中轨，衡量波动状态 |

### ATR 类

| 特征名          | 说明                                  |
| --------------- | ------------------------------------- |
| `atr14_ratio` | 14日平均真实波幅 / 收盘价，相对化 ATR |

### KDJ 类

| 特征名    | 说明                  |
| --------- | --------------------- |
| `kdj_k` | KDJ 的 K 值           |
| `kdj_d` | KDJ 的 D 值           |
| `kdj_j` | KDJ 的 J 值 = 3K - 2D |

### 成交量类

| 特征名         | 说明                                |
| -------------- | ----------------------------------- |
| `vol_ratio`  | 当日成交量 / 20日均成交量（量比）   |
| `obv_diff`   | OBV（On-Balance Volume）的5日变化率 |
| `turn_ratio` | 换手率 / 20日均换手率               |
| `amt_ratio`  | 成交额 / 20日均成交额               |

### 波动率类

| 特征名              | 说明                         |
| ------------------- | ---------------------------- |
| `volatility_5d`   | 5日收益率标准差              |
| `volatility_10d`  | 10日收益率标准差             |
| `volatility_20d`  | 20日收益率标准差             |
| `upside_vol_20`   | 20日上行波动率（仅正收益日） |
| `downside_vol_20` | 20日下行波动率（仅负收益日） |

### 价格位置类

| 特征名            | 说明                         |
| ----------------- | ---------------------------- |
| `price_pos_10d` | 收盘价在10日区间的百分位位置 |
| `price_pos_20d` | 收盘价在20日区间的百分位位置 |
| `price_pos_60d` | 收盘价在60日区间的百分位位置 |

### 基本面类

| 特征名        | 说明             |
| ------------- | ---------------- |
| `peTTM_chg` | 滚动市盈率变化率 |
| `pbMRQ_chg` | 市净率变化率     |
| `psTTM_chg` | 滚动市销率变化率 |

### 截面排名类

| 特征名           | 说明                       |
| ---------------- | -------------------------- |
| `peTTM_rank`   | 全市场PE百分位排名（截面） |
| `pbMRQ_rank`   | 全市场PB百分位排名（截面） |
| `ret_1d_rank`  | 全市场1日收益率排名        |
| `ret_5d_rank`  | 全市场5日收益率排名        |
| `ret_20d_rank` | 全市场20日收益率排名       |

### 市场环境类

| 特征名                | 说明                            |
| --------------------- | ------------------------------- |
| `mkt_ret_mean`      | 当日市场平均收益率（市场情绪）  |
| `mkt_advance_ratio` | 当日上涨股票占比                |
| `excess_ret_1d`     | 个股收益 - 市场均值（相对强弱） |

### 预测标签

```python
# T日决策，T+1开盘买入，持有5个交易日，T+6开盘卖出
label = open[t+6] / open[t+1] - 1 - buy_commission - sell_commission - stamp_tax
```

扣除成本：

- 买入手续费：0.0085%
- 卖出手续费：0.0085%
- 卖出印花税：0.05%

---

## 7. 模型设计详解

### 模型架构

- **类型**：LightGBM Regressor（梯度提升决策树）
- **损失函数**：Huber（`alpha=0.9`），对异常收益更鲁棒
- **集成**：5个模型（不同随机种子），取均值作为最终预测
- **并行训练**：使用 `ThreadPoolExecutor(max_workers=5)` 并行训练，加速约3倍
- **目标**：预测每只股票 T+1→T+6 净收益率（持有5个交易日）

### LightGBM 超参数

| 参数                      | 值        | 说明                         |
| ------------------------- | --------- | ---------------------------- |
| `objective`             | `huber` | Huber 损失函数，鲁棒性强     |
| `alpha`                 | 0.9       | Huber delta 参数             |
| `num_leaves`            | 31        | 叶节点数，控制模型复杂度     |
| `max_depth`             | 5         | 最大树深度                   |
| `learning_rate`         | 0.03      | 学习率                       |
| `num_boost_round`       | 800       | 最大迭代轮数                 |
| `feature_fraction`      | 0.6       | 每棵树使用60%特征            |
| `bagging_fraction`      | 0.7       | 每棵树使用70%样本            |
| `bagging_freq`          | 5         | 每5轮 bagging 一次           |
| `min_child_samples`     | 200       | 叶节点最小样本数，防止过拟合 |
| `lambda_l1`             | 1.0       | L1 正则化                    |
| `lambda_l2`             | 5.0       | L2 正则化                    |
| `early_stopping_rounds` | 50        | 验证集无改善则提前停止       |

### Walk-Forward 训练策略

```
时间轴示意：

2020-01  ←────── 3年训练窗口 ──────→  2023-01  → 预测
2020-04  ←────── 3年训练窗口 ──────→  2023-04  → 预测
2020-07  ←────── 3年训练窗口 ──────→  2023-07  → 预测
...（每22个交易日重训一次）
```

**具体规则**：

- 训练窗口：3年滚动（约756个交易日）
- 验证集：训练集最后10%（约75个交易日），中间加 HOLD_DAYS purge gap
- 重训频率：每22个交易日（约1个月）
- 首个预测起点：2023-01-01

**前视偏差防护（Lookahead Bias Prevention）**：

训练集末尾 `HOLD_DAYS=5` 条的 label 依赖 `target_date` 及之后的未来开盘价，必须从训练集中剔除：

```python
safe_end_idx = max(0, target_idx - HOLD_DAYS)
```

验证集与训练集之间同样加入 `HOLD_DAYS` 天的 purge gap：

```python
val_start = min(val_split + HOLD_DAYS, len(train_dates))
```

**单日快速模式**（`--date`）：

- 以指定日期前3年数据一次性训练
- 仅预测指定日期，耗时约1-3分钟
- 适合��常使用

### 置信度计算

```python
confidence = 1.0 / (1.0 + pred_std * 100)
```

- `pred_std`：5个模型预测值的标准差
- 标准差越大 → 模型分歧越大 → 置信度越低
- 范围：约 0.01（极低）到 0.99（极高）

### 选股过滤条件（py04_today.py）

```python
pred_return > 0.002          # 预测净收益率 > 0.2%
confidence > 0.5             # 置信度大于 0.5
```

---

## 8. 回测设计详解

### 总体设计原则

严格遵循 A 股真实交易规则，避免前视偏差（Look-Ahead Bias）：

- **成交价格**：统一使用开盘价（模拟集合竞价）
- **预测信号**：当日收盘后产生（T 日信号，T+1 执行）
- **T+1 锁仓**：T 日买入的股票 T+1 日才可卖出
- **涨跌停检测**：
  - 买入涨停封板 → 买入失败，不补仓
  - 卖出跌停封板 → 卖出失败，继续持仓

### 持仓管理

```
每日开盘执行：
1. 卖出：满足止损(-5%)、止盈(+8%)、超过持有期(5天)、或预测转负 的持仓
   （受T+1限制，当日买入不可当日卖出）
2. 买入：每天最多新增2只，从当日候选池中按预测收益从高到低选取
   （候选条件：pred_return>0.2% 且 confidence>0.5）
3. 仓位分配：等权重，每仓约占总资产的 1/MAX_POSITIONS，留有余量
```

### 交易成本

| 项目       | 费率                         | 触发时机 |
| ---------- | ---------------------------- | -------- |
| 买入手续费 | 成交额 × 0.0085%（最低1元） | 买入时   |
| 卖出手续费 | 成交额 × 0.0085%（最低1元） | 卖出时   |
| 印花税     | 成交额 × 0.05%              | 仅卖出时 |

**总成本**：买入 ~0.0085% + 卖出 ~0.0585% = 单次完整交易约 0.067%

### 回测绩效指标

| 指标        | 计算方式                                                  |
| ----------- | --------------------------------------------------------- |
| 总收益率    | (最终净值 / 初始资金) - 1                                 |
| 年化收益率  | (1 + 总收益率)^(252/交易天数) - 1                         |
| 夏普比率    | (年化收益率 - 无风险利率) / 年化波动率，无风险利率取 2.5% |
| 最大回撤    | max((高点净值 - 当前净值) / 高点净值)                     |
| Calmar 比率 | 年化收益率 / 最大回撤                                     |
| 胜率        | 盈利交易次数 / 总交易次数                                 |
| 盈亏比      | 平均盈利 / 平均亏损                                       |
| 年化波动率  | 日收益率标准差 × sqrt(252)                               |

---

## 9. 交易规则与成本

### 选股范围

- **沪市主板**：股票代码以 `sh.60` 开头（如 sh.600000 浦发银行）
- **深市主板**：股票代码以 `sz.00` 开头（如 sz.000001 平安银行）
- **排除**：科创板（sh.688*）、创业板（sz.30*）、北交所

### 过滤规则

| 规则       | 说明                                           |
| ---------- | ---------------------------------------------- |
| 剔除 ST    | 名称含 ST / *ST 的股票                         |
| 剔除停牌   | 成交量为0或涨跌幅缺失的行                      |
| 剔除新股   | 上市后前5个交易日                              |
| 流动性过滤 | 20日均成交额 < 500万元                         |
| 涨跌停过滤 | 买入时开盘涨幅 ≥ 涨停幅度 → 跳过（无法成交） |

### 交易机制

- **成交价格**：每日开盘价（集合竞价价格）
- **T+1 规则**：买入当日不可卖出，次日才可卖出
- **最大持仓**：同时持有不超过 5 只股票
- **每日限购**：每天最多新建仓 2 只（分批建仓，控制风险）
- **目标持有期**：5个交易日
- **止损**：持仓浮亏超过 5% 时在次日开盘卖出
- **止盈**：持仓浮盈超过 8% 时在次日开盘卖出
- **空仓条件**：若无满足条件的候选股，保持空仓持有现金

---

## 10. 输出文件说明

### 日常输出

| 文件                                  | 更新频率 | 说明                     |
| ------------------------------------- | -------- | ------------------------ |
| `output/trading_strategy.md`        | 每日     | 当日全市场 TOP5 买入建议 |
| `output/trading_strategy.html`      | 每日     | HTML 格式，供邮件推送    |
| `output/trading_strategy_{code}.md` | 按需     | 单只股票详细分析报告     |
| `output/history/strategy_{date}.md` | 每日     | 历史策略自动存档         |

### 回测输出

| 文件                                    | 说明                                  |
| --------------------------------------- | ------------------------------------- |
| `output/backtest/backtest_daily.csv`  | 每日净值、现金、持仓数、交易次数      |
| `output/backtest/trade_log.csv`       | 完整交易流水（价格、数量、成本、P&L） |
| `output/backtest/backtest_metrics.md` | 绩效指标摘要文字报告                  |

### 图表输出

| 文件                                       | 说明                        |
| ------------------------------------------ | --------------------------- |
| `output/backtest/equity_curve.png`       | 净值曲线（含基准对比）      |
| `output/backtest/drawdown_curve.png`     | 最大回撤曲线                |
| `output/backtest/daily_positions.png`    | 每日持仓数量 + 买卖笔数     |
| `output/backtest/monthly_returns.png`    | 月度收益热力图              |
| `output/backtest/feature_importance.png` | LightGBM 特征重要性（Gain） |

---

## 11. 依赖与环境

### Python 依赖

| 包             | 版本要求  | 用途                        |
| -------------- | --------- | --------------------------- |
| `baostock`   | ≥ 0.8.8  | A股数据接口（baostock.com） |
| `pandas`     | ≥ 2.0.0  | 数据处理与分析              |
| `numpy`      | ≥ 1.24.0 | 数值计算                    |
| `lightgbm`   | ≥ 4.0.0  | 机器学习模型                |
| `tqdm`       | ≥ 4.65.0 | 进度条显示                  |
| `matplotlib` | ≥ 3.7.0  | 可视化图表                  |

### 安装

```bash
pip install -r requirements.txt
```

### 系统要求

- **Python**：3.8+（推荐 3.11+）
- **操作系统**：macOS / Linux / Windows 均支持
- **内存**：建议 8GB+（处理全量特征数据时约需 4-6GB）
- **磁盘**：data/ 目录约占 2-5GB（月度 CSV + pkl 缓存）
- **网络**：首次下载数据需要访问 baostock.com

---

## 12. 常见问题排查

### Q1：首次运行很慢怎么办？

**原因**：需从 baostock 下载约5000只股票×6年历史数据（约140个月度 CSV 文件）。

**解决**：

- 全量下载正常耗时 30-60 分钟
- 支持中断续传：直接重新运行 `python src/py00_fetch_stock_data.py`，会从断点继续
- 后续增量更新仅需 1-5 分钟

---

### Q2：GitHub Actions 首次运行超时？

**原因**：首次运行无 pkl 缓存，全量下载数据量大。

**解决**：

1. 在本地完成初始化后，缓存由 Actions 的 `actions/cache` 持久化维护
2. 或在 Actions 页面多次手动触发，每次续传进度（断点续传支持）

---

### Q3：预测��件损坏或模型报错？

**解决**：

```bash
rm data/predictions.pkl data/features.pkl
./scripts/run_backtest.sh   # 重新生成
```

---

### Q4：邮件推送不工作？

**检查清单**：

1. GitHub Secrets 中 5 个变量是否全部设置（特别是 `EMAIL_RECIPIENT`）
2. Gmail 需使用「应用专用密码」（账户 → 安全 → 两步验证 → 应用密码）
3. SMTP 端口：587（STARTTLS）或 465（SSL）
4. 查看 Actions 运行日志，定位具体报错信息

---

### Q5：数据量太大，磁盘占满？

**data/ 目录结构**：

- `Stock_dailyK_*.csv`：月度K线，约2-4GB，**可删除**（下次运行会重新下载）
- `*.pkl`：约 200-500MB，**建议保留**（删除后需重新处理）

```bash
# 清理月度 CSV（保留最近12个月）
ls data/Stock_dailyK_*.csv | sort | head -n -12 | xargs rm
```

---

### Q6：baostock 连接失败？

```bash
# 检查 baostock 是否可访问
python -c "import baostock as bs; rs = bs.login(); print(rs.error_msg)"

# 如持续失败，删除状态文件后重试
rm data/.download_status.txt
python src/py00_fetch_stock_data.py
```

---

## 附录：项目设计原始需求

详见项目初始设计文档（原 `docs/backtest.md`）。该文件记录了系统的初始设计要求，包括：

- 交易规则约束（主板选股、集合竞价、T+1、5只上限、手续费）
- 模型设计理念（ML/DL 优先、不确定性量化、丰富特征工程）
- 回测要求（Walk-Forward、真实成本模拟）
- 输出标准（绩效指标、图表要求、今日决策格式）
