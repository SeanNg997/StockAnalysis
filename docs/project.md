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
   - [py04 — 回测引擎](#py04--回测引擎)
   - [py05 — 今日决策](#py05--今日决策)
   - [py06 — 可视化报告](#py06--可视化报告)
   - [py07 — 盘后复盘](#py07--盘后复盘)
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
│   ├── py03_model.py                       #   LightGBM 训练与预测
│   ├── py04_backtest.py                    #   回测引擎
│   ├── py05_today.py                       #   今日交易决策
│   ├── py06_report.py                      #   可视化报告
│   └── py07_review.py                      #   盘后复盘评估
├── scripts/
│   ├── run_daily.sh                        #   日常快速预测脚本
│   ├── run_daily_review.sh                 #   盘后复盘脚本
│   └── run_model.sh                        #   完整重训练脚本
├── data/                                   # 数据目录（.gitignore，通过 Actions cache 持久化）
│   ├── Stock_dailyK_YYYYMM.csv            #   月度K线数据（~140个文件）
│   ├── mainboard_clean.pkl                 #   清洗后的主板数据
│   ├── features.pkl                        #   特征工程数据集
│   ├── predictions.pkl                     #   模型预测结果
│   ├── .last_date.txt                      #   最后更新日期
│   └── .download_status.txt               #   下载进度追踪
├── output/                                 # 输出目录
│   ├── today_strategy.md                   #   当日策略报告
│   ├── backtest_daily.csv                  #   每日回测结果
│   ├── trade_log.csv                       #   交易流水
│   ├── backtest_metrics.txt                #   回测绩效指标
│   ├── equity_curve.png                    #   净值曲线图
│   ├── drawdown_curve.png                  #   最大回撤图
│   ├── daily_positions.png                 #   每日持仓图
│   ├── monthly_returns.png                 #   月度收益热力图
│   ├── feature_importance.png              #   特征重要性图
│   ├── today_review_YYYYMMDD.png           #   盘后复盘图
│   └── history/                            #   历史策略存档
├── docs/
│   ├── backtest.md                         #   策略设计原始需求文档
│   └── project.md                          #   本文件（项目详细文档）
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
│    ├─ 数据源：baostock API（免费，覆盖近10年A股日K线）               │
│    ├─ 股票范围：沪深主板（sh.60* / sz.00*），约5000只               │
│    ├─ 支持全量下载（首次）和增量更新（日常）                         │
│    └─ 输出：data/Stock_dailyK_YYYYMM.csv（月度拆分，约140个文件）   │
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
│    ├─ 计算预测标签：T+1开盘买入→T+2开盘卖出，扣除手续费+印花税         │
│    └─ 输出：data/features.pkl                                       │
│                          ↓                                          │
│  py03_model.py                                                      │
│    ├─ Walk-Forward 滚动训练（每22个交易日重训一次）                   │
│    ├─ 或 单日快速预测（以指定日期前3年数据训练，预测当天）              │
│    ├─ 模型：LightGBM Ensemble（5个不同随机种子）                     │
│    └─ 输出：data/predictions.pkl（含pred_return、pred_std、confidence）│
│                          ↓                    ↓                     │
│  py04_backtest.py                   py05_today.py                  │
│    ├─ 模拟真实交易                    ├─ 读取最新预测结果              │
│    ├─ T+1锁定、涨跌停检测              ├─ 过滤：预测收益>0.1%且置信度>中位数│
│    ├─ 最多5只持仓                     ├─ 输出TOP5买入建议              │
│    └─ 输出回测报告+交易流水            └─ 输出today_strategy.md       │
│                          ↓                                          │
│  py06_report.py                                                     │
│    └─ 生成可视化图表（净值、回撤、持仓、热力图、特征重要性）            │
│                          ↓（收盘后）                                 │
│  py07_review.py                                                     │
│    └─ 对比预测收益 vs 实际收益，生成盘后复盘图                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 模块详解

### py00 — 数据获取

**文件**：`src/py00_fetch_stock_data.py`

**核心功能**：

从 [baostock](http://baostock.com) 下载 A 股日 K 线数据，支持全量下载与断点续传增量更新。

**主要函数**：

| 函数 | 说明 |
|------|------|
| `get_stock_list()` | 获取所有主板股票代码（sh.60* / sz.00*） |
| `fetch_daily_full(code, start, end)` | 全量下载指定股票指定时间段的日K线 |
| `fetch_daily_increment(code, last_date)` | 增量下载：从上次更新日期到今日 |
| `load_existing_csv()` | 加载所有已下载的月度 CSV 文件 |
| `save_to_csv(df)` | 将数据按月拆分保存为 CSV |
| `save_incremental_months(df, new_df)` | 仅更新有变化的月度 CSV 文件 |

**数据字段**（原始 CSV 列）：

| 字段 | 说明 |
|------|------|
| `date` | 交易日期（YYYY-MM-DD） |
| `code` | 股票代码（如 sh.600000） |
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `volume` | 成交量（手） |
| `amount` | 成交额（元） |
| `turn` | 换手率（%） |
| `pctChg` | 涨跌幅（%） |
| `peTTM` | 滚动市盈率 |
| `pbMRQ` | 市净率 |
| `psTTM` | 滚动市销率 |
| `isST` | 是否ST（1=是） |

**运行方式**：

```bash
# 全量下载（首次，约30-60分钟）
python src/py00_fetch_stock_data.py

# 测试模式（仅下载前10只股票）
python src/py00_fetch_stock_data.py -n 10

# 增量更新（日常，约1-5分钟）
python src/py00_fetch_stock_data.py --update
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
4. 剔除 `isST == 1` 的股票
5. 剔除每只股票上市后前5个交易日（避免新股效应）
6. 按股票前向填充缺失行情数据
7. 流动性筛选：计算20日滚动平均成交额，剔除低于500万元的记录
8. 最小历史约束：剔除历史天数不足120日的股票
9. 保存为 `data/mainboard_clean.pkl`

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
# Walk-Forward 全量训练（用于回测，需30-90分钟）
python src/py03_model.py

# 单日快速预测（用于当日决策，约2-5分钟）
python src/py03_model.py --date 2025-03-14
```

**predictions.pkl 字段**：

| 字段 | 说明 |
|------|------|
| `date` | 预测日期 |
| `code` | 股票代码 |
| `pred_return` | 预测 T+1→T+2 净收益率 |
| `pred_std` | 5模型预测标准差（不确定性） |
| `confidence` | 置信度分数（1 / (1 + pred_std×100)） |

---

### py04 — 回测引擎

**文件**：`src/py04_backtest.py`

详细设计见[第8节 回测设计详解](#8-回测设计详解)。

**运行方式**：

```bash
python src/py04_backtest.py
```

**输出文件**：

- `output/backtest_daily.csv` — 每日投资组合净值、现金、持仓数量、交易次数
- `output/trade_log.csv` — 完整交易流水（买入/卖出价格、数量、手续费、P&L）
- `output/backtest_metrics.txt` — 回测绩效摘要

---

### py05 — 今日决策

**文件**：`src/py05_today.py`

**核心功能**：

读取 `predictions.pkl` 和 `features.pkl`，生成当日交易建议。

**两种模式**：

**1. 全市场报告** (`generate_today_strategy`)

- 过滤条件：`pred_return > 0.1%` 且 `confidence > 中位数置信度`
- 输出 TOP5 买入推荐
- 包含市场整体概览（涨跌家数、平均预测收益、行情热度）
- 输出完整 TOP20 排名

**2. 单股报告** (`generate_stock_report(code)`)

- 该股票当前关键指标（PE、PB、换手率等）
- 5日预测趋势
- 与市场对比（相对强弱）
- 投资建议（强烈买入 / 买入 / 谨慎 / 观察 / 回避）

**运行方式**：

```bash
# 全市场报告
python src/py05_today.py

# 单只股票报告
python src/py05_today.py 600000

# 指定历史日期
python src/py05_today.py --date 2025-03-14

# 历史日期 + 单只股票
python src/py05_today.py --date 2025-03-14 600000
```

**输出**：

- `output/today_strategy.md` — 全市场策略报告（Markdown 格式）
- `output/today_strategy_{code}.md` — 单股报告
- `output/history/strategy_{date}.md` — 历史策略自动存档

---

### py06 — 可视化报告

**文件**：`src/py06_report.py`

**生成的图表**：

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 净值曲线 | `equity_curve.png` | 策略净值 vs 基准（持有现金）对比 |
| 最大回撤 | `drawdown_curve.png` | 从历史高点的回撤百分比 |
| 每日持仓 | `daily_positions.png` | 持仓数量 + 每日买卖笔数 |
| 月度热力图 | `monthly_returns.png` | 按月/年分布的收益率（RdYlGn 色系） |
| 特征重要性 | `feature_importance.png` | LightGBM Gain 排名，TOP25 特征 |

**运行方式**：

```bash
python src/py06_report.py
```

---

### py07 — 盘后复盘

**文件**：`src/py07_review.py`

**功能**：

收盘后（16:00+）评估当日策略的实际执行效果：

1. 解析 `today_strategy.md` 中的决策日期和 TOP5 推荐
2. 加载实际市场数据（对应执行日的K线）
3. 计算：实际开盘价、收盘价、全天涨跌幅、状态（正常/涨停/跌停/停牌）
4. 对比预测收益 vs 实际收益
5. 生成可视化复盘图

**运行方式**：

```bash
# 正常模式（需要16:00后运行）
python src/py07_review.py

# 强制模式（调试用，跳过时间检查）
python src/py07_review.py --force

# 历史日期复盘
python src/py07_review.py --date 2025-03-14
```

**输出**：`output/today_review_YYYYMMDD.png`

---

## 4. Shell 脚本

### `scripts/run_daily.sh` — 日常快速预测

**典型耗时**：约30秒

```bash
./scripts/run_daily.sh                            # 当日全市场报告
./scripts/run_daily.sh --date 2025-03-14          # 历史日期报告
./scripts/run_daily.sh 600000                     # 当日单股报告
./scripts/run_daily.sh --date 2025-03-14 600000   # 历史日期单股报告
```

**内部流程**：

```
1. 增量数据更新（若指定 --date 则跳过）
2. 数据清洗（截止到指定日期）
3. 特征工程（截止到指定日期）
4. 单日快速预测（以指定日期前3年数据训练）
5. 生成策略报告 + 可视化图表
```

---

### `scripts/run_model.sh` — 完整重训练

**典型耗时**：约5-10分钟（Walk-Forward 全量训练）

```bash
./scripts/run_model.sh
```

**内部流程**：

```
1. 增量数据更新
2. 全量数据清洗
3. 全量特征工程
4. Walk-Forward 滚动训练（每22个交易日重训一次）
5. 回测模拟 + 生成回测报告
```

建议**每周或每月**运行一次，以更新模型的历史预测基准。

---

### `scripts/run_daily_review.sh` — 盘后复盘

```bash
./scripts/run_daily_review.sh             # 正常模式（16:00后）
./scripts/run_daily_review.sh --force     # 强制模式（调试）
./scripts/run_daily_review.sh --date 2025-03-14  # 历史复盘
```

---

## 5. GitHub Actions 自动化

**文件**：`.github/workflows/daily_report.yml`

### 触发时间

- **定时**：UTC 23:30（北京时间次日 7:30），对应 `cron: '30 23 * * 0-4'`
  - 注意：周日 23:30 UTC = 周一 7:30 北京时间，以此类推
- **手动**：支持 `workflow_dispatch` 在任意时间手动触发

### 工作流步骤

```
1. Checkout 仓库代码
2. 设置 Python 3.11 环境
3. 缓存 pip 依赖包（加速后续安装）
4. 恢复数据缓存（pkl 文件 + 月度 CSV）
5. 检测是否为交易日（非交易日跳过）
6. 判断是首次运行（无缓存）还是增量更新
   - 无缓存：全量下载10年数据
   - 有缓存：仅增量更新最新交易日
7. 数据清洗 + 特征工程（使用 --date 参数截止今日）
8. 单日快速预测
9. 生成策略报告 + 图表
10. 将策略报告转换为 HTML 邮件格式
11. 通过 SMTP 发送邮件通知（若配置了邮件 Secrets）
12. 将生成文件推送回仓库（策略归档）
```

### 缓存策略

使用 `actions/cache` 缓存以下文件（跨运行持久化）：

| 文件 | 缓存目的 |
|------|---------|
| `data/mainboard_clean.pkl` | 避免每次重新清洗全量数据 |
| `data/features.pkl` | 避免每次重新计算特征 |
| `data/predictions.pkl` | 保留历史预测供复盘对比 |
| `data/.last_date.txt` | 记录最后更新日期，用于增量检测 |
| `data/Stock_dailyK_*.csv` | 月度K线文件缓存，节省网络流量 |
| `data/.download_status.txt` | 下载进度追踪，支持断点续传 |

### 邮件推送配置

在 **GitHub → Settings → Secrets and variables → Actions** 中配置：

| Secret 名 | 说明 | 示例值 |
|-----------|------|--------|
| `EMAIL_SERVER` | SMTP 服务器地址 | `smtp.gmail.com` |
| `EMAIL_PORT` | SMTP 端口 | `587` |
| `EMAIL_USERNAME` | 发件人账户 | `your@gmail.com` |
| `EMAIL_PASSWORD` | 应用密码（非登录密码） | `xxxx xxxx xxxx xxxx` |
| `EMAIL_RECIPIENT` | 收件人邮箱 | `recipient@example.com` |

> **注意**：Gmail 需要开启两步验证并生成「应用专用密码」，不能使用账户密码。

### 首次启用

**方式一（推荐）**：直接在 GitHub Actions 页面手动触发 workflow

- GitHub → Actions → Daily Stock Strategy Report → Run workflow
- 首次运行自动执行全量数据下载（约30-60分钟）

**方式二**：本地初始化后推送

```bash
python src/py00_fetch_stock_data.py   # 下载全量数据
./scripts/run_model.sh                # 完整流水线
git push                              # 推送代码（数据由 Actions cache 维护）
```

---

## 6. 特征工程详解

共 **60+ 个特征**，分为以下类别：

### 价格动量类

| 特征名 | 计算方式 | 说明 |
|--------|---------|------|
| `ret_1d` | `close/close.shift(1) - 1` | 1日收益率 |
| `ret_3d` | `close/close.shift(3) - 1` | 3日收益率 |
| `ret_5d` | `close/close.shift(5) - 1` | 5日收益率 |
| `ret_10d` | `close/close.shift(10) - 1` | 10日收益率 |
| `ret_20d` | `close/close.shift(20) - 1` | 20日收益率 |
| `ret_60d` | `close/close.shift(60) - 1` | 60日收益率 |
| `log_ret_1d` | `log(close/close.shift(1))` | 1日对数收益率 |

### 均线系统类

| 特征名 | 说明 |
|--------|------|
| `ma_5`, `ma_10`, `ma_20`, `ma_60` | N日简单移动平均价格 |
| `ma_bias_5`, `ma_bias_10`, `ma_bias_20`, `ma_bias_60` | 价格相对MA的偏离度 = (close - MA) / MA |
| `ma_bull` | 多头排列信号（1 = MA5 > MA10 > MA20 > MA60） |

### MACD 类

| 特征名 | 说明 |
|--------|------|
| `macd_dif` | EMA12 - EMA26（DIF 线） |
| `macd_dea` | DIF 的 9日 EMA（DEA/Signal 线） |
| `macd_hist` | (DIF - DEA) × 2（MACD 柱） |

### RSI 类

| 特征名 | 说明 |
|--------|------|
| `rsi_6` | 6日 RSI |
| `rsi_12` | 12日 RSI |
| `rsi_24` | 24日 RSI |

### 布林带类

| 特征名 | 说明 |
|--------|------|
| `bb_pctb` | 价格在布林带中的位置（0=下轨，1=上轨） |
| `bb_width` | 带宽 = (上轨 - 下轨) / 中轨，衡量波动状态 |

### ATR 类

| 特征名 | 说明 |
|--------|------|
| `atr14_ratio` | 14日平均真实波幅 / 收盘价，相对化 ATR |

### KDJ 类

| 特征名 | 说明 |
|--------|------|
| `kdj_k` | KDJ 的 K 值 |
| `kdj_d` | KDJ 的 D 值 |
| `kdj_j` | KDJ 的 J 值 = 3K - 2D |

### 成交量类

| 特征名 | 说明 |
|--------|------|
| `vol_ratio` | 当日成交量 / 20日均成交量（量比） |
| `obv_diff` | OBV（On-Balance Volume）的5日变化率 |
| `turn_ratio` | 换手率 / 20日均换手率 |
| `amt_ratio` | 成交额 / 20日均成交额 |

### 波动率类

| 特征名 | 说明 |
|--------|------|
| `volatility_5d` | 5日收益率标准差（年化） |
| `volatility_10d` | 10日收益率标准差 |
| `volatility_20d` | 20日收益率标准差 |
| `upside_vol_20` | 20日上行波动率（仅正收益日） |
| `downside_vol_20` | 20日下行波动率（仅负收益日） |

### 价格位置类

| 特征名 | 说明 |
|--------|------|
| `price_pos_10d` | 收盘价在10日区间的百分位位置 |
| `price_pos_20d` | 收盘价在20日区间的百分位位置 |
| `price_pos_60d` | 收盘价在60日区间的百分位位置 |

### 基本面类

| 特征名 | 说明 |
|--------|------|
| `peTTM_chg` | 滚动市盈率变化率 |
| `pbMRQ_chg` | 市净率变化率 |
| `psTTM_chg` | 滚动市销率变化率 |

### 截面排名类

| 特征名 | 说明 |
|--------|------|
| `peTTM_rank` | 全市场PE百分位排名（截面） |
| `pbMRQ_rank` | 全市场PB百分位排名（截面） |
| `ret_1d_rank` | 全市场1日收益率排名 |
| `ret_5d_rank` | 全市场5日收益率排名 |
| `ret_20d_rank` | 全市场20日收益率排名 |

### 市场环境类

| 特征名 | 说明 |
|--------|------|
| `mkt_ret_mean` | 当日市场平均收益率（市场情绪） |
| `mkt_advance_ratio` | 当日上涨股票占比 |
| `excess_ret_1d` | 个股收益 - 市场均值（相对强弱） |

### 预测标签

```python
label = (open.shift(-1) - open.shift(-2)) / open.shift(-2)
      - buy_commission - sell_commission - stamp_tax
```

即：在 T+1 开盘价买入，T+2 开盘价卖出，扣除：
- 买入手续费：0.0085%
- 卖出手续费：0.0085%
- 卖出印花税：0.05%

---

## 7. 模型设计详解

### 模型架构

- **类型**：LightGBM Regressor（梯度提升决策树）
- **集成**：5个模型（不同随机种子），取均值作为最终预测
- **目标**：预测每只股票 T+1→T+2 净收益率

### LightGBM 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_leaves` | 63 | 叶节点数，控制模型复杂度 |
| `max_depth` | 7 | 最大树深度 |
| `learning_rate` | 0.05 | 学习率 |
| `n_estimators` | 2000 | 最大迭代轮数 |
| `feature_fraction` | 0.7 | 每棵树使用70%特征 |
| `bagging_fraction` | 0.8 | 每棵树使用80%样本 |
| `bagging_freq` | 5 | 每5轮 bagging 一次 |
| `early_stopping_rounds` | 30 | 验证集无改善则提前停止 |

### Walk-Forward 训练策略

```
时间轴示意：

2020-01  ←────── 3年训练窗口 ──────→  2023-01  → 预测
2020-04  ←────── 3年训练窗口 ──────→  2023-04  → 预测
2020-07  ←────── 3年训练窗口 ──────→  2023-07  → 预测
...（每22个交易日重训一次）
```

**具体规则**：
- 训练窗口：3年滚动（约750个交易日）
- 验证集：训练集最后10%（约75个交易日）
- 重训频率：每22个交易日（约1个月）
- 首个预测起点：2023-01-01

**单日快速模式**（`--date`）：
- 以指定日期前3年数据一次性训练
- 仅预测指定日期，耗时约2-5分钟
- 适合日常使用

### 置信度计算

```python
confidence = 1.0 / (1.0 + pred_std * 100)
```

- `pred_std`：5个模型预测值的标准差
- 标准差越大 → 模型分歧越大 → 置信度越低
- 范围：约 0.01（极低）到 0.99（极高）

### 选股过滤条件

```python
# 同时满足以下两个条件才进入候选池
pred_return > 0.001          # 预测净收益率 > 0.1%
confidence > median(confidence)  # 置信度高于中位数
```

---

## 8. 回测设计详解

### 总体设计原则

严格遵循 A 股真实交易规则，避免前视偏差（Look-Ahead Bias）：

- **成交价格**：统一使用开盘价（模拟集合竞价）
- **预测信号**：当日收盘后产生（T 日信号，T+1 执行）
- **T+1 锁仓**：T 日买入的股票 T+1 日才可卖出
- **涨跌停检测**：若开盘价涨幅 ≥ 9.8%，跳过买入（无法成交）

### 持仓管理

```
每日开盘前执行：
1. 卖出：预测收益为负 或 不在当日TOP5候选中 的持仓（受T+1限制）
2. 买入：补满空缺仓位至5只，从当日候选池中按预测收益从高到低选取
3. 仓位分配：等权重，每仓约占总资产的 1/5（留2%现金备用）
```

### 交易成本

| 项目 | 费率 | 触发时机 |
|------|------|---------|
| 买入手续费 | 成交额 × 0.0085% | 买入时 |
| 卖出手续费 | 成交额 × 0.0085% | 卖出时 |
| 印花税 | 成交额 × 0.05% | 仅卖出时 |

**总成本**：买入 0.0085% + 卖出 (0.0085% + 0.05%) = 单次完整交易约 0.0670%

### 回测绩效指标

| 指标 | 计算方式 |
|------|---------|
| 总收益率 | (最终净值 / 初始资金) - 1 |
| 年化收益率 | (1 + 总收益率)^(252/交易天数) - 1 |
| 夏普比率 | (年化收益率 - 无风险利率) / 年化波动率，无风险利率取 2.5% |
| 最大回撤 | max((高点净值 - 当前净值) / 高点净值) |
| Calmar 比率 | 年化收益率 / 最大回撤 |
| 胜率 | 盈利交易次数 / 总交易次数 |
| 盈亏比 | 平均盈利 / 平均亏损 |
| 年化波动率 | 日收益率标准差 × sqrt(252) |

### 2023-2026 回测结果

| 指标 | 数值 |
|------|------|
| 初始资金 | 1,000,000 元 |
| 年化收益率 | **161.85%** |
| 夏普比率 | **2.729** |
| 最大回撤 | **-26.44%** |
| Calmar 比率 | 6.121 |
| 胜率 | 54.67% |
| 盈亏比 | 1.798 |

> ⚠️ **风险提示**：回测结果不代表未来实盘表现。历史表现优异可能部分源于过拟合或市场特殊时段，实盘中需考虑更多不可控因素。

---

## 9. 交易规则与成本

### 选股范围

- **沪市主板**：股票代码以 `sh.60` 开头（如 sh.600000 浦发银行）
- **深市主板**：股票代码以 `sz.00` 开头（如 sz.000001 平安银行）
- **排除**：科创板（sh.688*）、创业板（sz.30*）、北交所

### 过滤规则

| 规则 | 说明 |
|------|------|
| 剔除 ST | `isST == 1` 的股票 |
| 剔除新股 | 上市后前5个交易日 |
| 流动性过滤 | 20日均成交额 < 500万元 |
| 涨跌停过滤 | 涨幅 ≥ 9.8% 时跳过买入 |

### 交易机制

- **成交价格**：每日开盘价（集合竞价价格）
- **T+1 规则**：买入当日不可卖出，次日才可卖出
- **最大持仓**：同时持有不超过 5 只股票
- **空仓条件**：若无满足条件的候选股，全部持有现金

---

## 10. 输出文件说明

### 日常输出

| 文件 | 更新频率 | 说明 |
|------|---------|------|
| `output/today_strategy.md` | 每日 | 当日全市场 TOP5 买入建议 |
| `output/today_strategy_{code}.md` | 按需 | 单只股票详细分析报告 |
| `output/today_review_YYYYMMDD.png` | 每日收盘后 | 预测 vs 实际复盘图 |
| `output/history/strategy_{date}.md` | 每日 | 历史策略自动存档 |

### 回测输出

| 文件 | 说明 |
|------|------|
| `output/backtest_daily.csv` | 每日净值、现金、持仓数、交易次数 |
| `output/trade_log.csv` | 完整交易流水（价格、数量、成本、P&L） |
| `output/backtest_metrics.txt` | 绩效指标摘要文字报告 |

### 图表输出

| 文件 | 说明 |
|------|------|
| `output/equity_curve.png` | 净值曲线（含基准对比） |
| `output/drawdown_curve.png` | 最大回撤曲线 |
| `output/daily_positions.png` | 每日持仓数量 + 买卖笔数 |
| `output/monthly_returns.png` | 月度收益热力图 |
| `output/feature_importance.png` | LightGBM 特征重要性（Gain） |

---

## 11. 依赖与环境

### Python 依赖

| 包 | 版本要求 | 用途 |
|----|---------|------|
| `baostock` | ≥ 0.8.8 | A股数据接口（baostock.com） |
| `pandas` | ≥ 2.0.0 | 数据处理与分析 |
| `numpy` | ≥ 1.24.0 | 数值计算 |
| `lightgbm` | ≥ 4.0.0 | 机器学习模型 |
| `tqdm` | ≥ 4.65.0 | 进度条显示 |
| `matplotlib` | ≥ 3.7.0 | 可视化图表 |

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

**原因**：需从 baostock 下载约5000只股票×10年历史数据（约140个月度 CSV 文件）。

**解决**：
- 全量下载正常耗时 30-60 分钟
- 支持中断续传：直接重新运行 `python src/py00_fetch_stock_data.py`，会从断点继续
- 后续增量更新仅需 1-5 分钟

---

### Q2：GitHub Actions 首次运行失败？

**原因**：首次运行无 pkl 缓存，全量下载超时（Actions 默认90分钟限制）。

**解决**：
1. 在本地完成初始化：
   ```bash
   python src/py00_fetch_stock_data.py   # 全量下载
   ./scripts/run_model.sh                # 完整流水线
   git push                              # 推送数据缓存
   ```
2. 或在 Actions 页面多次手动触发，每次续传进度

---

### Q3：预测文件损坏或模型报错？

**解决**：

```bash
rm data/predictions.pkl data/features.pkl
./scripts/run_model.sh   # 重新生成
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
python src/py00_fetch_stock_data.py --update
```

---

## 附录：项目设计原始需求

详见 `docs/backtest.md`。该文件记录了系统的初始设计要求，包括：
- 交易规则约束（主板选股、集合竞价、T+1、5只上限、手续费）
- 模型设计理念（ML/DL 优先、不确定性量化、丰富特征工程）
- 回测要求（Walk-Forward、真实成本模拟）
- 输出标准（绩效指标、图表要求、今日决策格式）
