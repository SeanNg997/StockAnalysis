# StockAnalysis — A股量化交易策略系统

基于 LightGBM 机器学习模型的 A 股量化选股策略，支持每日自动化运行和 GitHub Actions 定时推送。

## 策略概述

- **数据源**：baostock（免费A股日K线数据，近10年）
- **选股范围**：沪深主板（sh.60 / sz.00 开头），剔除 ST、停牌、低流动性股票
- **模型**：LightGBM Ensemble（5个不同种子），Walk-Forward 滚动训练
- **交易规则**：集合竞价开盘价成交，严格 T+1，最多持仓 5 只，含手续费和印花税
- **标签**：T+1 开盘买入 → T+2 开盘卖出的净收益率

## 回测表现（2023-01 ~ 2026-03）

| 指标 | 数值 |
|------|------|
| 年化收益率 | 161.85% |
| 夏普比率 | 2.729 |
| 最大回撤 | -26.44% |
| Calmar 比率 | 6.121 |
| 胜率 | 54.67% |
| 盈亏比 | 1.798 |

## 项目结构

```
StockAnalysis/
├── src/                          # 源代码
│   ├── py00_fetch_stock_data.py  #   数据爬取（全量/增量/断点续传）
│   ├── py01_data_loader.py       #   数据清洗（过滤主板、剔除ST、流动性筛选）
│   ├── py02_features.py          #   特征工程（60+技术/动量/截面特征）
│   ├── py03_model.py             #   模型训练（Walk-Forward + Ensemble）
│   ├── py04_backtest.py          #   回测引擎（T+1、涨跌停、交易成本）
│   ├── py05_today.py             #   今日决策（生成买入TOP5）
│   ├── py06_report.py            #   可视化报告（净值曲线、回撤、热力图）
│   └── py07_quick_predict.py     #   快速预测（每日使用，跳过完整回测）
├── scripts/
│   └── run_daily.sh              # 一键运行脚本
├── data/                         # 数据目录（gitignored，通过Actions cache持久化）
├── output/                       # 输出目录
│   ├── today_strategy.txt        #   最新交易决策
│   ├── backtest_metrics.txt      #   回测指标
│   ├── *.png                     #   可视化图表
│   └── history/                  #   历史策略归档
├── docs/
│   └── backtest.md               # 策略设计文档
├── .github/workflows/
│   └── daily_report.yml          # GitHub Actions 自动化
├── requirements.txt
└── README.md
```

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### 首次运行（全量数据下载 + 完整回测）

```bash
# 1. 下载全量数据（约5000只股票，耗时较长）
python src/py00_fetch_stock_data.py

# 2. 完整流水线（清洗 → 特征 → 模型训练 → 回测 → 策略 → 图表）
./scripts/run_daily.sh --full
```

### 每日运行

```bash
# 快速模式：增量更新 + 快速预测 + 生成策略（几分钟）
./scripts/run_daily.sh

# 完整模式：含 Walk-Forward 重训练 + 回测（耗时长，建议每周/月运行一次）
./scripts/run_daily.sh --full
```

策略报告输出到 `output/today_strategy.txt`。

## GitHub Actions 自动化

仓库已配置 GitHub Actions，每个交易日北京时间 6:00 自动运行：

1. 增量更新前一交易日数据
2. 数据清洗 + 特征工程
3. 快速模型预测
4. 生成策略报告并提交到 `output/history/`

### 推送通知（可选）

在 Settings → Secrets and variables → Actions 中添加 `WEBHOOK_URL`（企业微信/钉钉机器人 Webhook），即可每日自动推送策略报告。

### 首次启用 Actions

Actions 依赖缓存的数据文件。首次需要手动触发一次 workflow（或本地先运行一遍完整流水线以产生数据文件，然后通过 Actions 的 cache 机制持久化）。

## 特征工程

模型使用 60+ 特征，涵盖：

| 类别 | 特征示例 |
|------|----------|
| 价格动量 | 1/3/5/10/20/60日收益率、对数收益率 |
| 均线系统 | MA5/10/20/60 偏离度、多头排列信号 |
| 技术指标 | MACD、RSI、布林带、ATR、KDJ |
| 成交量 | 量比、OBV、换手率突变、成交额比 |
| 波动率 | 5/10/20日波动率、上行/下行波动率、最大回撤 |
| 价格位置 | 10/20/60日高低价位置 |
| 基本面 | PE/PB/PS 变化率、截面百分位排名 |
| 市场环境 | 市场平均收益、上涨比例、市场动量 |

## 风险提示

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。回测结果不代表未来收益。
