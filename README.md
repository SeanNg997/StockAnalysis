# StockAnalysis — A股量化交易策略系统

基于 LightGBM Ensemble + Walk-Forward 滚动训练的 A 股量化选股系统，支持每日自动化运行和 GitHub Actions 定时推送。

## 回测表现（2023-01 ~ 2026-03）

| 指标 | 数值 |
|------|------|
| 年化收益率 | 161.85% |
| 夏普比率 | 2.729 |
| 最大回撤 | -26.44% |
| Calmar 比率 | 6.121 |
| 胜率 | 54.67% |
| 盈亏比 | 1.798 |

## 快速开始

```bash
pip install -r requirements.txt

# 首次：全量下载数据（约30-60分钟）
python src/py00_fetch_stock_data.py

# 完整训练流水线（Walk-Forward + 回测）
./scripts/run_model.sh

# 日常：增量更新 + 快速预测（约30秒）
./scripts/run_daily.sh

# 盘后复盘（16:00后）
./scripts/run_daily_review.sh
```

## 项目结构

```
src/
├── py00_fetch_stock_data.py   数据获取（baostock，全量/增量/断点续传）
├── py01_data_loader.py        数据清洗（主板过滤、ST剔除、流动性筛选）
├── py02_features.py           特征工程（60+技术/动量/截面特征）
├── py03_model.py              LightGBM Ensemble 训练与预测
├── py04_backtest.py           回测引擎（T+1、涨跌停、交易成本）
├── py05_today.py              今日交易决策（生成TOP5买入建议）
├── py06_report.py             可视化报告（净值、回撤、热力图）
└── py08_review.py             盘后复盘（预测 vs 实际对比）
scripts/
├── run_daily.sh               日常快速预测
├── run_model.sh               完整重训练
└── run_daily_review.sh        盘后复盘
.github/workflows/
└── daily_report.yml           每个交易日北京时间7:30自动运行 + 邮件推送
```

## GitHub Actions

仓库已配置定时工作流，**每个交易日北京时间 7:30 自动运行**，完成数据更新 → 特征 → 预测 → 策略报告 → 邮件推送全流程。

**邮件推送**需在 Settings → Secrets and variables → Actions 中配置：
`EMAIL_SERVER` / `EMAIL_PORT` / `EMAIL_USERNAME` / `EMAIL_PASSWORD` / `EMAIL_RECIPIENT`

首次启用：在 GitHub Actions 页面手动触发一次 workflow 即可（自动全量初始化）。

## 详细文档

完整的模块说明、特征工程详解、模型设计、回测原理、常见问题等，请参阅：

- **[docs/project.md](docs/project.md)** — 项目完整技术文档
- **[docs/backtest.md](docs/backtest.md)** — 策略设计原始需求文档

## 风险提示

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。回测结果不代表未来收益。
