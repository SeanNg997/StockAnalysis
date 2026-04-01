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

### 首次运行（全量初始化）

```bash
# 1. 全量下载数据（约5000只股票×10年，首次耗时30-60分钟）
python src/py00_fetch_stock_data.py

# 2. 完整流水线（清洗 → 特征 → 模型训练 → 回测 → 策略 → 报告）
./scripts/run_daily.sh --full
```

### 日常快速运行

```bash
# 增量更新 + 快速预测（2-5分钟）
# 使用缓存的pkl文件和最新K线数据
./scripts/run_daily.sh

# 等效于：增量数据 → 清洗 → 特征 → 快速预测 → 生成策略
```

### 完整重训练（可选，周/月运行一次）

```bash
# 重新训练Walk-Forward模型并回测（30-90分钟）
./scripts/run_daily.sh --full

# 等效于：增量数据 → 清洗 → 特征 → 重训练 → 回测 → 策略 + 报告图表
```

### 单只股票分析（可选）

```bash
# 生成单只股票的策略报告
./scripts/run_daily.sh 600000        # 快速模式
./scripts/run_daily.sh --full 600000 # 完整模式（包含详细图表）
```

策略报告输出：
- 全市场：`output/today_strategy.md`
- 单只股票：`output/today_strategy_{stock_code}.md`
- 历史存档：`output/history/strategy_{date}.md`

## GitHub Actions 自动化

仓库已配置 GitHub Actions，**每个交易日北京时间 8:00 自动运行**（UTC 00:00）：

### 工作流程

1. **数据更新**（增量/全量）
   - 检测是否有缓存数据
   - 首次运行：全量下载过去10年数据
   - 后续运行：增量更新最新交易日数据

2. **数据处理**
   - 数据清洗 + 特征工程
   - Walk-Forward 模型训练
   - 快速预测 + 生成策略报告

3. **输出和归档**
   - 生成每日策略报告 `output/today_strategy.md`
   - 按日期归档到 `output/history/`
   - 可选邮件推送通知

### 缓存管理

GitHub Actions 使用 `actions/cache` 机制缓存以下文件，避免重复下载和计算：

- `data/mainboard_clean.pkl` - 清洗后的主板股票数据
- `data/features.pkl` - 特征工程结果
- `data/predictions.pkl` - 模型预测结果
- `data/.last_date.txt` - 最后更新日期
- `data/Stock_dailyK_*.csv` - 月度K线数据（节省网络带宽）

### 邮件推送设置（可选）

在 **Settings → Secrets and variables → Actions** 中添加以下环境变量：

| Secret | 说明 | 示例 |
|--------|------|------|
| `EMAIL_SERVER` | SMTP 服务器 | `smtp.gmail.com` |
| `EMAIL_PORT` | SMTP 端口 | `587` |
| `EMAIL_USERNAME` | 发件人账户 | `your-email@gmail.com` |
| `EMAIL_PASSWORD` | 邮箱密码/应用密码 | `app-password` |
| `EMAIL_RECIPIENT` | 收件人邮箱 | `recipient@gmail.com` |

### 首次启用工作流

1. **选项 A**：手动触发 workflow
   - 在 GitHub 页面 → Actions → Daily Stock Strategy Report → Run workflow
   - workflow 会自动下载全量数据并生成缓存

2. **选项 B**：本地生成缓存
   ```bash
   python src/py00_fetch_stock_data.py  # 全量下载
   ./scripts/run_daily.sh --full         # 完整流水线
   git push                              # 提交代码（缓存由Actions维护）
   ```

### 手动触发工作流

在任何时刻手动运行 workflow（不受定时限制）：
- GitHub 页面 → Actions → Daily Stock Strategy Report → Run workflow

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

## 常见问题

### 首次运行

**Q：为什么首次运行很慢？**
- A：首次需要从 baostock 下载10年全量数据（约140个月，5000+只股票），耗时30-60分钟。后续运行使用缓存加速。

**Q：GitHub Actions 首次运行失败？**
- A：首次 workflow 需要生成 pkl 缓存。可选：
  1. 本地运行 `python src/py00_fetch_stock_data.py && ./scripts/run_daily.sh --full` 生成缓存后 push
  2. 或在 GitHub Actions 页面手动触发一次 workflow

### 数据更新

**Q：CSV 文件为什么这么大？**
- A：项目缓存所有月度 CSV（`data/Stock_dailyK_*.csv`），便于快速增量更新。如需节省空间，删除早期 CSV 文件，下次运行会重新下载。

**Q：增量更新失败提示"网络错误"？**
- A：检查网络连接和 baostock 服务可用性。如果长期不可用，删除 `data/.download_status.txt` 重试全量下载。

### 模型预测

**Q：快速预测和完整训练的区别？**
- A：
  - **快速预测**（`py07_quick_predict.py`）：使用缓存的历史模型和最新K线，2-5分钟完成，适合日常使用
  - **完整训练**（`py03_model.py + py04_backtest.py`）：重新Walk-Forward训练和回测，30-90分钟，适合周/月复盘

**Q：预测文件 `predictions.pkl` 坏了怎么办？**
- A：删除 `data/predictions.pkl` 和 `data/features.pkl`，重新运行数据流水线。

### 邮件推送

**Q：邮件推送不工作？**
- A：检查：
  1. Secrets 是否正确设置（特别是 `EMAIL_RECIPIENT` 不能为空）
  2. SMTP 服务器和端口是否正确（Gmail/腾讯企业邮箱配置参考官方文档）
  3. 邮箱密码是否为"应用密码"而非账户密码（有些服务要求）

**Q：历史报告保留在哪？**
- A：每日报告自动存档到 `output/history/` 目录，以日期命名（`strategy_YYYY-MM-DD.md`）。

## 系统依赖

- Python >= 3.8（推荐 3.11+）
- macOS / Linux / Windows 均可运行
- 网络连接（首次下载数据）

## 项目文件说明

| 文件 | 描述 |
|------|------|
| `py00_fetch_stock_data.py` | baostock 数据获取（全量/增量）|
| `py01_data_loader.py` | 数据清洗与过滤 |
| `py02_features.py` | 特征工程（60+技术面特征） |
| `py03_model.py` | LightGBM Ensemble 训练 |
| `py04_backtest.py` | Walk-Forward 回测 |
| `py05_today.py` | 生成今日策略报告 |
| `py06_report.py` | 可视化报告（净值曲线、回撤、热力图） |
| `py07_quick_predict.py` | 快速预测（日常使用） |
| `run_daily.sh` | 一键运行脚本 |

## 风险提示

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。回测结果不代表未来收益。
