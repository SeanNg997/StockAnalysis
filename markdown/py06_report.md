# py06_report.py - 报告与可视化模块详细解释

## 文件概述
该文件负责生成回测结果的各种可视化图表，包括累计收益率曲线、回撤曲线、每日持仓数量曲线、月度收益热力图和特征重要性分析。

## 代码逐行解释

### 1-10 行：文件头部注释

```python
"""
py06_report.py — 报告与可视化模块
====================================
职责：
1. 绘制累计收益率曲线（策略 vs 基准）
2. 回撤曲线
3. 每日持仓数量曲线
4. 月度收益热力图
5. 特征重要性
"""
```

- **1-10行**：文件头部的文档字符串，说明了该模块的名称和主要职责，包括绘制五种不同类型的图表。

### 12-28 行：导入必要的库和配置

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings

from config import CONFIG

warnings.filterwarnings('ignore')

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
```

- **12行**：导入pandas库，用于数据处理和分析。
- **13行**：导入numpy库，用于数值计算。
- **14行**：导入matplotlib库，用于数据可视化。
- **15行**：设置matplotlib使用非交互式后端'Agg'，适合生成图片文件而不需要显示。
- **16行**：导入matplotlib的pyplot模块，用于创建图表。
- **17行**：导入matplotlib的dates模块，用于日期格式化。
- **18行**：导入os模块，用于文件路径操作。
- **19行**：导入warnings模块，用于控制警告信息。
- **21行**：从config模块导入CONFIG配置字典。
- **23行**：忽略所有警告信息，使输出更干净。
- **26行**：配置matplotlib使用中文字体，确保图表中的中文能正确显示。
- **27行**：确保负号能正确显示。

### 29-34 行：路径和参数设置

```python
BASE_DIR = CONFIG['paths']['BASE_DIR']
BACKTEST_OUTPUT_DIR = CONFIG['paths']['BACKTEST_OUTPUT_DIR']
FEATURE_PKL = CONFIG['paths']['FEATURE_PKL']

INITIAL_CAPITAL = CONFIG['backtest']['INITIAL_CAPITAL']
```

- **29行**：从CONFIG中获取基础目录路径。
- **30行**：从CONFIG中获取回测输出目录路径。
- **31行**：从CONFIG中获取特征数据 pickle 文件路径。
- **34行**：从CONFIG中获取初始资金配置。

### 36-67 行：绘制累计收益率曲线函数

```python
def plot_equity_curve(daily_df: pd.DataFrame):
    """绘制累计收益率曲线"""
    print("绘制收益率曲线...")
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(daily_df['date'])
    strategy_pct = (daily_df['portfolio_value'] / INITIAL_CAPITAL - 1) * 100

    ax.plot(dates, strategy_pct, label='策略收益率', color='#e74c3c', linewidth=1.5)
    ax.fill_between(dates, strategy_pct, 0,
                    where=(strategy_pct >= 0), color='#e74c3c', alpha=0.1)
    ax.fill_between(dates, strategy_pct, 0,
                    where=(strategy_pct < 0), color='#2ecc71', alpha=0.1)

    # 基准线（0%）
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='基准(0%)')

    ax.set_title('A股量化策略 — 累计收益率曲线', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('累计收益率 (%)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(BACKTEST_OUTPUT_DIR, 'equity_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")
```

- **36行**：定义`plot_equity_curve`函数，接收一个DataFrame参数`daily_df`，包含每日回测数据。
- **37行**：函数的文档字符串，说明函数的作用是绘制累计收益率曲线。
- **38行**：打印提示信息，表明正在绘制收益率曲线。
- **39行**：创建一个新的图表，设置图表大小为14x6英寸。
- **41行**：将`date`列转换为 datetime 类型。
- **42行**：计算策略的累计收益率（百分比），通过当前组合价值除以初始资金再减1，然后乘以100。
- **44行**：绘制策略收益率曲线，使用红色线条，线宽1.5。
- **45-46行**：当收益率大于等于0时，在曲线下方填充淡红色。
- **47-48行**：当收益率小于0时，在曲线下方填充淡绿色。
- **51行**：绘制一条水平基准线（0%），使用灰色虚线。
- **53行**：设置图表标题为'A股量化策略 — 累计收益率曲线'，字体大小14，粗体。
- **54行**：设置x轴标签为'日期'。
- **55行**：设置y轴标签为'累计收益率 (%)'。
- **56行**：设置y轴刻度格式为带正负号的百分比形式。
- **57行**：添加图例，字体大小11。
- **58行**：显示网格，透明度0.3。
- **59行**：设置x轴日期格式为'年-月'。
- **60行**：设置x轴主刻度为每3个月一个。
- **61行**：将x轴标签旋转45度，避免重叠。
- **62行**：自动调整图表布局，确保所有元素都能显示。
- **64行**：构建保存路径，将图表保存到回测输出目录。
- **65行**：保存图表，设置分辨率为150 dpi。
- **66行**：关闭图表，释放内存。
- **67行**：打印保存路径信息。

### 70-94 行：绘制回撤曲线函数

```python
def plot_drawdown(daily_df: pd.DataFrame):
    """绘制回撤曲线"""
    print("绘制回撤曲线...")
    fig, ax = plt.subplots(figsize=(14, 4))

    dates = pd.to_datetime(daily_df['date'])
    cummax = daily_df['portfolio_value'].cummax()
    drawdown = (daily_df['portfolio_value'] - cummax) / cummax * 100

    ax.fill_between(dates, drawdown, 0, color='#e74c3c', alpha=0.4)
    ax.plot(dates, drawdown, color='#e74c3c', linewidth=0.8)

    ax.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('回撤 (%)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(BACKTEST_OUTPUT_DIR, 'drawdown_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")
```

- **70行**：定义`plot_drawdown`函数，接收一个DataFrame参数`daily_df`。
- **71行**：函数的文档字符串，说明函数的作用是绘制回撤曲线。
- **72行**：打印提示信息，表明正在绘制回撤曲线。
- **73行**：创建一个新的图表，设置图表大小为14x4英寸。
- **75行**：将`date`列转换为 datetime 类型。
- **76行**：计算组合价值的累积最大值，用于计算回撤。
- **77行**：计算每日回撤（百分比），公式为（当前组合价值 - 累积最大值）/ 累积最大值 * 100。
- **79行**：在日期和回撤值之间填充红色区域，透明度0.4。
- **80行**：绘制回撤曲线，使用红色线条，线宽0.8。
- **82行**：设置图表标题为'回撤曲线'，字体大小14，粗体。
- **83行**：设置x轴标签为'日期'。
- **84行**：设置y轴标签为'回撤 (%)'。
- **85行**：显示网格，透明度0.3。
- **86行**：设置x轴日期格式为'年-月'。
- **87行**：设置x轴主刻度为每3个月一个。
- **88行**：将x轴标签旋转45度。
- **89行**：自动调整图表布局。
- **91行**：构建保存路径。
- **92行**：保存图表，设置分辨率为150 dpi。
- **93行**：关闭图表。
- **94行**：打印保存路径信息。

### 97 行：获取最大持仓数量配置

```python
MAX_POSITIONS = CONFIG['backtest']['MAX_POSITIONS']  # 最大持仓数量（对应100%仓位）
```

- **97行**：从CONFIG中获取最大持仓数量配置，注释说明这对应100%仓位。

### 100-131 行：绘制每日仓位比例与交易次数函数

```python
def plot_positions(daily_df: pd.DataFrame):
    """绘制每日仓位比例与交易次数"""
    print("绘制持仓数量曲线...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    dates = pd.to_datetime(daily_df['date'])
    position_pct = daily_df['n_positions'] / MAX_POSITIONS * 100

    # 仓位百分比
    axes[0].bar(dates, position_pct, color='#3498db', alpha=0.7, width=1)
    axes[0].set_ylabel('仓位比例 (%)')
    axes[0].set_title('每日仓位比例与交易次数', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 110)
    axes[0].axhline(y=100, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1, label='满仓(100%)')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 交易次数
    axes[1].bar(dates, daily_df['n_trades'], color='#e67e22', alpha=0.7, width=1)
    axes[1].set_ylabel('交易次数')
    axes[1].set_xlabel('日期')
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(BACKTEST_OUTPUT_DIR, 'daily_positions.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")
```

- **100行**：定义`plot_positions`函数，接收一个DataFrame参数`daily_df`。
- **101行**：函数的文档字符串，说明函数的作用是绘制每日仓位比例与交易次数。
- **102行**：打印提示信息，表明正在绘制持仓数量曲线。
- **103行**：创建一个包含2个子图的图表，设置图表大小为14x6英寸，共享x轴。
- **105行**：将`date`列转换为 datetime 类型。
- **106行**：计算每日仓位比例（百分比），公式为持仓数量除以最大持仓数量再乘以100。
- **109行**：在第一个子图中绘制仓位比例的条形图，使用蓝色，透明度0.7，宽度1。
- **110行**：设置第一个子图的y轴标签为'仓位比例 (%)'。
- **111行**：设置第一个子图的标题为'每日仓位比例与交易次数'，字体大小14，粗体。
- **112行**：设置第一个子图的y轴范围为0到110。
- **113行**：在第一个子图中绘制一条水平参考线（100%），表示满仓状态。
- **114行**：设置第一个子图的y轴刻度格式为百分比形式。
- **115行**：在第一个子图中添加图例，字体大小9。
- **116行**：在第一个子图中显示网格，透明度0.3。
- **119行**：在第二个子图中绘制交易次数的条形图，使用橙色，透明度0.7，宽度1。
- **120行**：设置第二个子图的y轴标签为'交易次数'。
- **121行**：设置第二个子图的x轴标签为'日期'。
- **122行**：在第二个子图中显示网格，透明度0.3。
- **123行**：设置第二个子图的x轴日期格式为'年-月'。
- **124行**：设置第二个子图的x轴主刻度为每3个月一个。
- **125行**：将x轴标签旋转45度。
- **126行**：自动调整图表布局。
- **128行**：构建保存路径。
- **129行**：保存图表，设置分辨率为150 dpi。
- **130行**：关闭图表。
- **131行**：打印保存路径信息。

### 134-171 行：绘制月度收益热力图函数

```python
def plot_monthly_returns(daily_df: pd.DataFrame):
    """绘制月度收益热力图"""
    print("绘制月度收益热力图...")
    # 避免不必要的复制
    daily = daily_df.copy()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['daily_return'] = daily['portfolio_value'].pct_change()
    
    # 月度收益率
    monthly = daily.groupby([daily['date'].dt.year, daily['date'].dt.month])['daily_return'].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    # A股惯例：红涨绿跌，使用反转的 RdYlGn_r（正收益=红，负收益=绿）
    im = ax.imshow(monthly.values * 100, cmap='RdYlGn_r', aspect='auto', vmin=-10, vmax=10)

    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels([f'{m}月' for m in monthly.columns])
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index)

    # 在每个格子中标注数值
    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            val = monthly.values[i, j] * 100
            if not np.isnan(val) and val != 0:
                ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                        fontsize=9, color='white' if abs(val) >= 4 else 'black')

    plt.colorbar(im, ax=ax, label='月度收益率 (%)')
    ax.set_title('月度收益率热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(BACKTEST_OUTPUT_DIR, 'monthly_returns.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  保存: {path}")
```

- **134行**：定义`plot_monthly_returns`函数，接收一个DataFrame参数`daily_df`。
- **135行**：函数的文档字符串，说明函数的作用是绘制月度收益热力图。
- **136行**：打印提示信息，表明正在绘制月度收益热力图。
- **138行**：创建`daily_df`的副本，避免修改原始数据。
- **139行**：将`date`列转换为 datetime 类型。
- **140行**：计算每日收益率，使用pct_change()方法。
- **143-145行**：按年和月分组，计算月度收益率（复利计算），然后将结果转换为宽格式，缺失值填充为0。
- **147行**：创建一个新的图表，设置图表大小为14x5英寸。
- **149行**：使用imshow绘制热力图，将月度收益率乘以100转换为百分比，使用反转的RdYlGn_r颜色映射（A股惯例：红涨绿跌），设置颜色范围为-10到10。
- **151行**：设置x轴刻度位置，对应每个月份。
- **152行**：设置x轴刻度标签为'1月'、'2月'等格式。
- **153行**：设置y轴刻度位置，对应每一年。
- **154行**：设置y轴刻度标签为年份。
- **157-162行**：在热力图的每个格子中添加数值标签，格式为带正负号的百分比，当绝对值大于等于4时使用白色字体，否则使用黑色字体。
- **164行**：添加颜色条，标签为'月度收益率 (%)'。
- **165行**：设置图表标题为'月度收益率热力图'，字体大小14，粗体。
- **166行**：自动调整图表布局。
- **168行**：构建保存路径。
- **169行**：保存图表，设置分辨率为150 dpi。
- **170行**：关闭图表。
- **171行**：打印保存路径信息。

### 174-226 行：绘制特征重要性函数

```python
def plot_feature_importance(model_path=None):
    """
    绘制特征重要性（如果模型可用）
    这里使用一个简化方法：读取最新模型的特征重要性
    """
    # 由于walk-forward中模型没有持久化到文件，
    # 我们用一个简化方式：训练一个小模型获取特征重要性
    print("计算特征重要性...")
    try:
        import lightgbm as lgb

        df = pd.read_pickle(FEATURE_PKL)

        # 使用最近2年数据
        recent = df[df['date'] >= '2024-01-01'].copy()

        exclude = {'代码', '名称', 'date', 'open', 'high', 'low', 'close',
                    'volume', 'amount', 'turn', 'pctChg',
                    'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', 'label',
                    'ma_5', 'ma_10', 'ma_20', 'ma_60'}
        feature_cols = [c for c in df.columns if c not in exclude]

        X = recent[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = recent['label'].dropna()
        valid = y.index.intersection(X.index)
        X = X.loc[valid]
        y = y.loc[valid]

        # 快速训练
        dtrain = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression', 'metric': 'mae',
            'num_leaves': 31, 'learning_rate': 0.1, 'verbose': -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=100)

        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=True).tail(25)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance['feature'], importance['importance'], color='#3498db')
        ax.set_title('特征重要性 (Top 25 - Gain)', fontsize=14, fontweight='bold')
        ax.set_xlabel('重要性 (Gain)')
        plt.tight_layout()

        path = os.path.join(BACKTEST_OUTPUT_DIR, 'feature_importance.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  保存: {path}")
    except Exception as e:
        print(f"  特征重要性绘制失败: {e}")
```

- **174行**：定义`plot_feature_importance`函数，接收一个可选参数`model_path`，默认为None。
- **175-178行**：函数的文档字符串，说明函数的作用是绘制特征重要性，并解释由于walk-forward中模型没有持久化到文件，使用了一个简化方法。
- **180-181行**：注释说明使用简化方式训练一个小模型来获取特征重要性。
- **182行**：打印提示信息，表明正在计算特征重要性。
- **183行**：开始try-except块，捕获可能的异常。
- **184行**：导入lightgbm库，用于训练模型。
- **186行**：从FEATURE_PKL路径读取特征数据。
- **189行**：筛选2024年1月1日之后的数据，创建副本。
- **191-193行**：定义需要排除的列，包括基本信息、价格数据、原始指标和标签。
- **194行**：从所有列中排除上述列，得到特征列列表。
- **196行**：创建特征矩阵X，替换无穷值为NaN，然后填充为0。
- **197行**：创建目标变量y，删除缺失值。
- **198行**：计算X和y的索引交集，确保两者行数一致。
- **199-200行**：根据交集索引过滤X和y。
- **203行**：创建LightGBM训练数据集。
- **204-207行**：设置LightGBM模型参数，使用回归任务，MAE指标，31个叶子节点，学习率0.1，关闭 verbose 输出。
- **208行**：训练LightGBM模型，使用100轮迭代。
- **210-213行**：创建特征重要性DataFrame，使用'gain'类型的重要性，按重要性排序，取后25个（即最重要的25个特征）。
- **215行**：创建一个新的图表，设置图表大小为10x8英寸。
- **216行**：使用水平条形图绘制特征重要性，使用蓝色。
- **217行**：设置图表标题为'特征重要性 (Top 25 - Gain)'，字体大小14，粗体。
- **218行**：设置x轴标签为'重要性 (Gain)'。
- **219行**：自动调整图表布局。
- **221行**：构建保存路径。
- **222行**：保存图表，设置分辨率为150 dpi。
- **223行**：关闭图表。
- **224行**：打印保存路径信息。
- **225-226行**：捕获异常，打印错误信息。

### 229-247 行：生成所有报告和图表函数

```python
def generate_all_reports():
    """生成所有报告和图表"""
    os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)

    daily_path = os.path.join(BACKTEST_OUTPUT_DIR, 'backtest_daily.csv')
    if not os.path.exists(daily_path):
        print("错误: 未找到回测结果文件，请先运行 py05_backtest.py")
        return

    daily_df = pd.read_csv(daily_path)
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    plot_equity_curve(daily_df)
    plot_drawdown(daily_df)
    plot_positions(daily_df)
    plot_monthly_returns(daily_df)
    plot_feature_importance()

    print(f"\n✅ 所有图表已保存至 {BACKTEST_OUTPUT_DIR}/")
```

- **229行**：定义`generate_all_reports`函数，无参数。
- **230行**：函数的文档字符串，说明函数的作用是生成所有报告和图表。
- **231行**：创建回测输出目录，如果已存在则忽略。
- **233行**：构建回测每日数据文件路径。
- **234-236行**：检查文件是否存在，如果不存在则打印错误信息并返回。
- **238行**：读取回测每日数据文件。
- **239行**：将`date`列转换为 datetime 类型。
- **241行**：调用`plot_equity_curve`函数，绘制累计收益率曲线。
- **242行**：调用`plot_drawdown`函数，绘制回撤曲线。
- **243行**：调用`plot_positions`函数，绘制每日仓位比例与交易次数。
- **244行**：调用`plot_monthly_returns`函数，绘制月度收益热力图。
- **245行**：调用`plot_feature_importance`函数，绘制特征重要性。
- **247行**：打印所有图表已保存的信息。

### 250-251 行：主程序入口

```python
if __name__ == '__main__':
    generate_all_reports()
```

- **250-251行**：如果直接运行该文件，则调用`generate_all_reports`函数生成所有报告和图表。

## 功能总结

py06_report.py模块提供了以下功能：

1. **绘制累计收益率曲线**：展示策略的累计收益表现，与基准（0%）进行比较，使用颜色填充区分正负收益。

2. **绘制回撤曲线**：展示策略的最大回撤情况，帮助评估风险。

3. **绘制每日仓位比例与交易次数**：展示策略的仓位使用情况和交易频率。

4. **绘制月度收益热力图**：以热力图形式展示各年各月的收益情况，使用红涨绿跌的A股惯例。

5. **绘制特征重要性**：通过训练一个简化的LightGBM模型，分析并展示最重要的25个特征。

所有图表都会保存到回测输出目录，为策略评估和分析提供直观的可视化工具。