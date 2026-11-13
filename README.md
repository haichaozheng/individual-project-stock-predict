# Week11 案例：用上证指数预测次日收益率——线性回归完整流程

> 对应课堂 notebook：`../machine learning for stock prediction.ipynb`（§prediction tasks → §gradient descent → §线性回归）
>
> 课堂用 `gdp_satisfaction.csv`（GDP 与生活满意度）演示线性回归。本案例把同样的流程**原样搬到金融数据上**：用上证指数日线构造技术指标特征，预测次日收益率。学生照着改，就能接到自己的项目数据。

## 案例问题

> **上证指数明天涨多少？顺带对比：明天的振幅能预测吗？**

不是"涨不涨"（那是 W12 KNN 分类的任务），而是**涨多少**——一个回归问题。用历史日线的**技术指标 + 基本面因子**（均线、波动率、振幅、换手率、PE、PB 等）作为特征，用线性回归同时预测两个标签：

- **次日收益率** `next_return`：测试 R² ≈ 0（甚至为负）——收益被噪声主导，几乎不可预测
- **次日振幅** `next_range`：测试 R² ≈ 0.3——波动有聚集性（大波动后面跟大波动），可以预测

**同一批特征、同一个流程，可预测性天差地别**——这是本案例最重要的课堂体验。

## 学习与运行指南

### 三种使用方式

| 方式 | 适合谁 | 命令 | 看什么 |
|------|--------|------|--------|
| 交互探索 | 深入理解原理 | `streamlit run app.py` | 调参数（特征选择/学习率/split比例）→ 实时看图表变化 |
| 一键复现 | 看完整流程 | `python linear_regression.py` | 控制台输出每步指标 + output/ 目录生成 4 张图 |
| 看预置结果 | 快速浏览 | 打开 `sse_daily.csv` 看数据 | 表格数据，无需运行代码 |

### 建议学习顺序

1. 先读上方"案例问题"——带着问题往下看
2. 启动 `streamlit run app.py`，先用默认参数看一遍结果
3. 逐步调参数：取消某个特征看 R² 怎么变；调学习率看 SGD 是否发散
4. 跑 `python linear_regression.py`，对照控制台输出理解每一步
5. 读"改造指引"，接到自己的项目数据

### 前置条件

- 数据：`sse_daily.csv` 已预置在本目录，无需 tushare token 即可运行
- 如需重新获取数据：`python fetch_data.py`（需在项目根目录 `.env` 中配置 `TUSHARE_TOKEN`）
- 环境：项目 `.venv` 已激活，依赖已安装（`pip install -r requirements.txt`）

## 为什么需要这个案例？

| 课堂 notebook | 本案例 |
|---|---|
| `gdp_satisfaction.csv`：29 个国家，1 个特征（GDP），1 个标签（生活满意度） | 上证指数 20+ 年日线+因子，10 个特征，1 个标签（次日收益率） |
| 玩具数据，学生难以迁移 | 金融数据，直接对应学生项目场景 |
| 特征少，看不到特征工程的必要性 | 多特征，自然引出 W12/W13 的特征选择与重要性 |
| 数据量小，过拟合不明显 | 样本量大（5500+），train/test 差距可见 |
| 无量纲问题 | 特征量级差异巨大（均线数千 vs 收益率 ±0.1），引出归一化 |

## 目录结构

```
cases/
├── README.md              ← 本文件
├── requirements.txt       ← 依赖
├── fetch_data.py          ← 用 tushare 获取上证指数日线+因子并落盘（本目录自带数据，非首次无需运行）
├── sse_daily.csv          ← 预置数据（上证指数日线+因子，fetch_data.py 生成）
├── linear_regression.py   ← 五步流程：数据采集 → 可视化探索 → 划分与预处理 → 建模 → 评估（双目标）
├── app.py                 ← Streamlit 交互探索器：调参数实时看图表变化
├── fonts/
│   └── NotoSansCJKsc-Regular.otf  ← 自带中文字体（Noto Sans CJK，OFL 开源协议），
│                                    保证 Linux/Streamlit Cloud 上图表中文不乱码
└── output/                ← 图表输出目录（运行后生成）
    ├── price_trend.png                        价格走势
    ├── next_return_dist.png                   次日收益率分布
    ├── next_range_dist.png                    次日振幅分布
    ├── scatter_return1d_vs_next.png           当日收益率 vs 次日收益率散点
    ├── correlation_heatmap.png                特征相关性热力图（含双标签）
    ├── regression_fit_next_return.png         收益预测 vs 真实
    ├── regression_fit_next_range.png          振幅预测 vs 真实
    ├── sgd_vs_sklearn_next_return.png         SGD 损失曲线（收益）
    ├── sgd_vs_sklearn_next_range.png          SGD 损失曲线（振幅）
    ├── overfitting_check_next_return.png      过拟合自检（收益）
    └── overfitting_check_next_range.png       过拟合自检（振幅）
```

## 运行

```bash
# 在仓库根目录激活虚拟环境
.\.venv\Scripts\activate

# 安装依赖
pip install -r data-analysis/weeks/week11/cases/requirements.txt

# 数据已预置（sse_daily.csv），直接运行：

# 一键复现（控制台输出 + output/ 图表）
python data-analysis/weeks/week11/cases/linear_regression.py

# 或交互探索（Streamlit 网页）
streamlit run data-analysis/weeks/week11/cases/app.py

# 如需重新获取数据（需 TUSHARE_TOKEN）：
python data-analysis/weeks/week11/cases/fetch_data.py
```

输出：控制台打印各阶段指标，`output/` 目录保存图表。

### 部署到 Streamlit Cloud 的注意事项

- 图表中的中文依赖 `fonts/NotoSansCJKsc-Regular.otf`（已随本目录提供，OFL 开源协议）。
  脚本会自动检测并注册该字体；**部署时务必把它连同代码一起提交到 Git 仓库**，
  否则云端 Linux 没有微软雅黑/黑体，图内中文会显示为方框。
- 页面正文（非图表）的中文由浏览器渲染，与字体文件无关，无需额外处理。
- `.csv` 数据已预置，云端运行不需要 `TUSHARE_TOKEN`，也不要把 `.env` 提交到仓库。

## 数据

| 文件 | 内容 |
|---|---|
| `sse_daily.csv` | 上证指数（000001.SH）日线+因子，2004-01 至今 |

> 因子接口 `index_dailybasic` 对 000001.SH 最早覆盖到 2004-01-02，故数据从 2004 年开始，保证特征完整。`index_dailybasic` 单次最多返回 3000 条，`fetch_data.py` 已按年分页拉取。

字段（`fetch_data.py` 从 tushare `index_daily` + `index_dailybasic` 获取）：

| 字段 | 接口 | 说明 |
|---|---|---|
| `trade_date` | index_daily | 交易日期 |
| `open` / `high` / `low` / `close` | index_daily | 开高低收 |
| `vol` | index_daily | 成交量（手） |
| `amount` | index_daily | 成交额（千元） |
| `pct_chg` | index_daily | 涨跌幅（%） |
| `turnover_rate` | index_dailybasic | 换手率（%） |
| `pe` | index_dailybasic | 指数市盈率 |
| `pb` | index_dailybasic | 指数市净率 |

## 特征构造

`linear_regression.py` 构造以下 12 个特征：

| 特征 | 计算 | 直觉 |
|---|---|---|
| `ma_5` | 5 日收盘均价 | 短期趋势 |
| `ma_20` | 20 日收盘均价 | 中期趋势 |
| `ma_diff` | ma_5 − ma_20 | 金叉/死叉信号 |
| `vol_ratio` | 当日成交量 / 5 日均量 | 量价配合 |
| `volatility_5` | 5 日收益率标准差 | 近期波动 |
| `return_1d` | 当日涨跌幅 | 短期动量 |
| `return_5d` | 5 日累计收益率 | 中期动量 |
| `range_1d` | 当日振幅 (high−low)/low | **振幅预测的最强特征**（波动聚集性） |
| `range_5` | 5 日平均振幅 | 近期波动水平 |
| `turnover_rate` | 换手率（接口直取） | 情绪热度/分歧程度 |
| `pe` | 指数市盈率（接口直取） | 估值水平 |
| `pb` | 指数市净率（接口直取） | 估值水平 |

**双标签**：

| 标签 | 计算 | 测试 R²（时间切分） |
|---|---|---|
| `next_return` 次日收益率 | close[t+1]/close[t] − 1 | ≈ 0（收益几乎不可预测） |
| `next_range` 次日振幅 | (high[t+1]−low[t+1])/low[t+1] | ≈ 0.3（波动聚集性，可预测） |

## 完整流程（五步法）

| 步骤 | 课堂 notebook | 本案例 | 对应代码 |
|---|---|---|---|
| 1. 数据采集与特征构造 | `gdp_satisfaction.csv` | `sse_daily.csv`（日线+因子）+ 特征构造 | `load_and_engineer()` |
| 2. 可视化探索 | GDP vs 满意度散点图 | 价格走势、双标签分布、因子-标签散点、相关性热力图 | `explore_visualize()` |
| 3. 数据集划分与预处理 | `train_test_split` | **先按时间切（前 80% 训练），再用训练集拟合 StandardScaler 分别变换训练/测试集** | `split_and_preprocess()` |
| 4. 模型构建 | `LinearRegression().fit()` + `gd()` 填空 | sklearn 回归 + 手写 SGD（共用第 3 步的标准化特征），对两个标签各跑一遍 | `sklearn_fit()` / `sgd_fit()` |
| 5. 模型评估 | R² / MSE | 训练/测试指标、预测 vs 真实散点、双目标 R² 对比、过拟合自检（全量 / 随机 / 时间三种 split） | `evaluate()` / `overfitting_check()` |

### 关键设计 1：时间序列的 train/test split

课堂用 `train_test_split(random_state=42)` 随机打乱。本案例**按时间切**——前 80% 训练、后 20% 测试——因为金融数据有强时序性，随机打乱会造成数据泄漏（用未来信息预测过去）。这一点在 W12 数据泄漏一节会正式讲，这里先埋下。

### 关键设计 2：归一化在切分之后，参数只来自训练集

10 个特征量级差异巨大（`ma_5` 数千、`return_1d` ±0.1、`pe` 十几），必须归一化。但顺序有讲究：

```
✅ 正确：先按时间切分 → scaler.fit(训练集) → 分别 transform 训练集/测试集
❌ 错误：先对全量数据归一化 → 再切分
```

错误做法里，归一化参数（均值、标准差）混入了测试集（未来）的信息——训练过程等于偷看了未来，评估结果虚高，实盘失效。这是机器学习预处理的铁律，本案例 `split_and_preprocess()` 严格按正确顺序实现，并打印 scaler 学到的均值/标准差供课堂检查。

## 与课堂 notebook 的对应关系

| notebook 单元 | 本案例代码 |
|---|---|
| §prediction tasks + §what is ML | README"案例问题"部分 |
| §gradient descent | `sgd_fit()`：完整实现梯度下降 |
| §train/test split | `split_and_preprocess()`：时间序列分割 + 归一化 |
| §线性回归 sklearn | `sklearn_fit()` |
| `gd()` 函数填空 | `sgd_fit()`（完整版，可对照 `code.docx`） |
| `gdp_satisfaction.csv` | `sse_daily.csv` + 技术指标与基本面因子 |

## 改造指引（接到自己的项目数据）

1. **换数据**：`fetch_data.py` 里的 `ts_code` 换成你的股票/指数（个股用 `daily` + `daily_basic` 接口，因子字段更丰富），或直接读 W4 落盘的 CSV
2. **换特征**：在 `load_and_engineer()` 里增减特征列
3. **换标签**：把 `next_return` 换成你要预测的目标（如 5 日收益率、波动率、是否涨停）
4. **保持流程不变**：load → explore → split & preprocess → fit → evaluate

## 教学要点

- **双目标对比是核心体验**：同一批特征、同一个流程——收益率测试 R² ≈ 0，振幅测试 R² ≈ 0.3。问学生："为什么振幅能预测、收益不能？"（波动聚集性 vs 噪声主导）——建模前先想清楚要预测的量有没有可预测的成分
- **fit 不是魔法**：手写 SGD 让学生看到 `fit` 就是用梯度下降反复更新参数——`code.docx` 的答案可以对照
- **先切分，再归一化**：scaler 参数只来自训练集。先归一化再切分 = 把未来信息泄进训练。可以现场演示错误做法，看测试分虚高
- **时间序列不能随机 split**：与课堂随机 split 形成对比，为 W12 数据泄漏埋伏笔
- **R² 可能是负的**：收益预测的测试 R² 接近 0 甚至为负（模型还不如直接猜均值）是正常的——这正是"预测很难"的定量表述
- **特征多≠更好**：12 个特征里 ma_5 / ma_20 / pe 高度相关（多重共线性），有几个可能没用，为 W13 特征重要性埋伏笔

## 密钥安全

- 如使用 `.env`，已在 `.gitignore`，不要提交到仓库
- 不要在代码中硬编码 token
