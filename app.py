"""Week11 Streamlit 探索器：线性回归预测次日收益率。

完全自包含——数据、特征工程、模型训练全部在本文件内。
交互调参：特征选择 / train-test 比例 / 学习率 / SGD 迭代次数。

运行：streamlit run app.py
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 中文显示：优先用仓库自带的开源字体（Noto Sans CJK，OFL 协议可自由分发）。
# 本地 Windows 有系统字体也能画，但 Streamlit Cloud 等 Linux 环境没有
# 微软雅黑/黑体，不自带字体的话图里中文会变成方框。
_BUNDLED_FONT = Path(__file__).resolve().parent / "fonts" / "NotoSansCJKsc-Regular.otf"
if _BUNDLED_FONT.exists():
    fm.fontManager.addfont(str(_BUNDLED_FONT))
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ALL_FEATURES = [
    # 技术指标（日线自算）
    "ma_5", "ma_20", "ma_diff", "vol_ratio",
    "volatility_5", "return_1d", "return_5d",
    "range_1d", "range_5",
    # 基本面/交易因子（index_dailybasic）
    "turnover_rate", "pe", "pb",
]
FEATURE_LABELS = {
    "ma_5": "5日均线", "ma_20": "20日均线", "ma_diff": "均线差",
    "vol_ratio": "量比", "volatility_5": "5日波动率",
    "return_1d": "当日收益率", "return_5d": "5日收益率",
    "range_1d": "当日振幅", "range_5": "5日均振幅",
    "turnover_rate": "换手率", "pe": "市盈率PE", "pb": "市净率PB",
}
LABEL_COLS = {
    "next_return": "次日收益率",
    "next_range": "次日振幅",
}


def load_and_engineer() -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parent / "sse_daily.csv"
    if not csv_path.exists():
        st.error(f"未找到 {csv_path}，请先运行 `python fetch_data.py` 获取数据。")
        st.stop()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    factor_cols = ["turnover_rate", "pe", "pb"]
    if not set(factor_cols).issubset(df.columns):
        st.error(
            "sse_daily.csv 缺少因子列（旧版数据），"
            "请重新运行 `python fetch_data.py`。"
        )
        st.stop()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    close = df["close"]
    df["ma_5"] = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_diff"] = df["ma_5"] - df["ma_20"]
    df["vol_ratio"] = df["vol"] / df["vol"].rolling(5).mean()
    daily_return = close.pct_change()
    df["volatility_5"] = daily_return.rolling(5).std()
    df["return_1d"] = df["pct_chg"] / 100
    df["return_5d"] = close / close.shift(5) - 1
    df["range_1d"] = df["high"] / df["low"] - 1
    df["range_5"] = df["range_1d"].rolling(5).mean()
    df["next_return"] = close.shift(-1) / close - 1
    df["next_range"] = df["high"].shift(-1) / df["low"].shift(-1) - 1
    df = df.dropna(
        subset=ALL_FEATURES + list(LABEL_COLS)
    ).reset_index(drop=True)
    return df


st.set_page_config(page_title="线性回归探索器", layout="wide")
st.title("线性回归探索器：上证指数次日收益率预测")

df = load_and_engineer()

# ── 侧栏 ──
with st.sidebar:
    st.header("参数设置")
    label_col = st.radio(
        "预测目标（双目标对比）", list(LABEL_COLS),
        format_func=lambda x: LABEL_COLS[x],
        help="同一批特征：收益率几乎不可预测（R²≈0），"
             "振幅因波动聚集性可预测（R²≈0.3）。",
    )
    label_name = LABEL_COLS[label_col]
    selected_features = st.multiselect(
        "选择特征", ALL_FEATURES, default=ALL_FEATURES,
        format_func=lambda x: FEATURE_LABELS.get(x, x),
    )
    split_ratio = st.slider("训练集比例", 0.5, 0.9, 0.8, 0.05)
    normalize = st.checkbox(
        "归一化特征（StandardScaler）", value=True,
        help="先切分，再用训练集拟合均值/标准差，分别变换训练/测试集。"
             "关掉它对比一下：系数会被大量纲特征（ma_5/pe）主导。",
    )
    eta = st.slider("SGD 学习率", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
    n_iter = st.slider("SGD 迭代次数", 10000, 200000, 100000, 10000)

if len(selected_features) == 0:
    st.warning("请至少选择一个特征。")
    st.stop()

# ── Section 0: 变量说明 ──
with st.expander("📖 变量说明：预测目标与特征的定义、计算公式", expanded=False):
    st.markdown("#### 预测目标（y，二选一）")
    st.markdown(
        r"""
| 目标 | 名称 | 计算公式 | 含义 | 可预测性 |
|---|---|---|---|---|
| `next_return` | 次日收益率 | $r_{t+1} = \dfrac{close_{t+1}}{close_t} - 1$ | 今天收盘买入、明天收盘卖出的收益（不含交易成本） | 测试 R²≈0：被噪声主导，几乎不可预测 |
| `next_range` | 次日振幅 | $range_{t+1} = \dfrac{high_{t+1} - low_{t+1}}{low_{t+1}}$ | 明天盘中最高价比最低价高多少，衡量单日波动大小 | 测试 R²≈0.3：波动有聚集性，可预测 |

> ⚠ 注意方向：所有特征只用 **今天（t）及之前** 的信息，两个目标都用 **明天（t+1）** 的数据——这正是"用过去预测未来"，切分与预处理也据此设计。
"""
    )
    st.markdown("#### 特征（X，共 12 个，全部只用 t 日及之前的信息）")
    st.markdown(
        r"""
**技术指标（由日线自算，6+2 个）**

| 特征 | 名称 | 计算公式 | 直觉 |
|---|---|---|---|
| `ma_5` | 5日均线 | $MA_5 = \frac{1}{5}\sum_{i=0}^{4} close_{t-i}$ | 短期趋势水平 |
| `ma_20` | 20日均线 | $MA_{20} = \frac{1}{20}\sum_{i=0}^{19} close_{t-i}$ | 中期趋势水平 |
| `ma_diff` | 均线差 | $MA_5 - MA_{20}$ | >0 短期均线在长期上方（多头形态/金叉） |
| `vol_ratio` | 量比 | $\dfrac{vol_t}{\frac{1}{5}\sum_{i=0}^{4} vol_{t-i}}$ | 今日成交量相对近期放大（>1）还是萎缩（<1），量价配合 |
| `volatility_5` | 5日波动率 | $\sigma_5 = std(r_{t-4},\dots,r_t)$，日收益率标准差 | 近期波动大小 |
| `return_1d` | 当日收益率 | $r_t = \dfrac{close_t}{close_{t-1}} - 1$ | 今日涨跌（短期动量） |
| `return_5d` | 5日收益率 | $\dfrac{close_t}{close_{t-5}} - 1$ | 近一周累计涨跌（中期动量） |
| `range_1d` | 当日振幅 | $\dfrac{high_t - low_t}{low_t}$ | 今日盘中波动，**振幅预测的最强特征** |
| `range_5` | 5日均振幅 | $\frac{1}{5}\sum_{i=0}^{4} range_{t-i}$ | 近期波动水平 |

**基本面/交易因子（tushare `index_dailybasic` 接口直取，3 个）**

| 特征 | 名称 | 来源 | 直觉 |
|---|---|---|---|
| `turnover_rate` | 换手率（%） | index_dailybasic | 市场交投活跃度与分歧程度 |
| `pe` | 指数市盈率 | index_dailybasic | 估值水平（贵/便宜） |
| `pb` | 指数市净率 | index_dailybasic | 估值水平（贵/便宜） |
"""
    )
    st.info(
        "为什么量纲差异巨大还要先归一化？ma_5/ma_20 是几千的点位，"
        "return_1d 是 ±0.1 的小数，pe 是十几到几十——"
        "不归一化的话，系数量级完全不可比，梯度下降也会被大量纲特征主导。"
    )

# ── Section 1: 数据概览 ──
st.subheader("1. 数据概览")
c1, c2, c3 = st.columns(3)
c1.metric("样本数", f"{len(df)}")
c2.metric("起始日期", df["trade_date"].iloc[0].strftime("%Y-%m-%d"))
c3.metric("结束日期", df["trade_date"].iloc[-1].strftime("%Y-%m-%d"))
st.dataframe(df[["trade_date", "close"] + selected_features].head(10),
             use_container_width=True)

# ── Section 2: 标签分布与因子相关性（可视化探索）──
st.subheader("2. 标签分布与因子相关性（可视化探索）")

# 2.1 双标签分布并排——无论当前选了哪个目标，两个都展示，强化双目标对比
c1, c2 = st.columns(2)
for col_ax, (lab_col, lab_name) in zip((c1, c2), LABEL_COLS.items()):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(df[lab_col], bins=80, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linewidth=1)
    ax.set_xlabel(lab_name)
    ax.set_ylabel("天数")
    ax.set_title(f"{lab_name}分布\n均值 {df[lab_col].mean():.5f}，"
                 f"标准差 {df[lab_col].std():.5f}", fontsize=10)
    ax.grid(True, alpha=0.3)
    if lab_col == "next_range":
        # 振幅恒为正，横轴从 0 起
        ax.set_xlim(0, df[lab_col].max() * 1.05)
    fig.tight_layout()
    col_ax.pyplot(fig)

st.caption(
    "对比两个分布：次日收益率以 0 为中心近似对称（涨跌各半，噪声主导）；"
    "次日振幅恒为正且右偏（多数日子波动小，偶尔极端大波动）。"
    "分布形态本身就暗示了两个预测目标的难度差异。"
)

# 2.2 相关性热力图（当前所选特征 + 双标签）
corr_mat = df[selected_features + list(LABEL_COLS)].corr()
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_mat)))
ax.set_yticks(range(len(corr_mat)))
ax.set_xticklabels(corr_mat.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr_mat.index, fontsize=9)
for i in range(len(corr_mat)):
    for j in range(len(corr_mat)):
        val = corr_mat.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(val) > 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("特征与双标签的相关性热力图（随侧栏所选特征联动）")
fig.tight_layout()
st.pyplot(fig)

st.caption(
    "重点看最后两行（特征与标签的相关性）："
    "① range_1d / range_5 / volatility_5 与次日振幅相关最强（波动聚集性）；"
    "② 所有特征与次日收益率的相关都接近 0——单特征层面就看不到信号；"
    "③ 特征之间 pe 与 pb 高度相关（多重共线性），ma_5 / ma_20 同理，"
    "为 W13 特征重要性分析埋下伏笔。"
)

# ── Section 3: 散点图直觉 ──
# 每个目标配一个最具代表性的单特征：
#   收益 → 当日收益率（动量，corr≈0，体会"收益难预测"）
#   振幅 → 当日振幅（波动聚集性，corr≈0.5，体会"振幅可预测"）
SCATTER_FEATURES = {
    "next_return": ("return_1d", "当日收益率"),
    "next_range": ("range_1d", "当日振幅"),
}
scatter_col, scatter_name = SCATTER_FEATURES[label_col]
st.subheader(f"3. {scatter_name} vs {label_name}（单特征直觉）")
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df[scatter_col], df[label_col], alpha=0.15, s=10)
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel(scatter_name)
ax.set_ylabel(label_name)
corr = df[scatter_col].corr(df[label_col])
ax.set_title(f"相关系数 = {corr:.4f}")
ax.grid(True, alpha=0.3)
if label_col == "next_range":
    # 振幅恒为正，坐标轴从 0 起，不留空白负半轴
    ax.set_xlim(0, df[scatter_col].max() * 1.05)
    ax.set_ylim(0, df[label_col].max() * 1.05)
fig.tight_layout()
st.pyplot(fig)

# ── 准备数据（先切分，再归一化）──
split_idx = int(len(df) * split_ratio)
X_train = df.iloc[:split_idx][selected_features].values
y_train = df.iloc[:split_idx][label_col].values
X_test = df.iloc[split_idx:][selected_features].values
y_test = df.iloc[split_idx:][label_col].values

if normalize:
    # 归一化参数只在训练集上拟合——这是预处理的铁律：
    # 若先对全量数据归一化再切分，测试集信息会泄漏进训练过程。
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

# ── Section 4: sklearn 回归 ──
st.subheader("4. sklearn 线性回归")
model = LinearRegression()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


def metric_row(y_true, y_hat):
    """一组常用的回归评估指标。"""
    mse = mean_squared_error(y_true, y_hat)
    return {
        "R²": r2_score(y_true, y_hat),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_hat),
    }


metrics_df = pd.DataFrame({
    "训练集": metric_row(y_train, pred_train),
    "测试集": metric_row(y_test, pred_test),
})
train_r2 = metrics_df.loc["R²", "训练集"]
test_r2 = metrics_df.loc["R²", "测试集"]

st.dataframe(metrics_df.style.format("{:.6f}"), use_container_width=True)

with st.expander("📐 评估指标：公式与读法（R² / MSE / RMSE / MAE）", expanded=False):
    st.markdown(
        "先记住分工：**R² 回答“有没有真本事”（无量纲，以猜均值为基准）；"
        "RMSE / MAE 回答“错多少”（带量纲，能换算成实际含义）；"
        "MSE 是训练时真正在最小化的目标。**"
    )

    # ── R² ──
    st.markdown("### R²（决定系数）")
    st.markdown(
        r"分子 = 模型误差；分母 = **笨基准**的误差——不看任何特征、"
        r"闭眼猜均值 $\bar{y}$ 所犯的错："
    )
    st.latex(
        r"R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}"
        r"{\sum_i (y_i - \bar{y})^2}"
        r"= 1 - \frac{MSE_{model}}{Var(y_{test})}"
    )
    st.markdown(
        r"""R² 回答的问题：**模型比“猜均值”好多少？**

| R² | 含义 |
|---|---|
| 1 | 完美预测 |
| 0.27（本案例振幅） | 把笨基准的误差减少了 27% |
| 0 | 和闭眼猜均值一样好，特征白用了 |
| **< 0** | **比闭眼猜均值还差**——特征帮了倒忙 |
"""
    )
    st.markdown(
        r"""R² < 0 的两类常见原因：① 数据里没有信号（如股票次日收益——模型拟合到的
“规律”其实是训练集噪声，到测试集上反而把预测推离均值）；② 训练期与测试期分布
漂移，旧规律外推失效。**流程正确时，负 R² 是“数据无信号”的定量表述，
不是代码 bug。**"""
    )

    # ── MSE ──
    st.markdown("### MSE（均方误差）")
    st.latex(r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
    st.markdown(
        "- 平方放大极端误差：错 2 倍 → 罚 4 倍，对异常值敏感\n"
        "- 单位是 y 的平方（如“百分比²”），不能直接解释\n"
        "- 线性回归的 `fit()` 就是在找让 MSE 最小的系数——"
        "它既是评估指标，也是训练目标"
    )

    # ── RMSE ──
    st.markdown("### RMSE（均方根误差）")
    st.latex(r"RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}")
    st.markdown(
        "- MSE 开根号，单位回到 y 本身，可解释性强\n"
        "- 读法：“平均一天的预测偏差是多少”\n"
        "- 例：本案例振幅 RMSE ≈ 0.006 → 平均偏差约 0.6 个百分点，"
        "可直接向业务方转述"
    )

    # ── MAE ──
    st.markdown("### MAE（平均绝对误差）")
    st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")
    st.markdown(
        "- 绝对值不平方，个别极端日的巨大误差不会被放大，比 RMSE 稳健\n"
        "- 在 0 处不可导，手写梯度下降最小化 MAE 需要特殊处理（次梯度）"
    )

    # ── 怎么配合用 ──
    st.markdown("### 四者怎么配合用")
    st.markdown(
        """
| 场景 | 用谁 | 原因 |
|---|---|---|
| 比较“模型有没有真本事” | R² | 无量纲，以猜均值为基准 |
| 向业务方解释误差大小 | RMSE / MAE | 有量纲，能换算成实际含义 |
| 数据里有极端值，怕被带偏 | MAE | 不平方，不放大极端误差 |
| 训练模型本身 | MSE | 凸且光滑，梯度下降友好 |
"""
    )

    st.markdown(
        "**进阶指标（后续课程展开）**：调整后 R²（W13 特征选择）· "
        "方向准确率/涨跌命中率（W12 分类视角）· "
        "IC 信息系数（W14 量化回测标配）"
    )

coef_df = pd.DataFrame({
    "特征": selected_features,
    "系数": model.coef_,
}).style.format({"系数": "{:.6f}"})
st.write("系数：")
st.dataframe(coef_df, use_container_width=True)

y_pred = model.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_pred, alpha=0.2, s=10)
if label_col == "next_range":
    # 振幅恒为正：两轴从 0 起，参考线也只画正区间
    top = max(y_test.max(), y_pred.max()) * 1.05
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.plot([0, top], [0, top], "r--", linewidth=1, label="y=x")
else:
    lim = max(abs(y_test).max(), abs(y_pred).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y=x")
ax.set_xlabel(f"真实{label_name}")
ax.set_ylabel(f"预测{label_name}")
ax.set_title(f"预测 vs 真实（测试集 R2={test_r2:.4f}）")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

# ── Section 5: 手写 SGD ──
st.subheader("5. 手写 SGD（梯度下降）")
with st.spinner("SGD 训练中…"):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    Xn_train = (X_train - mean) / std
    Xn_test = (X_test - mean) / std
    w = np.zeros(len(selected_features))
    b = 0.0
    losses = []
    for i in range(n_iter):
        pred = Xn_train @ w + b
        error = pred - y_train
        grad_w = 2 * (Xn_train.T @ error) / len(y_train)
        grad_b = 2 * error.mean()
        w = w - eta * grad_w
        b = b - eta * grad_b
        if i % max(1, n_iter // 20) == 0:
            losses.append((i, (error ** 2).mean()))

    sgd_pred = Xn_test @ w + b

sgd_metrics = metric_row(y_test, sgd_pred)
compare_df = pd.DataFrame({
    "sklearn（闭式解）": metrics_df["测试集"],
    "手写 SGD": sgd_metrics,
})
st.dataframe(compare_df.style.format("{:.6f}"), use_container_width=True)
st.caption(
    "两个模型在全部指标上几乎完全相等不是巧合：线性回归的损失函数是凸曲面，"
    "只有唯一的全局最低点。sklearn 用闭式解一步到达，"
    "手写梯度下降迭代收敛后也停在同一个点（真实差异在 1e-7 量级，"
    "显示时被四舍五入掩盖）。把迭代次数调小，就能看到两者拉开差距。"
    "公式见第 4 节的 📐 展开。"
)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot([l[0] for l in losses], [l[1] for l in losses], marker="o")
ax.set_xlabel("迭代次数")
ax.set_ylabel("MSE 损失")
ax.set_title("SGD 损失下降曲线")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

# ── Section 6: 过拟合自检 ──
st.subheader("6. 过拟合自检")


def gd_r2(Xtr, ytr, Xte, yte, eta, n_iter):
    """在给定切分上跑手写梯度下降，返回（训练 R2, 测试 R2）。

    归一化参数只来自该切分自己的训练部分——
    无论哪种切分方式，预处理规则都一致。
    """
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-8
    Xn_tr = (Xtr - mean) / std
    Xn_te = (Xte - mean) / std
    w = np.zeros(Xtr.shape[1])
    b = 0.0
    for _ in range(n_iter):
        err = Xn_tr @ w + b - ytr
        w = w - eta * 2 * (Xn_tr.T @ err) / len(ytr)
        b = b - eta * 2 * err.mean()
    return r2_score(ytr, Xn_tr @ w + b), r2_score(yte, Xn_te @ w + b)


X_all = df[selected_features].values
y_all = df[label_col].values

with st.spinner("三种切分 × 两种模型 训练中…"):
    # ① 全量 fit（训练集 = 测试集 = 全部数据，必然高估）
    m_full = LinearRegression().fit(X_all, y_all)
    full_r2 = r2_score(y_all, m_full.predict(X_all))
    gd_full_train, gd_full_test = gd_r2(X_all, y_all, X_all, y_all, eta, n_iter)

    # ② 随机 split（打乱后切，测试集混着"过去"，评估掺假）
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42)
    m_rand = LinearRegression().fit(X_tr2, y_tr2)
    rand_train_r2 = r2_score(y_tr2, m_rand.predict(X_tr2))
    rand_test_r2 = r2_score(y_te2, m_rand.predict(X_te2))
    gd_rand_train, gd_rand_test = gd_r2(X_tr2, y_tr2, X_te2, y_te2, eta, n_iter)

    # ③ 时间 split（第 3 节的切分，最接近实盘）
    gd_time_train, gd_time_test = gd_r2(
        X_train, y_train, X_test, y_test, eta, n_iter)

check_df = pd.DataFrame(
    {
        "sklearn R2": [full_r2, rand_train_r2, rand_test_r2,
                       train_r2, test_r2],
        "手写SGD R2": [gd_full_test, gd_rand_train, gd_rand_test,
                       gd_time_train, gd_time_test],
    },
    index=["全量fit（训练=测试）", "随机split 训练", "随机split 测试",
           "时间split 训练", "时间split 测试"],
)
st.dataframe(check_df.style.format("{:.4f}"), use_container_width=True)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(check_df))
width = 0.38
b1 = ax.bar(x - width / 2, check_df["sklearn R2"], width,
            label="sklearn（闭式解）", color="#3b82f6", alpha=0.7)
b2 = ax.bar(x + width / 2, check_df["手写SGD R2"], width,
            label="手写 SGD", color="#f59e0b", alpha=0.7)
# 训练行蓝、测试行红：测试柱加红边框提示"这是该看的数字"
for i, bar_group in enumerate([b1, b2]):
    for j, bar in enumerate(bar_group):
        if "测试" in check_df.index[j]:
            bar.set_edgecolor("#ef4444")
            bar.set_linewidth(1.5)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(check_df.index, fontsize=9)
ax.set_ylabel("R2")
ax.set_title("过拟合自检：三种切分 × 两种模型（红框 = 测试集，重点看）")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
span = (max(check_df.max()) - min(check_df.min().min(), 0)) or 1
off = span * 0.02
for bars_ in (b1, b2):
    for bar in bars_:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + off if val >= 0 else val - off,
                f"{val:.4f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8)
fig.tight_layout()
st.pyplot(fig)

st.info(
    "sklearn 和手写 SGD 两列几乎相同——再次印证上一节：两者收敛到同一个最优解，"
    "过拟合的结论与用什么优化器无关。"
    "两种 split 的测试对象是不同时段，单次对比会有波动；"
    "关键区别在于——随机打乱让评估混入未来信息（数据泄漏），"
    "时间切分才符合实盘。W12 数据泄漏一节正式展开。"
)
