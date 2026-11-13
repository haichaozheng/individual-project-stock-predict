"""上证指数次日收益率预测：线性回归完整流程（五步法）。

流程：
  1. 数据采集与特征构造：读入 sse_daily.csv（fetch_data.py 生成），
     技术指标 + 基本面因子，构造两个标签（次日收益率 / 次日振幅）
  2. 可视化探索：价格走势、标签分布、因子-标签散点、相关性热力图
  3. 数据集划分与预处理：先按时间切 train/test，
     再用训练集拟合 StandardScaler，分别变换训练/测试集
  4. 模型构建：sklearn LinearRegression + 手写 SGD（梯度下降）
  5. 模型评估：训练/测试 R² 与 MSE、预测散点图、过拟合自检

双目标对比（本案例的核心教学点之一）：
  同一批特征、同一个流程——
  · 预测次日收益率：测试 R² ≈ 0（收益几乎不可预测，噪声主导）
  · 预测次日振幅：  测试 R² 通常 0.3~0.6（波动有聚集性，可预测）
  这就是"可预测性"的直观体会：不是所有预测问题难度都一样。

对应课堂 notebook：machine learning for stock prediction.ipynb
  §gradient descent → §线性回归 → §train/test split
课堂用 gdp_satisfaction.csv；本案例把同一流程搬到上证指数上。

⚠ 预处理铁律（本案例的核心教学点）：
  归一化参数（均值、标准差）只能在训练集上拟合。
  如果先把全量数据归一化再切分，测试集的信息（未来信息）会
  泄漏进训练过程——模型评估结果会虚高，实盘会失效。
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "output"

# 中文显示：优先用仓库自带的开源字体（Noto Sans CJK，OFL 协议可自由分发）。
# 本地 Windows 有系统字体也能画，但 Streamlit Cloud 等 Linux 环境没有
# 微软雅黑/黑体，不自带字体的话图里中文会变成方框。
_BUNDLED_FONT = DATA_DIR / "fonts" / "NotoSansCJKsc-Regular.otf"
if _BUNDLED_FONT.exists():
    fm.fontManager.addfont(str(_BUNDLED_FONT))
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SPLIT_RATIO = 0.8  # 前 80% 训练，后 20% 测试（按时间）

FEATURE_COLS = [
    # ── 技术指标（日线自算）──
    "ma_5", "ma_20", "ma_diff", "vol_ratio",
    "volatility_5", "return_1d", "return_5d",
    "range_1d", "range_5",
    # ── 基本面/交易因子（index_dailybasic）──
    "turnover_rate", "pe", "pb",
]

# 双标签：同一批特征，预测难度天差地别
LABEL_COLS = {
    "next_return": "次日收益率",
    "next_range": "次日振幅",
}


# ════════════════════════════════════════════════════════
# 第 1 步 · 数据采集与特征构造
# ════════════════════════════════════════════════════════

def load_and_engineer() -> pd.DataFrame:
    """读入上证指数日线+因子，构造特征与标签。"""
    csv_path = DATA_DIR / "sse_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到 {csv_path}\n"
            "请先运行：python fetch_data.py"
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 因子列缺失说明 CSV 是旧版生成的，提示重新采集
    need_cols = ["turnover_rate", "pe", "pb"]
    if not set(need_cols).issubset(df.columns):
        raise KeyError(
            f"sse_daily.csv 缺少因子列 {need_cols}。\n"
            "这是旧版数据，请重新运行：python fetch_data.py"
        )

    close = df["close"]

    # ── 技术指标（由日线自算）──
    df["ma_5"] = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_diff"] = df["ma_5"] - df["ma_20"]
    df["vol_ratio"] = df["vol"] / df["vol"].rolling(5).mean()

    # 5 日收益率标准差（波动率）
    daily_return = close.pct_change()
    df["volatility_5"] = daily_return.rolling(5).std()

    # 动量特征
    df["return_1d"] = df["pct_chg"] / 100  # pct_chg 是百分比，转小数
    df["return_5d"] = (close / close.shift(5) - 1)

    # 振幅特征（波动聚集性：今天的振幅是明天振幅的最强预测因子）
    df["range_1d"] = df["high"] / df["low"] - 1
    df["range_5"] = df["range_1d"].rolling(5).mean()

    # ── 基本面因子（fetch_data.py 已从 index_dailybasic 采集）──
    #    turnover_rate 换手率：情绪热度/分歧程度
    #    pe / pb：指数估值水平

    # ── 标签 1：次日简单收益率（几乎不可预测——噪声主导）──
    df["next_return"] = close.shift(-1) / close - 1

    # ── 标签 2：次日振幅 = 次日(最高-最低)/最低（波动有聚集性，可预测）──
    df["next_range"] = df["high"].shift(-1) / df["low"].shift(-1) - 1

    # 丢掉无效行：前 20 行（MA20 需要预热）、最后一行（无次日数据）、
    # 以及个别因子缺失日。必须在切分之前完成，保证两个集都没有空值。
    df = df.dropna(subset=FEATURE_COLS + list(LABEL_COLS)).reset_index(drop=True)

    return df


# ════════════════════════════════════════════════════════
# 第 2 步 · 可视化探索
# ════════════════════════════════════════════════════════

def explore_visualize(df: pd.DataFrame) -> None:
    """建模前先"看"数据：走势、分布、散点、相关性。"""
    # 2.1 价格走势（带均线）——直观感受趋势与波动
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["trade_date"], df["close"], linewidth=0.8, label="收盘价")
    ax.plot(df["trade_date"], df["ma_20"], linewidth=0.8, alpha=0.7, label="MA20")
    ax.set_title("上证指数收盘价走势（2015 至今）")
    ax.set_ylabel("点位")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_trend.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / 'price_trend.png'}")

    # 2.2 标签分布——回归任务先看 y 长什么样（两个标签各一张）
    for label_col, label_name in LABEL_COLS.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[label_col], bins=80, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linewidth=1)
        ax.set_xlabel(label_name)
        ax.set_ylabel("天数")
        ax.set_title(f"{label_name}分布（均值 {df[label_col].mean():.5f}，"
                     f"标准差 {df[label_col].std():.5f}）")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{label_col}_dist.png", dpi=150)
        plt.close(fig)
        print(f"  → {OUT_DIR / f'{label_col}_dist.png'}")

    # 2.3 当日 vs 次日收益率散点——看有没有线性关系
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["return_1d"], df["next_return"], alpha=0.15, s=10)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("当日收益率")
    ax.set_ylabel("次日收益率")
    ax.set_title("当日收益率 vs 次日收益率")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_return1d_vs_next.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / 'scatter_return1d_vs_next.png'}")

    corr = df["return_1d"].corr(df["next_return"])
    print(f"  当日收益率与次日收益率相关系数：{corr:.4f}")
    print("  （接近 0 说明单变量线性关系很弱——这正是需要多特征的原因）")

    # 2.4 特征相关性热力图——检查特征之间、特征与两个标签的相关性
    corr_mat = df[FEATURE_COLS + list(LABEL_COLS)].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_mat)))
    ax.set_yticks(range(len(corr_mat)))
    ax.set_xticklabels(corr_mat.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_mat.columns, fontsize=9)
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat)):
            ax.text(j, i, f"{corr_mat.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr_mat.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, shrink=0.8)
    ax.set_title("特征相关性热力图（含标签）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / 'correlation_heatmap.png'}")
    print("  （ma_5 / ma_20 / pe 相关性很高——多重共线性，W13 特征选择时处理）")


# ════════════════════════════════════════════════════════
# 第 3 步 · 数据集划分与预处理
# ════════════════════════════════════════════════════════

def split_and_preprocess(df: pd.DataFrame):
    """先按时间切分，再用训练集拟合归一化参数。

    ⚠ 顺序不能反：
      ✅ 正确：先切分 → scaler.fit(训练集) → transform 训练/测试集
      ❌ 错误：先对全量数据归一化 → 再切分
         （测试集的均值/标准差混进了归一化参数 = 数据泄漏）
    金融数据也不能随机打乱（train_test_split(shuffle 默认 True）
    ——随机打乱会用未来数据训练模型，同样是数据泄漏。

    特征与标签共用同一切分：两个标签（收益率/振幅）用同一批
    X_train / X_test，分别配各自的 y。
    """
    split_idx = int(len(df) * SPLIT_RATIO)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train_raw = train[FEATURE_COLS].values
    X_test_raw = test[FEATURE_COLS].values

    print(f"  训练集：{len(train)} 条（{train['trade_date'].iloc[0].date()} ~ "
          f"{train['trade_date'].iloc[-1].date()}）")
    print(f"  测试集：{len(test)} 条（{test['trade_date'].iloc[0].date()} ~ "
          f"{test['trade_date'].iloc[-1].date()}）")

    # ── 归一化：scaler 只看训练集 ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)   # fit + transform：只在训练集
    X_test = scaler.transform(X_test_raw)         # 只 transform：沿用训练集参数

    print(f"\n  归一化参数（全部来自训练集）：")
    print(f"  {'特征':<16s} {'均值':>12s} {'标准差':>12s}")
    for name, mu, sigma in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
        print(f"  {name:<16s} {mu:>12.4f} {sigma:>12.4f}")

    return train, test, X_train, X_test, scaler


# ════════════════════════════════════════════════════════
# 第 4 步 · 模型构建
# ════════════════════════════════════════════════════════

def sklearn_fit(X_train, y_train, label_name) -> LinearRegression:
    """sklearn LinearRegression：闭式解，一行 fit。"""
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"\n  sklearn LinearRegression（目标：{label_name}）")
    print("  （特征已标准化，系数可直接比较大小）")
    print(f"  {'特征':<16s} {'系数':>12s}")
    for name, coef in zip(FEATURE_COLS, model.coef_):
        print(f"  {name:<16s} {coef:>12.6f}")
    print(f"  {'截距':<16s} {model.intercept_:>12.6f}")

    return model


def sgd_fit(X_train, y_train, X_test, y_test,
            label_col, label_name, eta=0.001, n_iter=100_000):
    """手写梯度下降拟合线性回归。

    对应课堂 notebook 中的 gd() 函数（code.docx 有答案）。
    这里用矩阵形式实现，但逻辑完全一致：
      预测 = X @ w + b
      损失 = (预测 - y)²
      梯度 = 2 * (预测 - y) * X   （对 w）
      更新 = w - eta * 梯度
    特征已由第 3 步标准化——梯度下降依赖这一步，
    否则不同量级的特征收敛速度差异巨大。
    """
    n_samples, n_features = X_train.shape

    # 初始化参数
    w = np.zeros(n_features)
    b = 0.0

    # 记录损失变化
    losses = []

    for i in range(n_iter):
        pred = X_train @ w + b
        error = pred - y_train

        # 梯度
        grad_w = 2 * (X_train.T @ error) / n_samples
        grad_b = 2 * error.mean()

        # 更新
        w = w - eta * grad_w
        b = b - eta * grad_b

        if i % 10000 == 0:
            loss = (error ** 2).mean()
            losses.append((i, loss))
            print(f"    iter {i:>6d}  loss = {loss:.8f}")

    # 与 sklearn 对比：参数应基本一致（说明 fit 不是魔法）
    y_test_pred = X_test @ w + b
    sgd_r2 = r2_score(y_test, y_test_pred)
    sgd_mse = mean_squared_error(y_test, y_test_pred)
    sgd_rmse = np.sqrt(sgd_mse)
    sgd_mae = np.abs(y_test - y_test_pred).mean()

    print(f"\n  手写 SGD 结果（目标：{label_name}）")
    print(f"  测试集 R2 = {sgd_r2:.4f}")
    print(f"  测试集 MSE = {sgd_mse:.8f}  RMSE = {sgd_rmse:.6f}  "
          f"MAE = {sgd_mae:.6f}")
    print("  （四个指标都应与 sklearn 几乎一致——凸损失只有唯一最低点）")

    fig, ax = plt.subplots(figsize=(7, 4))
    iters = [l[0] for l in losses]
    vals = [l[1] for l in losses]
    ax.plot(iters, vals, marker="o")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("MSE 损失")
    ax.set_title(f"手写 SGD 损失下降曲线（{label_name}）")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"sgd_vs_sklearn_{label_col}.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / f'sgd_vs_sklearn_{label_col}.png'}")

    print("\n  对照课堂 code.docx 答案：")
    print("  课堂 gd(eta, w, b, x, y) 处理单变量；")
    print("  本案例扩展到多变量，梯度逻辑相同：grad_w = 2 * X.T @ (X@w + b - y) / n")

    return w, b


# ════════════════════════════════════════════════════════
# 第 5 步 · 模型评估
# ════════════════════════════════════════════════════════

def evaluate(model, X_train, y_train, X_test, y_test,
             label_col, label_name) -> float:
    """训练/测试指标 + 预测 vs 真实散点图。返回测试 R²。

    指标口径：
      R²   = 1 - 模型误差 / 猜均值基准误差（<0 即比猜均值还差）
      MSE  = 均方误差（平方量纲，是线性回归实际最小化的目标）
      RMSE = √MSE（回到原量纲，"平均偏差多少"）
      MAE  = 平均绝对误差（对极端值稳健）
    """
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.abs(y_test - y_test_pred).mean()

    print(f"\n  目标：{label_name}")
    print(f"  训练集  R2 = {train_r2:.4f}  MSE = {train_mse:.8f}")
    print(f"  测试集  R2 = {test_r2:.4f}  MSE = {test_mse:.8f}")
    print(f"  测试集  RMSE = {test_rmse:.6f}  MAE = {test_mae:.6f}"
          f"  （平均偏差，原量纲）")
    if label_col == "next_return":
        print("  （测试 R2 接近 0 甚至为负——收益被噪声主导，"
              "这就是'预测股价很难'的定量表述）")

    # 预测值 vs 真实值散点
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_test_pred, alpha=0.2, s=10)
    lim = max(abs(y_test).max(), abs(y_test_pred).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y = x（完美预测）")
    ax.set_xlabel(f"真实{label_name}")
    ax.set_ylabel(f"预测{label_name}")
    ax.set_title(f"sklearn 预测 vs 真实（{label_name}，测试集 R²={test_r2:.4f}）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"regression_fit_{label_col}.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / f'regression_fit_{label_col}.png'}")

    return test_r2


def overfitting_check(df: pd.DataFrame, label_col: str, label_name: str) -> None:
    """对比：全量 fit vs 随机 split vs 时间 split。

    随机 split 的 scaler 同样只在它自己的训练部分拟合——
    预处理规则在任何切分方式下都一样。
    """
    X = df[FEATURE_COLS].values
    y = df[label_col].values

    # 全量 fit（仅供对照——真实场景不允许这样做）
    model_full = LinearRegression()
    model_full.fit(X, y)
    full_r2 = r2_score(y, model_full.predict(X))

    # 随机 split（课堂 gdp_satisfaction 的做法）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler_rand = StandardScaler().fit(X_tr)  # 归一化参数只来自随机训练集
    model_split = LinearRegression()
    model_split.fit(scaler_rand.transform(X_tr), y_tr)
    train_r2 = r2_score(y_tr, model_split.predict(scaler_rand.transform(X_tr)))
    test_r2 = r2_score(y_te, model_split.predict(scaler_rand.transform(X_te)))

    # 时间 split（本案例的做法）
    split_idx = int(len(df) * SPLIT_RATIO)
    scaler_time = StandardScaler().fit(X[:split_idx])
    model_time = LinearRegression()
    model_time.fit(scaler_time.transform(X[:split_idx]), y[:split_idx])
    time_train_r2 = r2_score(
        y[:split_idx], model_time.predict(scaler_time.transform(X[:split_idx])))
    time_test_r2 = r2_score(
        y[split_idx:], model_time.predict(scaler_time.transform(X[split_idx:])))

    print(f"\n  过拟合自检（目标：{label_name}）")
    print(f"  全量 fit R2           = {full_r2:.4f}")
    print(f"  随机 split 训练 R2     = {train_r2:.4f}")
    print(f"  随机 split 测试 R2     = {test_r2:.4f}")
    print(f"  时间 split 训练 R2     = {time_train_r2:.4f}")
    print(f"  时间 split 测试 R2     = {time_test_r2:.4f}")
    print(f"  （差距大 = 过拟合；差距小 = 模型可能本身就没什么用）")

    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ["全量\nfit", "随机 split\n训练", "随机 split\n测试",
                  "时间 split\n训练", "时间 split\n测试"]
    values = [full_r2, train_r2, test_r2, time_train_r2, time_test_r2]
    colors = ["#3b82f6", "#3b82f6", "#ef4444", "#3b82f6", "#ef4444"]
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("R²")
    ax.set_title(f"过拟合自检：不同 split 方式的 R² 对比（{label_name}）")
    ax.grid(True, alpha=0.3, axis="y")
    # R² 值很小时，用相对偏移放标签，避免和标题/横轴重叠
    span = max(values) - min(values)
    off = max(span * 0.08, 1e-4)
    ax.set_ylim(min(values) - off * 2, max(values) + off * 4)
    for bar, val in zip(bars, values):
        if val >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + off,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val - off,
                    f"{val:.4f}", ha="center", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"overfitting_check_{label_col}.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT_DIR / f'overfitting_check_{label_col}.png'}")


# ════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("案例：用上证指数预测次日行情——线性回归五步法（双目标对比）")
    print("=" * 60)

    # 第 1 步 · 数据采集与特征构造
    print("\n[1/5] 数据采集与特征构造")
    df = load_and_engineer()
    print(f"  样本数：{len(df)}")
    print(f"  日期范围：{df['trade_date'].iloc[0].date()} ~ "
          f"{df['trade_date'].iloc[-1].date()}")
    print(f"  特征列（{len(FEATURE_COLS)} 个）：{FEATURE_COLS}")
    print(f"  标签列（{len(LABEL_COLS)} 个）：{list(LABEL_COLS)}")

    # 第 2 步 · 可视化探索
    print("\n[2/5] 可视化探索")
    explore_visualize(df)

    # 第 3 步 · 数据集划分与预处理（先切分，再归一化）
    print(f"\n[3/5] 数据集划分与预处理（按时间前 {SPLIT_RATIO:.0%} 训练 / "
          f"后 {1 - SPLIT_RATIO:.0%} 测试 → 训练集拟合 scaler）")
    train, test, X_train, X_test, scaler = split_and_preprocess(df)

    # 第 4、5 步 · 对两个标签分别建模与评估
    results = {}
    for label_col, label_name in LABEL_COLS.items():
        print(f"\n{'─' * 56}")
        print(f"目标：{label_name}（{label_col}）")
        print(f"{'─' * 56}")

        y_train = train[label_col].values
        y_test = test[label_col].values

        print("\n[4/5] 模型构建：sklearn + 手写 SGD")
        model = sklearn_fit(X_train, y_train, label_name)
        sgd_fit(X_train, y_train, X_test, y_test, label_col, label_name)

        print("\n[5/5] 模型评估")
        results[label_col] = evaluate(
            model, X_train, y_train, X_test, y_test, label_col, label_name)
        overfitting_check(df, label_col, label_name)

    # 双目标对比收束
    print("\n" + "=" * 60)
    print("双目标对比：同一批特征，可预测性天差地别")
    print("=" * 60)
    print(f"  次日收益率（next_return）测试 R2 = {results['next_return']:.4f}")
    print(f"  次日振幅  （next_range）  测试 R2 = {results['next_range']:.4f}")
    print("\n  收益被噪声主导，几乎不可预测；")
    print("  波动有聚集性（大波动后面跟大波动），可以用同样的特征预测。")
    print("  → 建模前先想清楚：你要预测的量，本身有没有可预测的成分？")
    print("\n  两种 split 的测试对象是不同时段，单次对比会有波动；")
    print("  关键区别在于——随机打乱让评估混入未来信息（数据泄漏），")
    print("  时间切分才符合实盘。W12 数据泄漏一节正式展开。")

    print("\n下一步：")
    print("  - 换成你 W4 落盘的项目数据（参考 README.md 改造指引）")
    print("  - W12 用 KNN 把'涨跌方向'做成分类问题")
    print("  - W13 用决策树/随机森林升级模型，看特征重要性")
    print("  - W14 用振幅预测做仓位/止损，用收益预测做方向信号，回测验证")


if __name__ == "__main__":
    main()
