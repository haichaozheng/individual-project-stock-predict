"""获取上证指数日线数据与基本面因子并落盘为 CSV。

用 tushare pro_api 获取 000001.SH（上证指数）的数据：
  - index_daily      日线行情（开高低收、成交量、涨跌幅）
  - index_dailybasic 指数基本面因子（换手率、PE、PB）
按 trade_date 合并后保存为 sse_daily.csv，供 linear_regression.py 使用。

运行前：
  - 安装依赖：pip install tushare pandas python-dotenv
  - 配置密钥：在项目根目录 .env 文件中写入 TUSHARE_TOKEN=你的token
    （参考 weeks/week07/cases/.env.example）
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

DATA_DIR = Path(__file__).resolve().parent
load_dotenv(DATA_DIR.parent.parent.parent.parent / ".env")

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

INDEX_CODE = "000001.SH"
# 指数基本面因子（index_dailybasic）对 000001.SH 最早覆盖到 2004-01-02，
# 2004 年之前 PE/PB/换手率无数据，故从 2004 年开始采集，保证特征完整。
START_DATE = "20040101"
# 结束日期动态取当天，数据保持最新
END_DATE = datetime.now().strftime("%Y%m%d")


def fetch_index_daily(pro) -> pd.DataFrame:
    """从 tushare 获取上证指数日线。"""
    print(f"正在获取 {INDEX_CODE} 日线数据（{START_DATE} ~ {END_DATE}）…")
    df = pro.index_daily(
        ts_code=INDEX_CODE,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if df is None or df.empty:
        raise RuntimeError("tushare 返回空数据，请检查 token 是否有效、是否有指数权限。")

    # tushare 默认按日期降序，改为升序
    df = df.sort_values("trade_date").reset_index(drop=True)

    print(f"获取成功：{len(df)} 条日线，"
          f"{df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")
    return df


def fetch_index_factors(pro) -> pd.DataFrame:
    """从 tushare 获取上证指数基本面因子：换手率、PE、PB。

    index_dailybasic 每天一行，与日线按 trade_date 一一对应。
    注意：该接口单次最多返回 3000 条，2004~2026 一次拉不完，
    所以按年循环拉取再拼接。
    """
    frames = []
    start_year = int(START_DATE[:4])
    end_year = int(END_DATE[:4])
    for year in range(start_year, end_year + 1):
        y_start = max(f"{year}0101", START_DATE)
        y_end = min(f"{year}1231", END_DATE)
        print(f"正在获取 {INDEX_CODE} 基本面因子 {y_start} ~ {y_end} …")
        part = pro.index_dailybasic(
            ts_code=INDEX_CODE,
            start_date=y_start,
            end_date=y_end,
            fields="trade_date,turnover_rate,pe,pb",
        )
        if part is not None and not part.empty:
            frames.append(part)

    if not frames:
        raise RuntimeError(
            "index_dailybasic 返回空数据。请确认 token 积分 >= 2000（该接口的权限要求）。"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="trade_date")
    df = df.sort_values("trade_date").reset_index(drop=True)

    print(f"因子获取成功：{len(df)} 条，"
          f"{df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")
    return df


def main() -> None:
    if not TUSHARE_TOKEN:
        raise ValueError(
            "未找到 TUSHARE_TOKEN。请在项目根目录 .env 文件中配置：\n"
            "  TUSHARE_TOKEN=你的token\n"
            "参考 weeks/week07/cases/.env.example"
        )

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    daily = fetch_index_daily(pro)
    factors = fetch_index_factors(pro)

    # 按 trade_date 左连接：个别日期因子缺失时保留日线，因子列用前值填充
    df = daily.merge(factors, on="trade_date", how="left")
    df[["turnover_rate", "pe", "pb"]] = (
        df[["turnover_rate", "pe", "pb"]].ffill()
    )

    missing = df[["turnover_rate", "pe", "pb"]].isna().sum().sum()
    if missing:
        print(f"提示：因子列仍有 {missing} 个缺失值（多为最早几天），"
              "linear_regression.py 中会统一 dropna 处理。")

    out_path = DATA_DIR / "sse_daily.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存至 {out_path}")
    print(f"字段：{list(df.columns)}")
    print("前 5 行：")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
