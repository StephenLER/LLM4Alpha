"""
feature_engineering.py

对 5 家公司的 OHLCV 做特征工程，生成论文中使用的技术指标特征。

输入：
    - 每家公司一个 CSV，至少包含：
        Date, Open, High, Low, Close, Volume
    - Date 可以是单独列（后面会设为 index）

输出：
    - 每家公司一个 *_with_tech.csv：原始 + 技术指标
    - 可选：一个 *_with_tech_standardized.csv：做过 Z-score 标准化的版本
"""

import os
from pathlib import Path

import pandas as pd
import pandas_ta as ta


# ======== 1. 配置区域  ========

# 每个文件至少包含: Date, Open, High, Low, Close, Volume
DATA_DIR = Path("./data")

# 这里用一个 dict 指定“公司名 -> CSV 文件名”
COMPANY_FILES = {
    "apple":   "data_price_AAPL_US.csv",
    "hsbc":    "data_price_5_HK.csv",
    "pepsi":   "data_price_PEP_US.csv",
    "toyota":  "data_price_7203_JP.csv",
    "tencent": "data_price_700_HK.csv",
}

# 输出目录
OUT_DIR = Path("./data_with_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ======== 2. 技术指标函数 ========

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原 OHLCV DataFrame 上添加论文中用到的技术指标。

    要求:
        - df 至少包含列: ['Open', 'High', 'Low', 'Close', 'Volume']
        - index 为 DatetimeIndex

    返回:
        - 添加若干列后的 DataFrame（不做 dropna）
    """
    df = df.copy()
    df = df.sort_index()

    # 确保列存在
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"DataFrame 缺少必须列: {c}")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1) 简单移动均线 SMA
    df["SMA_5"] = ta.sma(close=close, length=5)
    df["SMA_20"] = ta.sma(close=close, length=20)

    # 2) 指数移动均线 EMA
    df["EMA_10"] = ta.ema(close=close, length=10)

    # 3) 动量 Momentum
    df["MOM_3"] = ta.mom(close=close, length=3)
    df["MOM_10"] = ta.mom(close=close, length=10)

    # 4) RSI 相对强弱指数
    df["RSI_14"] = ta.rsi(close=close, length=14)

    # 5) MACD（快线、慢线、信号线）
    macd_df = ta.macd(close=close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        # 通常列名类似: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df["MACD"] = macd_df.iloc[:, 0]         # MACD 主线
        df["MACD_Signal"] = macd_df.iloc[:, 2]  # 信号线

    # 6) 布林带 Bollinger Bands
    bb_df = ta.bbands(close=close, length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        # 通常列顺序: [BBL, BBM, BBU, BBB, BBP]
        df["BB_Lower"] = bb_df.iloc[:, 0]
        df["BB_Upper"] = bb_df.iloc[:, 2]

    # 7) OBV 能量潮
    df["OBV"] = ta.obv(close=close, volume=volume)

    return df


# ======== 3. 可选：Z-score 标准化（给 LLM 用的时候方便） ========

def zscore_standardize(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """
    对 df 中的数值列做 Z-score 标准化：
        x' = (x - mean) / std

    exclude_cols: 不参与标准化的列名列表（比如 'Close' 或者未来你要保留原始值的列）

    返回:
        - 新的 DataFrame（不在 exclude_cols 的数值列全部标准化）
    """
    if exclude_cols is None:
        exclude_cols = []

    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]

    for c in cols_to_scale:
        mean = df[c].mean()
        std = df[c].std()
        if std == 0 or pd.isna(std):
            # 全部一样/方差为 0 的列就不动它
            continue
        df[c] = (df[c] - mean) / std

    return df


# ======== 4. 主流程 ========

def main():
    print("=== 开始对 5 家公司做特征工程（技术指标） ===")

    for company_name, filename in COMPANY_FILES.items():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"[WARN] 公司 {company_name} 的文件不存在: {file_path}，跳过。")
            continue

        print(f"\n--- 处理 {company_name} ---")
        print(f"读取: {file_path}")

        # 读取 CSV
        df = pd.read_csv(file_path)

        # 处理日期列 -> DatetimeIndex
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            # 如果原本 index 就是日期，你可以视情况省略这一步
            # 这里做一个简单检查：
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{file_path} 既没有 Date 列，也没有 DatetimeIndex，"
                                 f"请确认你的 CSV 格式。")

        # 添加技术指标
        df_feat = add_technical_indicators(df)

        # 一般前面几十行会因为技术指标窗口变长出现 NaN，这里统一 drop 掉
        df_feat = df_feat.dropna()

        # 保存“原始 + 技术指标”版本
        out_file = OUT_DIR / f"{company_name}_with_tech.csv"
        df_feat.to_csv(out_file)
        print(f"已保存带技术指标的特征到: {out_file}，形状: {df_feat.shape}")

        # 可选：再生成一个标准化版本（给 LLM 输入更方便）
        # 比如保留原始 Close，不参与标准化：
        df_std = zscore_standardize(df_feat, exclude_cols=["Close"])
        out_file_std = OUT_DIR / f"{company_name}_with_tech_standardized.csv"
        df_std.to_csv(out_file_std)
        print(f"已保存标准化后的特征到: {out_file_std}，形状: {df_std.shape}")

    print("\n=== 特征工程完成 ===")


if __name__ == "__main__":
    main()
