"""
prepare_llm_prompts_from_paper_style.py

功能：
    参考论文《Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha》
    中 Table 3 的结构化提示词设计，但因为当前数据集中没有新闻文本 / 情感相关列，
    所以只使用「价格 + 技术指标」特征来生成 LLM 的 Prompt。

输入：
    data_with_features/
        apple_with_tech_standardized.csv
        hsbc_with_tech_standardized.csv
        pepsi_with_tech_standardized.csv
        toyota_with_tech_standardized.csv
        tencent_with_tech_standardized.csv

输出：
    prompts_from_paper/
        prompt_apple.txt
        prompt_hsbc.txt
        prompt_pepsi.txt
        prompt_toyota.txt
        prompt_tencent.txt

"""

from pathlib import Path
import pandas as pd
import json
from textwrap import dedent


# ========= 1. 配置区域：根据你本地情况改文件名 =========

FEATURE_DIR = Path("data_with_features")

# 逻辑公司名 -> 特征工程后的 CSV 文件名（标准化版本）
COMPANY_FILES = {
    "Apple":   "apple_with_tech_standardized.csv",
    "HSBC":    "hsbc_with_tech_standardized.csv",
    "Pepsi":   "pepsi_with_tech_standardized.csv",
    "Toyota":  "toyota_with_tech_standardized.csv",
    "Tencent": "tencent_with_tech_standardized.csv",
}

# 每家公司给 LLM 看多少行样本（从末尾截取）
N_ROWS_FOR_SAMPLE = 40

# Prompt 输出目录
PROMPT_DIR = Path("prompts_from_paper")
PROMPT_DIR.mkdir(parents=True, exist_ok=True)


# ========= 2. 辅助函数：按列名简单分类 =========

def categorize_columns(cols):
    """
    根据常见命名把列粗略分为三类：
        - stock_features: OHLCV
        - tech_indicators: 技术指标
        - others: 其它数值特征（如果有）
    """
    stock_features = []
    tech_indicators = []
    others = []

    for c in cols:
        lc = c.lower()
        if lc in ["open", "high", "low", "close", "volume"]:
            stock_features.append(c)
        elif any(
            kw in lc
            for kw in [
                "sma", "ema", "mom", "momentum", "rsi",
                "macd", "bb_", "bollinger", "obv"
            ]
        ):
            tech_indicators.append(c)
        else:
            others.append(c)

    return stock_features, tech_indicators, others


def describe_columns(stock_cols, tech_cols, other_cols):
    """
    为 prompt 生成一段列说明文本。
    """
    lines = []

    if stock_cols:
        lines.append("Stock Features:")
        for c in stock_cols:
            lines.append(f"- {c}: basic OHLCV-related feature (price or volume).")

    if tech_cols:
        lines.append("")
        lines.append("Technical Indicators:")
        for c in tech_cols:
            lc = c.lower()
            if "sma" in lc:
                desc = "simple moving average of Close."
            elif "ema" in lc:
                desc = "exponential moving average of Close."
            elif "mom" in lc or "momentum" in lc:
                desc = "price momentum over a short rolling window."
            elif "rsi" in lc:
                desc = "Relative Strength Index (overbought/oversold oscillator)."
            elif "macd_signal" in lc:
                desc = "signal line of MACD indicator."
            elif "macd" in lc:
                desc = "MACD line (trend/momentum indicator)."
            elif "bb_upper" in lc:
                desc = "upper band of Bollinger Bands (volatility)."
            elif "bb_lower" in lc:
                desc = "lower band of Bollinger Bands (volatility)."
            elif "obv" in lc:
                desc = "On-Balance Volume (volume-based trend)."
            else:
                desc = "technical indicator derived from price or volume."
            lines.append(f"- {c}: {desc}")

    if other_cols:
        lines.append("")
        lines.append("Other Numerical Features:")
        for c in other_cols:
            lines.append(f"- {c}: additional numerical feature.")

    return "\n".join(lines)


# ========= 3. 构造单个公司的 Prompt =========

def build_prompt_for_company(company_name: str, df: pd.DataFrame) -> str:
    """
    构造一个模仿论文 Table 3 结构的 Prompt，但只使用价格 + 技术指标特征。
    """

    # 选取最近 N 行样本，保留 Date 列以便 LLM 感知时间
    df_sample = df.tail(N_ROWS_FOR_SAMPLE).reset_index()  # index 变成 'Date' 列
    # 转成 JSON 字符串，方便直接嵌入 prompt
    df_json = json.dumps(
        df_sample.to_dict(orient="records"),
        ensure_ascii=False,
        indent=2
    )

    # 列分类
    data_cols = list(df.columns)
    stock_cols, tech_cols, other_cols = categorize_columns(data_cols)
    col_description = describe_columns(stock_cols, tech_cols, other_cols)

    stock_cols_str = ", ".join(stock_cols) if stock_cols else "basic OHLCV features"
    tech_cols_str = ", ".join(tech_cols) if tech_cols else "multiple technical indicators"
    other_cols_str = ", ".join(other_cols) if other_cols else "none"

    prompt = f"""
Task Prompt: Generating Predictive Alphas for {company_name}'s Stock Prices

Objective:
Generate formulaic alpha signals to predict {company_name}'s stock prices using:

1. Stock features (e.g., Close, Open, High, Low, Volume).
2. Technical indicators (e.g., RSI, moving averages, MACD, Bollinger Bands, momentum, OBV).

There is NO news text or sentiment data in the current dataset. All alphas must be based only on price
and technical indicator features.

Input Data:

You are given a single pandas.DataFrame with rows representing trading days and columns including:

- Stock Features: {stock_cols_str}
- Technical Indicators: {tech_cols_str}
- Other Numerical Features (if any): {other_cols_str}

The DataFrame is already z-score normalized (standardized) for most columns.

Below is a JSON-formatted sample (last {N_ROWS_FOR_SAMPLE} rows) of the DataFrame
for {company_name}. The field "Date" is the trading date:

```json
{df_json}
Column description:
{col_description}

Requirements:

Alpha Formulation:

Propose 5 formulaic alphas: alpha1, alpha2, alpha3, alpha4, alpha5.

Each alpha should be a deterministic function of the available daily features in the DataFrame.

You may use lagged values through expressions such as:

df['Close'].shift(1)

df['RSI_14'].shift(3)

df['SMA_5'].shift(5)

You must NOT use any sentiment, news, or text-based variables (they are not available).

Design Principles:

The alphas should be interpretable and reflect common trading ideas such as:

trend-following (e.g., price above/below moving averages),

mean-reversion (e.g., price deviating from a band and reverting),

momentum (e.g., short-term vs long-term momentum),

volume confirmation (e.g., OBV, Volume-based filters),

overbought/oversold conditions (e.g., RSI).

Encourage diversity: each alpha should capture a different aspect of market behavior
(do not just repeat the same structure with slightly different coefficients).

Implementation Constraints:

Assume a pandas DataFrame named df with the same columns as in the JSON sample.

Use valid Python expressions that can be evaluated on df, for example:
(df['Close'] - df['SMA_20']) / df['SMA_20']
(df['RSI_14'] - 50) / 50
(df['MACD'] - df['MACD_Signal'])
(df['OBV'] - df['OBV'].shift(5))

You may use functions like max(), min(), abs(), and logical operators (&, |, ~).

Avoid any pseudo-code, loops, or natural language inside the formulas.

Example (purely for illustration, DO NOT reuse as final answer):

alpha_example = (df['Close'] - df['SMA_20']) / df['SMA_20'] + 0.3 * (df['RSI_14'] - 50) / 50

Output Format (very important):

Return ONLY the 5 alpha definitions in exactly the following style, one per line:

alpha1 = <Python expression using df[...]>
alpha2 = <Python expression using df[...]>
alpha3 = <Python expression using df[...]>
alpha4 = <Python expression using df[...]>
alpha5 = <Python expression using df[...]>
"""
    return dedent(prompt).strip()


# ========= 4. 主流程 =========
def main():
    for company_name, filename in COMPANY_FILES.items():
        csv_path = FEATURE_DIR / filename
        if not csv_path.exists():
            print(f"[WARN] File not found for {company_name}: {csv_path}")
            continue

        print(f"=== Building LLM prompt for {company_name} from {csv_path} ===")

        # 读取 CSV
        df = pd.read_csv(csv_path)

        # 如果有 Date 列，设为索引，方便 tail 和分类
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

        prompt_text = build_prompt_for_company(company_name, df)

        out_path = PROMPT_DIR / f"prompt_{company_name.lower()}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        print(f"Saved prompt to: {out_path}")

    print("\nAll prompts generated in:", PROMPT_DIR.resolve())


if __name__ == "__main__":
    main()
