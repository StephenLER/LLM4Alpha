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
    prompts/
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


# ========= 1. 配置区域 =========

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
N_ROWS_FOR_SAMPLE = 2

# Prompt 输出目录
PROMPT_DIR = Path("prompts")
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
    # 将 Date 列转化为字符串，以避免 json.dumps 中的错误
    df_sample['Date'] = df_sample['Date'].astype(str)

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
任务提示：为 {company_name} 的股票价格生成预测性 Alpha 信号

目标：
使用以下数据生成公式化的 alpha 信号，预测 {company_name} 的股票价格：

1. 股票特征（例如：收盘价、开盘价、最高价、最低价、成交量）。
2. 技术指标（例如：RSI、各种移动平均线、MACD、布林带、动量、OBV）。

输入数据：

你将获得一个 pandas.DataFrame，行表示交易日，列包括：

- 股票特征：{stock_cols_str}
- 技术指标：{tech_cols_str}

该 DataFrame 已对大部分列进行了 z-score 标准化。

下面是该 DataFrame 的 JSON 格式样本（最后 {N_ROWS_FOR_SAMPLE} 行），字段 "Date" 是交易日期：

```json
{df_json}
列说明：
{col_description}

要求：

1. Alpha 公式设计
   - 提出 5 个公式化 alpha：alpha1, alpha2, alpha3, alpha4, alpha5。
   - 每个 alpha 必须是 `df` 中每日特征的确定性函数。
   - 你可以使用滞后值，例如：
       df['Close'].shift(1)
       df['RSI_14'].shift(3)
       df['SMA_5'].shift(5)
   - 不得使用任何情感、新闻或文本相关变量（这些数据不存在）。

2. 公式解释：
   - 对每个 alpha 公式（例如：alpha1）进行详细解释。
   - 解释公式的构成逻辑，包括它捕捉到的市场行为（例如：趋势跟随、均值回归）。
   - 解释每个特征（例如：`df['Close']`、`df['RSI_14']`、`df['SMA_20']`）如何为预测做出贡献。
   - 公式背后的思想（例如：该公式设计用来捕捉动量、均值回归或市场的超买/超卖状态等）。
   - 你必须使用以下格式来组织每个 alpha 和它的解释：

     **返回格式（非常重要）**：
     
     alpha1 = <Python 表达式使用 df[...]>
     
     解释：<解释公式的逻辑，特征的作用，捕捉的市场行为>
     
     alpha2 = <Python 表达式使用 df[...]>
     
     解释：<解释公式的逻辑，特征的作用，捕捉的市场行为>

     以此类推，直到 alpha5。

3. 设计原则：
   - Alpha 公式应该是可解释的，并且能够反映常见的交易理念，例如：
       • 趋势跟随（例如：价格在移动平均线之上/之下）
       • 均值回归（例如：价格偏离某一带之后回归）
       • 动量（例如：短期与长期动量的关系）
       • 成交量确认（例如：OBV、基于成交量的过滤器）
       • 超买/超卖状态（例如：RSI）
   - 鼓励多样性：每个 alpha 应该捕捉市场行为的不同方面，
     不要只是稍微调整系数重复同样的结构。

4. 实现约束
   - 假设你有一个名为 `df` 的 pandas DataFrame，列如上所列。
   - 使用可以在 `df` 上计算的合法 Python 表达式，例如：
       (df['Close'] - df['SMA_20']) / df['SMA_20']
       (df['RSI_14'] - 50) / 50
       (df['MACD'] - df['MACD_Signal'])
       (df['OBV'] - df['OBV'].shift(5))
   - 你可以使用如 max()、min()、abs() 等函数，以及逻辑运算符 (&, |, ~)。
   - 不要在公式中使用伪代码、循环或自然语言描述。

示例（仅供参考，请不要直接使用）：

alpha_example = (df['Close'] - df['SMA_20']) / df['SMA_20'] + 0.3 * (df['RSI_14'] - 50) / 50

输出格式（非常重要）：

只输出 5 个 alpha 定义，每个公式一行，格式如下：

alpha1 = <使用 df[...] 的 Python 表达式>
alpha2 = <使用 df[...] 的 Python 表达式>
alpha3 = <使用 df[...] 的 Python 表达式>
alpha4 = <使用 df[...] 的 Python 表达式>
alpha5 = <使用 df[...] 的 Python 表达式>

然后，为每个 alpha 公式提供解释：

explain for alpha1：
<解释公式的逻辑，特征的作用，捕捉的市场行为>

explain for alpha2：
<...>

...
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