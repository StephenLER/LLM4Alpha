整体上，这篇文章的工作流可以看成一个**自上而下的流水线**：
先做数据与特征，再让 LLM 写公式化 alpha，然后用这些 alpha 去喂 Transformer（加上一些对比模型）做**次日收盘价预测**。

## 一、总览：各阶段流程与 I/O

| 阶段 | 主要操作            | 输入                        | 输出                                   |
| -- | --------------- | ------------------------- | ------------------------------------ |
| 1  | 原始数据采集          | 目标公司列表、时间区间、数据 API        | 日度 OHLCV 行情、新闻文本                     |
| 2  | 特征工程（技术指标 + 情感） | 日度 OHLCV、新闻文本             | 带技术指标 & 多公司情感分数的 DataFrame           |
| 3  | LLM 生成公式化 Alpha | 结构化 DataFrame 示例 + Prompt | 每公司 5 条公式化 alpha（文本）+ 推理说明           |
| 4  | 计算 Alpha 序列     | DataFrame + alpha 公式文本    | 日度 alpha_t 数值序列（添加到 DataFrame）       |
| 5  | 构造时序样本（滑动窗口）    | 完整特征表（含 alpha）            | Transformer/LSTM/TCN/SVR/RF 的训练和测试张量 |
| 6  | 训练 Transformer  | 训练张量 + 超参数                | 训练好的 Transformer 模型、测试集预测            |
| 7  | 训练对比模型          | 同上（输入特征基本一致）              | 各模型预测结果                              |
| 8  | 评估与扩展指标         | 真实价格 + 预测结果 + alpha 序列    | MSE、带/不带情感对比、不同 alpha 来源对比、IC        |


## 阶段 1：原始数据采集

### 输入

1. **目标公司列表**（5 家）：

   * Apple, HSBC, Pepsi, Toyota, Tencent 
2. **时间区间**

   * 2016-05-01 ～ 2024-05-08
3. **数据源**

   * 行情：`yfinance`（Open/High/Low/Close/Volume）
   * 新闻：EODHD 的 “Financial News Feed and Stock News Sentiment Data API”

### 输出

1. **日度行情表**（每家公司一份 DataFrame）

   * 列：`[Date, Open, High, Low, Close, Volume]`
2. **新闻原始数据**

   * 每条新闻至少包含：`[date, company, title/body text, …]`
   * 公司既包括目标公司，也包括**相关新闻中的关联公司**（表 2 给出了最终的关联公司集合）

> 复现时：
>
> * 行情用 yfinance 很好做。
> * 新闻用不到同一个 API 也行，只要能拿到“公司 + 日期 + 文本”即可。

---

## 阶段 2：特征工程（技术指标 + 情感）

### 2.1 技术指标计算

#### 输入

* 阶段 1 输出的 OHLCV 日线数据

#### 操作

* 使用 `pandas-ta` 计算多种指标（表 1）：

  * SMA_5、SMA_20
  * EMA_10
  * Momentum_3、Momentum_10
  * RSI_14
  * MACD、MACD_Signal
  * BB_Upper、BB_Lower
  * OBV 

#### 输出

* 扩展后的行情 DataFrame：

  * 列包含：`[Open, High, Low, Close, Volume, SMA_5, SMA_20, EMA_10, …, OBV]`

### 2.2 关联公司集合（已给出）

#### 输入

* 来自前一篇工作的 NER 和共现分析结果（论文里直接给了表 2）

#### 输出

* 对于每个目标公司，有一个“关联公司列表”（长度大约 10）

> 复现时你可以直接**硬编码表 2** 里的关联公司，而不用再做一次 NER。

### 2.3 新闻情感打分

#### 输入

* 阶段 1 的新闻文本（目标公司 + 关联公司）
* 关联公司列表

#### 操作

1. 用 **VADER** 对每条新闻做情感分析，得到 `polarity` 分数。
2. 对每个公司、每一天，把所有新闻的 polarity **平均**：

   * 得到：`Sentiment(company, date)` 的时间序列。

#### 输出

* 一个“公司–日期–情感”的表/多列：

  * 对于 Apple：`Apple_polarity_t`
  * 对于 Google：`Google_polarity_t`
  * ……（表 2 中列出的所有公司）

### 2.4 对齐、清洗与滑动窗口准备

#### 输入

* 行情 + 技术指标 DataFrame
* 多公司日度情感表

#### 操作

1. 按日期对齐，inner join，去掉缺失过多的行。
2. 对所有数值特征可以做标准化（论文中提示使用 Z-score 等标准化方法作为 LLM 输入要求的一部分）。
3. 数据集划分：前 70% 为训练，后 30% 为测试。
4. 设置滑动窗口 **长度 5 天**：稍后要用于构造模型输入。

#### 输出

* 每个目标公司的一个“干净的、对齐后的” DataFrame：

  * 列包括：OHLCV、所有技术指标、目标公司及关联公司的情感分数。
  * 有清晰的日期 index，适合之后喂给 LLM/模型。

---

## 阶段 3：LLM 生成公式化 alpha

### 输入

1. **结构化 Prompt**（论文表 3）：

   * 描述任务目标：生成 5 个 alpha，用于预测某公司的股价。
   * 明确输入数据格式：一个 pandas DataFrame（JSON 格式示例）。
   * 要求 alpha 结合：

     * 股价特征（Open, High, Low, Close, Volume）
     * 技术指标（RSI, SMA, EMA, MACD 等）
     * 目标公司与关联公司的日度情感分数 

2. **数据示例**：

   * 不是把全量数据塞进去，而是取几行（或仅列名+描述），转成 JSON 片段作为 LLM 的上下文。

3. **具体公司名**（通过 `{Company}` 占位替换为 Apple/HSBC/...）

4. **LLM 模型**：

   * deepseek-r1-distill-llama-70b（在 Groq 上运行）

### 输出

1. **每家公司 5 条 Alpha 公式（文本形式）**

   * 论文表 4 给出了最终采纳的 5 条 alpha，比如：

     * Apple 的 `alpha1_t = (C_t - O_t) / O_t + 0.5 * (Apple_polarity_t + Google_polarity_t)`
     * Toyota 的 `alpha5_t = (OBV_t > OBV_t-1) * (C_t > C_t-1) * (1 if BYD_polarity_t > 0 and TSLA_polarity_t > 0 else 0)` 等等。
2. **LLM 的自然语言推理过程**（比如图 3 展示 Toyota 的推理）

   * 主要用于解释“为什么这样写公式”。

> 复现选择：
>
> * **严格复现思路**：你自己也调用同类 LLM，用论文的 prompt 和你的数据结构去生成 alpha。
> * **严格复现结果**：直接使用表 4 中给出的公式，跳过 LLM 这一步，把它当作“已生成的公式”。

---

## 阶段 4：把 alpha 文本公式变成数值特征

### 输入

* 阶段 2 输出的 DataFrame（含所有基础特征和情感）
* 阶段 3 生成的 5 条 alpha 公式（每公司一套）

### 操作

对每一天 t，按公式计算 `alpha1_t ~ alpha5_t`：

* 直接是算术表达式的公式：

  * 用向量运算即可，如：

    * `alpha1_t = (C_t - O_t) / O_t + 0.5*(Apple_polarity_t + Google_polarity_t)`
* 带条件的公式（Toyota 有不少是 0/1 特征）：

  * 在代码中用布尔判断转成 0/1。

### 输出

* 每家公司一个新的 DataFrame：

  * 在原有列的基础上新增：

    * `alpha1`, `alpha2`, `alpha3`, `alpha4`, `alpha5`
  * 这些就是之后模型主要使用的“高层约束特征”。

---

## 阶段 5：构造时序样本（模型输入张量）

### 输入

* 阶段 4 的 DataFrame（对每家公司）

### 操作：按滑动窗口（长度 5 天）构造模型输入/输出

1. **Transformer Encoder 输入**：

   * 对每个样本 i，取过去 **5 天的 5 个 alpha**：

     * 形状（伪）≈ `(window=5, num_alpha=5)`
2. **Transformer Decoder 输入**：

   * 同样是过去 5 天的 **收盘价 Close**：

     * 形状≈ `(window=5, 1)`
3. **时间特征**：

   * 从日期提取 `day_of_week, day_of_month, month` 等，作为 Temporal Embedding 输入。
4. **标签 y**：

   * 第 i 个样本标签：第 6 天（窗口之后）的 Close，即 **次日收盘价**。

### 输出

* 训练/验证/测试集张量：

  * `X_enc`（alpha 序列 + 时间/位置嵌入）
  * `X_dec`（历史 Close 序列 + 时间/位置嵌入）
  * `y`（下一天 Close）

同样的窗口构造也会被 LSTM/TCN/SVR/RandomForest 使用，区别只是输入的维度/形状稍微不一样。

---

## 阶段 6：训练 Transformer

### 输入

* 阶段 5 的训练集张量
* 模型结构和超参数设定：

  * Encoder / Decoder 结构（图 2）
  * 1D Conv Embedding + Temporal Embedding + Position Embedding
  * 损失函数：MSE
  * 优化器：Adam，学习率 5e-5
  * Epoch：50
  * Batch size：8～64（不同股票略有差异）

### 输出

1. 训练好的 **Transformer 模型**（对当前公司）
2. 对测试集的预测：

   * `y_pred`：测试集中每一天对应的预测 Close
   * 和真实 `y_true` 对比可画出图 4 那种曲线，也可计算 MSE（表 5、表 6）

---

## 阶段 7：训练对比模型（LSTM / TCN / SVR / Random Forest）

### 输入

* 与 Transformer 相同的数据（同一批 alpha 特征 + 滑动窗口构造出 X, y）

### 操作

* 分别搭建四种模型：

  * LSTM（时序 RNN）
  * TCN（1D 卷积时序网络）
  * SVR（支持向量回归）
  * Random Forest（树模型集成）
* 损失/评估同样用 MSE。

### 输出

* 各模型在测试集上的预测结果 & MSE：

  * 汇总在表 5（只看 LLM alpha）
  * 表 10（对比不同 Alpha 来源：LLM / Featuretools / 101 人工 Alpha）

---

## 阶段 8：评估与扩展指标

### 8.1 预测性能评估

#### 输入

* 各模型预测的 `y_pred`
* 对应真实 `y_true`（测试集 Close）

#### 输出

* **MSE**（每公司、每模型）
* 对比：

  * Transformer vs 其他模型（表 5）
  * 有无情感特征 vs 有情感特征（表 6）
  * 不同 Alpha 生成方式（LLM / Featuretools / 101 人工 Alpha，表 10）

### 8.2 alpha 质量评估（IC）

#### 输入

* 每个 alpha_t 序列
* 对应的未来收益（return，论文中后面用 IC 来测 alpha 对收益的预测力）

#### 输出

* Information Coefficient（IC）：即 alpha 排序 vs 实际收益排序的相关系数（表 11）
* 文章结论：

  * LLM alpha 对“价格”预测很好，对“收益”预测相对较弱，但这不影响其在价格预测任务中的效果。

---

## 小结

1. **先跑完整的数据与特征流程（阶段 1–2）**：

   * 确保每家公司有一个 DataFrame，包含 OHLCV + 技术指标 + 目标公司/关联公司情感。

2. **决定 A 或 B：**

   * A：完全照论文的 alpha（表 4）重写计算代码。
   * B：自己再调一次 LLM 生成新的 alpha（保持 prompt 结构一致），做一次“方法级复现”。

3. **用这些 alpha 构造 5 天滑动窗口样本**，做成 Transformer 所需的 Encoder/Decoder 输入 + 标签。

4. **搭建 Transformer 结构**（照图 2 + 文中描述）并训练，记录测试 MSE。

5. **可选**：再训练 LSTM/TCN/SVR/RF，对比结果；再实现 Featuretools/101 Alpha 的流程，做三种 Alpha 的对比。