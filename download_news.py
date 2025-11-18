"""
download_news_eodhd.py

使用 EODHD 的 Financial News API，为 5 家公司下载新闻，
并生成后续情感分析所需的 CSV 文件。

输出：
    - news_AAPL.csv
    - news_PEP.csv
    - news_TOYOTA.csv
    - news_TENCENT.csv
    - news_HSBC.csv
    - news_all.csv  （汇总所有公司）
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import requests


# ========= 1. 配置区域 =========

# TODO: 在 EODHD 注册后，把你的 API Token 填到这里：
# 注册地址在官网（Sign up & Get free API key）:contentReference[oaicite:4]{index=4}
API_TOKEN = "PUT_YOUR_EODHD_API_TOKEN_HERE"

BASE_URL = "https://eodhd.com/api/news"

# 论文的时间区间
START_DATE = "2016-05-01"
END_DATE = "2024-05-08"

# 为了避免一次请求太大，按窗口分段（比如每 90 天一个区间）
WINDOW_DAYS = 90

# 分页参数：一次最多取多少条（EODHD 文档写最大 1000）:contentReference[oaicite:5]{index=5}
LIMIT = 200

# 右侧是 EODHD 使用的代码（可能需要你根据实际账号中的支持列表再微调）
COMPANIES = {
    "AAPL":   "AAPL.US",   # Apple
    "PEP":    "PEP.US",    # PepsiCo
    "TOYOTA": "7203.TSE",  # Toyota on Tokyo (若报错可以尝试 TM.US 或 7203.T)
    "TENCENT": "0700.HK",  # Tencent Holdings (HK)
    "HSBC":   "0005.HK",   # HSBC Holdings (HK)
}

# 输出目录
OUTPUT_DIR = "news"


# ========= 2. 工具函数 =========

def daterange_chunks(start_date: str, end_date: str, window_days: int):
    """
    把 [start_date, end_date] 这个时间段切成若干小块，
    每块长度 <= window_days（避免一次拉太多数据）
    """
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=window_days - 1), end)
        yield cur.date().isoformat(), chunk_end.date().isoformat()
        cur = chunk_end + timedelta(days=1)


def fetch_news_for_symbol(company_name: str,
                          api_symbol: str,
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
    """
    为某一个公司（一个 EODHD ticker）在指定时间范围内拉全部新闻，
    自动处理时间分段 + 分页。
    返回 DataFrame，列包括：
        company, api_symbol, date, title, content, link, symbols, tags,
        api_polarity, api_neg, api_neu, api_pos
    """
    if API_TOKEN == "PUT_YOUR_EODHD_API_TOKEN_HERE":
        raise ValueError("请先在脚本顶部填入你的 EODHD API_TOKEN。")

    all_rows = []

    print(f"\n=== Fetching news for {company_name} ({api_symbol}) ===")

    for chunk_start, chunk_end in daterange_chunks(start_date, end_date, WINDOW_DAYS):
        print(f"  - 时间窗口: {chunk_start} ~ {chunk_end}")

        offset = 0
        while True:
            params = {
                "s": api_symbol,
                "from": chunk_start,
                "to": chunk_end,
                "offset": offset,
                "limit": LIMIT,
                "api_token": API_TOKEN,
                "fmt": "json",
            }

            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                print(f"    [ERROR] 请求失败 offset={offset}: {e}")
                # 碰到网络/权限问题时可以 break 或重试，这里先 break
                break

            try:
                data = resp.json()
            except Exception as e:
                print(f"    [ERROR] JSON 解析失败 offset={offset}: {e}")
                break

            # 如果该窗口没有更多新闻，data 会是空列表
            if not data:
                print(f"    [INFO] 该时间窗口在 offset={offset} 处无更多数据。")
                break

            # 解析每条新闻
            for article in data:
                # EODHD 文档里的字段：date, title, content, link, symbols, tags, sentiment:contentReference[oaicite:6]{index=6}
                date_str = article.get("date")
                title = article.get("title")
                content = article.get("content")
                link = article.get("link")
                symbols = article.get("symbols", [])
                tags = article.get("tags", [])
                sentiment = article.get("sentiment", {}) or {}

                row = {
                    "company": company_name,        # 我们内部使用的公司标记
                    "api_symbol": api_symbol,       # 调 API 用的 ticker 名
                    "date": date_str,
                    "title": title,
                    "content": content,
                    "link": link,
                    "symbols": ",".join(symbols),
                    "tags": ",".join(tags),
                    "api_polarity": sentiment.get("polarity"),
                    "api_neg": sentiment.get("neg"),
                    "api_neu": sentiment.get("neu"),
                    "api_pos": sentiment.get("pos"),
                }
                all_rows.append(row)

            print(f"    [INFO] 获取 {len(data)} 条新闻（offset={offset}）")

            # 如果本页少于 LIMIT，说明没有下一页了
            if len(data) < LIMIT:
                break

            offset += LIMIT
            # 稍微睡一下，避免触发频率限制
            time.sleep(0.2)

        # 每个时间窗口之间也休息一下
        time.sleep(0.5)

    if not all_rows:
        print(f"[WARN] {company_name} 在整个时间段内未获取到任何新闻。")

    df = pd.DataFrame(all_rows)

    # 规范处理日期格式
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df


# ========= 3. 主流程 =========

def main():
    all_dfs = []

    # 确保输出目录存在
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for company_name, api_symbol in COMPANIES.items():
        df_company = fetch_news_for_symbol(
            company_name=company_name,
            api_symbol=api_symbol,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        # 为每个公司单独存一份
        out_path = f"{OUTPUT_DIR}/news_{company_name}.csv"
        df_company.to_csv(out_path, index=False)
        print(f"=== 保存 {company_name} 新闻到: {out_path}，共 {len(df_company)} 条 ===")

        all_dfs.append(df_company)

    # 合并所有公司新闻
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all = df_all.sort_values(["date", "company"]).reset_index(drop=True)

        # 对齐后续情感代码使用的格式：date / ticker / title / content
        df_all_for_sentiment = pd.DataFrame({
            "date": df_all["date"].dt.date.astype(str),
            "ticker": df_all["company"],  # 用逻辑公司名当作 ticker
            "title": df_all["title"],
            "content": df_all["content"],
            "link": df_all["link"],
        })

        out_all_path = f"{OUTPUT_DIR}/news_all.csv"
        df_all_for_sentiment.to_csv(out_all_path, index=False)
        print(f"\n=== 合并所有公司新闻到: {out_all_path}，共 {len(df_all_for_sentiment)} 条 ===")
    else:
        print("[WARN] 没有任何新闻被下载到，可能是 API token / 计划权限问题。")


if __name__ == "__main__":
    main()
