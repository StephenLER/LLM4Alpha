import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime
import time

# Stooq 的 ticker 写法（可以先用这版试，拉不到再单独调整）
# - .US: 美国
# - .HK: 香港
# - .JP: 日本
TICKERS_STOOQ = {
    "AAPL.US":  "Apple",       # Apple (US)
    "PEP.US":   "PepsiCo",     # Pepsi (US)
    "7203.JP":  "Toyota",      # Toyota (Tokyo)
    "700.HK":  "Tencent",     # Tencent (HK)
    "5.HK":  "HSBC",        # HSBC Holdings (HK) — 你也可以改成 HSBA.UK 试试伦敦上市
}

START_DATE = "2016-05-01"
END_DATE   = "2024-05-08"


def download_from_stooq(symbol, start, end, max_retries=3):
    """
    使用 pandas-datareader 从 Stooq 下载单个股票的日线数据。
    返回 DataFrame（列通常是 Open/High/Low/Close/Volume），
    Stooq 默认是由近到远排序，需要我们手动 sort_index()。
    """
    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)

    for attempt in range(1, max_retries + 1):
        try:
            df = pdr.DataReader(symbol, "stooq", start, end)
            # Stooq 返回是倒序（最新在前），改成按时间正序：
            df = df.sort_index()

            if df is None or df.empty:
                print(f"[WARN] {symbol} 从 Stooq 下载成功但数据为空")
                return None

            # 统一一下列名
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })
            df.index.name = "Date"
            return df

        except Exception as e:
            print(f"[ERROR] 第 {attempt} 次下载 {symbol} 失败: {e}")
            time.sleep(2 * attempt)

    print(f"[FATAL] {symbol} 在重试 {max_retries} 次后仍然失败，跳过。")
    return None


def download_ohlcv_for_tickers_stooq(ticker_map, start, end):
    data_by_symbol = {}
    for symbol, name in ticker_map.items():
        print(f"\n=== 正在从 Stooq 下载 {symbol} ({name}) ===")
        df = download_from_stooq(symbol, start, end)
        if df is None or df.empty:
            print(f"[WARN] {symbol} 没有可用数据，略过。")
        else:
            print(df.head())
            print(df.tail())
            print(f"{symbol} 共 {len(df)} 条记录")
            df.to_csv(f"/home/wangyuting/share/quant/wangyuting/liangjian/alpha/data/data_price_{symbol.replace('.', '_')}.csv")
            data_by_symbol[symbol] = df
        time.sleep(1)
    return data_by_symbol


if __name__ == "__main__":
    data_dict = download_ohlcv_for_tickers_stooq(
        TICKERS_STOOQ, START_DATE, END_DATE
    )
