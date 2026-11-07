"""Downloading and caching daily price data from Yahoo Finance.

Yahoo revises adjusted prices over time and drops delisted tickers, so every download is
cached as a parquet file in ``data/``. Later runs read the cache, which keeps results
reproducible. Pass ``refresh=True`` to re-download.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import DATA_DIR, END_DATE, START_DATE


def download_close(tickers, start=START_DATE, end=END_DATE, allow_missing=False):
    # Auto-adjusted daily closing prices from Yahoo Finance as a (dates x tickers) DataFrame.
    # Tickers with no data (e.g. delisted) raise an error, or are dropped with a warning if allow_missing.
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    close = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing and not allow_missing:
        raise RuntimeError(f"No data returned for: {missing}")
    if missing:
        print(f"WARNING: no data returned for {missing} (possibly delisted); excluding them.")
    return close[[t for t in tickers if t not in missing]].sort_index()


def load_prices(name, tickers, allow_missing=False, refresh=False, cache_dir=DATA_DIR):
    # Read closing prices from data/<name>.parquet; download (and cache) them if the file doesn't exist.
    path = Path(cache_dir) / f"{name}.parquet"
    if path.exists() and not refresh:
        prices = pd.read_parquet(path)
        print(f"Loaded cached {path.name}: {prices.shape[0]} days x {prices.shape[1]} tickers")
        return prices
    prices = download_close(tickers, allow_missing=allow_missing)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    print(f"Downloaded and cached {path.name}: {prices.shape[0]} days x {prices.shape[1]} tickers")
    return prices
