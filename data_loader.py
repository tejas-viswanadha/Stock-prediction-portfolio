"""
Data Loader Module: Fetches, caches, and validates stock price history.
"""

import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from config import DEFAULT_STOCKS, EXPANDED_STOCK_POOL, DEFAULT_START_DATE, DEFAULT_END_DATE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def select_random_stocks(n: int = 10, seed: Optional[int] = None) -> List[str]:
    """Select n random stock tickers from the expanded pool."""
    if seed is not None:
        random.seed(seed)
    return random.sample(EXPANDED_STOCK_POOL, min(n, len(EXPANDED_STOCK_POOL)))


def generate_synthetic_stock_data(
    ticker: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    initial_price: float = 150.0
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data as a fallback."""
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    n = len(dates)
    np.random.seed(abs(hash(ticker)) % 100000)

    # Geometric Brownian Motion
    daily_returns = np.random.normal(0.0005, 0.018, n)
    price_series = initial_price * np.exp(np.cumsum(daily_returns))

    high = price_series * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = price_series * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_prices = low + (high - low) * np.random.uniform(0.1, 0.9, n)
    volume = np.random.lognormal(mean=16, sigma=0.5, size=n).astype(int)

    df = pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": price_series,
        "Adj Close": price_series,
        "Volume": volume
    }, index=dates)
    df.index.name = "Date"
    return df


def fetch_stock_data(
    ticker: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV data for a ticker with automatic daily refresh.
    First checks local cache to see if today's latest market close is present.
    If older than the latest business day or force_refresh=True, fetches fresh live data.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    cache_path = os.path.join(DATA_DIR, f"{ticker}_daily.csv")

    # Check cache freshness
    if use_cache and not force_refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty and len(df) >= 50:
                last_dt = pd.to_datetime(df.index[-1]).date()
                today_dt = datetime.today().date()
                # If cached within 3 days (accounts for weekend Fri-Sun/Mon), considered fresh
                days_diff = (today_dt - last_dt).days
                if days_diff <= 3:
                    return df
        except Exception:
            pass

    # Fetch live data via yfinance
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        req_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if "Adj Close" not in data.columns and "Close" in data.columns:
            data["Adj Close"] = data["Close"]

        data = data[[c for c in req_cols if c in data.columns]]
        data = data.dropna()

        if len(data) >= 50:
            if use_cache:
                data.to_csv(cache_path)
            return data
    except Exception as e:
        print(f"Notice: Live download error for {ticker} ({e}). Checking cache/fallback.")

    # Fallback to local cache if available
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty and len(df) >= 30:
                return df
        except Exception:
            pass

    # Fallback to realistic synthetic series if network is completely unreachable
    df = generate_synthetic_stock_data(ticker, start_date, end_date)
    if use_cache:
        df.to_csv(cache_path)
    return df


def load_multiple_stocks(
    tickers: List[str],
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """Download or load historical data for a list of tickers, updating daily."""
    stocks_data = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        stocks_data[ticker] = df
    return stocks_data


if __name__ == "__main__":
    sample_tickers = list(DEFAULT_STOCKS.keys())[:3]
    print(f"Testing data loader for: {sample_tickers}")
    data = load_multiple_stocks(sample_tickers)
    for t, df in data.items():
        print(f"{t}: {df.shape[0]} rows, latest close = ${df['Close'].iloc[-1]:.2f}")
