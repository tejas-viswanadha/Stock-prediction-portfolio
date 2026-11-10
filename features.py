"""
Feature Engineering Module: Computes technical indicators and target variables.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from config import (
    SMA_WINDOWS,
    EMA_WINDOWS,
    RSI_WINDOW,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_WINDOW,
    BOLLINGER_STD,
    LAG_DAYS,
)


def compute_rsi(series: pd.Series, period: int = RSI_WINDOW) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's EMA smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing is equivalent to EMA with com = period - 1
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series,
    window: int = BOLLINGER_WINDOW,
    num_std: float = BOLLINGER_STD
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (Upper, Lower, Middle)."""
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    return upper, middle, lower


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive technical indicators and historical lag features.
    """
    data = df.copy()

    # 1. Moving Averages
    for w in SMA_WINDOWS:
        data[f"SMA_{w}"] = data["Close"].rolling(window=w).mean()
        data[f"Close_to_SMA_{w}"] = data["Close"] / (data[f"SMA_{w}"] + 1e-10)

    for w in EMA_WINDOWS:
        data[f"EMA_{w}"] = data["Close"].ewm(span=w, adjust=False).mean()

    # 2. RSI
    data["RSI"] = compute_rsi(data["Close"])

    # 3. MACD
    macd, signal, hist = compute_macd(data["Close"])
    data["MACD"] = macd
    data["MACD_Signal"] = signal
    data["MACD_Hist"] = hist

    # 4. Bollinger Bands
    upper, middle, lower = compute_bollinger_bands(data["Close"])
    data["BB_Upper"] = upper
    data["BB_Lower"] = lower
    data["BB_Width"] = (upper - lower) / (middle + 1e-10)
    data["BB_Pct"] = (data["Close"] - lower) / (upper - lower + 1e-10)

    # 5. Volatility & ATR
    data["Daily_Return"] = data["Close"].pct_change()
    data["Vol_10d"] = data["Daily_Return"].rolling(window=10).std()
    data["Vol_20d"] = data["Daily_Return"].rolling(window=20).std()
    data["ATR_14"] = compute_atr(data, window=14)

    # 6. Price Momentum & Lags
    for lag in LAG_DAYS:
        data[f"Lag_Return_{lag}d"] = data["Daily_Return"].shift(lag)
        data[f"Lag_Close_{lag}d"] = data["Close"].shift(lag)

    # 7. Volume features
    data["Vol_SMA_10"] = data["Volume"].rolling(window=10).mean()
    data["Volume_Ratio"] = data["Volume"] / (data["Vol_SMA_10"] + 1e-10)

    # 8. High-Low Spread
    data["HL_Spread"] = (data["High"] - data["Low"]) / (data["Close"] + 1e-10)

    # Historical Performance Tracking (Actual Past Loss/Gain)
    data["Hist_1d_Return(%)"] = data["Close"].pct_change(1) * 100
    data["Hist_Weekly_Return(%)"] = data["Close"].pct_change(5) * 100
    data["Hist_Monthly_Return(%)"] = data["Close"].pct_change(21) * 100

    # Target variables (Lookahead for multi-horizon training)
    data["Target_Tomorrow_Close"] = data["Close"].shift(-1)
    data["Target_Tomorrow_Return"] = (data["Target_Tomorrow_Close"] - data["Close"]) / data["Close"]

    data["Target_Weekly_Close"] = data["Close"].shift(-5)
    data["Target_Weekly_Return"] = (data["Target_Weekly_Close"] - data["Close"]) / data["Close"]

    data["Target_Monthly_Close"] = data["Close"].shift(-21)
    data["Target_Monthly_Return"] = (data["Target_Monthly_Close"] - data["Close"]) / data["Close"]

    # Aliases for 1-day ahead strategy/backtest compatibility
    data["Target_Next_Close"] = data["Target_Tomorrow_Close"]
    data["Target_Next_Return"] = data["Target_Tomorrow_Return"]
    data["Target_Direction"] = (data["Target_Next_Return"] > 0).astype(int)

    return data


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Cleans dataset, drops NaNs from rolling indicators, and separates feature matrix X from targets.
    """
    featured = extract_features(df)
    
    # Exclude targets and raw pricing columns from features
    target_cols = [
        "Target_Tomorrow_Close", "Target_Tomorrow_Return",
        "Target_Weekly_Close", "Target_Weekly_Return",
        "Target_Monthly_Close", "Target_Monthly_Return",
        "Target_Next_Close", "Target_Next_Return", "Target_Direction"
    ]
    raw_cols = ["Open", "High", "Low", "Adj Close"]
    exclude_cols = target_cols + raw_cols

    # Drop rows that don't have valid tomorrow targets for next-day model
    valid_data = featured.dropna(subset=["Target_Tomorrow_Close"]).copy()
    feature_cols = [c for c in valid_data.columns if c not in exclude_cols]

    X = valid_data[feature_cols]
    y_reg = valid_data["Target_Tomorrow_Close"]
    y_clf = valid_data["Target_Direction"]

    return X, y_reg, y_clf, feature_cols
