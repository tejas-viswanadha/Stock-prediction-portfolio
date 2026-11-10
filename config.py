"""
Configuration settings for the 10-Stock Prediction Project.
"""

from datetime import datetime, timedelta

# Common stock company name directory
COMPANY_NAMES = {
    "AAPL": "Apple Inc. (Technology)",
    "MSFT": "Microsoft Corp. (Technology)",
    "NVDA": "NVIDIA Corp. (Semiconductors)",
    "AMZN": "Amazon.com Inc. (Consumer Cyclical)",
    "GOOGL": "Alphabet Inc. (Communication Services)",
    "META": "Meta Platforms Inc. (Technology)",
    "TSLA": "Tesla Inc. (Automotive / Clean Energy)",
    "JPM": "JPMorgan Chase & Co. (Financial Services)",
    "JNJ": "Johnson & Johnson (Healthcare)",
    "WMT": "Walmart Inc. (Consumer Defensive)",
    "XOM": "Exxon Mobil Corp. (Energy)",
    "BRK-B": "Berkshire Hathaway (Financial Services)",
    "LLY": "Eli Lilly & Co. (Healthcare)",
    "V": "Visa Inc. (Financial Services)",
    "UNH": "UnitedHealth Group (Healthcare)",
    "MA": "Mastercard Inc. (Financial Services)",
    "PG": "Procter & Gamble (Consumer Defensive)",
    "HD": "Home Depot (Consumer Cyclical)",
    "COST": "Costco Wholesale (Consumer Defensive)",
    "ABBV": "AbbVie Inc. (Healthcare)",
    "MRK": "Merck & Co. (Healthcare)",
    "CVX": "Chevron Corp. (Energy)",
    "CRM": "Salesforce Inc. (Technology)",
    "BAC": "Bank of America (Financial Services)",
    "KO": "The Coca-Cola Co. (Consumer Defensive)",
    "NFLX": "Netflix Inc. (Communication Services)",
    "AMD": "Advanced Micro Devices (Semiconductors)",
    "PEP": "PepsiCo Inc. (Consumer Defensive)",
    "TMO": "Thermo Fisher Scientific (Healthcare)",
    "ADBE": "Adobe Inc. (Technology)",
    "PLTR": "Palantir Technologies (Technology)",
    "INTC": "Intel Corp. (Semiconductors)",
    "DIS": "Walt Disney Co. (Communication Services)",
    "COIN": "Coinbase Global (Financial / Crypto)"
}

DEFAULT_STOCKS = {
    "AAPL": "Apple Inc. (Technology)",
    "MSFT": "Microsoft Corp. (Technology)",
    "NVDA": "NVIDIA Corp. (Semiconductors)",
    "AMZN": "Amazon.com Inc. (Consumer Cyclical)",
    "GOOGL": "Alphabet Inc. (Communication Services)",
    "TSLA": "Tesla Inc. (Automotive / Clean Energy)",
    "JPM": "JPMorgan Chase & Co. (Financial Services)",
    "JNJ": "Johnson & Johnson (Healthcare)",
    "WMT": "Walmart Inc. (Consumer Defensive)",
    "XOM": "Exxon Mobil Corp. (Energy)"
}

# Broader pool for random selection
EXPANDED_STOCK_POOL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "V",
    "JPM", "UNH", "WMT", "MA", "XOM", "JNJ", "PG", "HD", "COST", "ABBV",
    "MRK", "CVX", "CRM", "BAC", "KO", "NFLX", "AMD", "PEP", "TMO", "ADBE"
]

# Historical data range
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = datetime.today().strftime("%Y-%m-%d")

# Feature parameters
SMA_WINDOWS = [5, 10, 20, 50]
EMA_WINDOWS = [12, 26]
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
LAG_DAYS = [1, 2, 3, 5]

# Multi-horizon forecast parameters
HORIZONS = {
    "tomorrow": {"days": 1, "label": "Tomorrow (1-Day)", "col": "Target_Tomorrow_Close"},
    "weekly": {"days": 5, "label": "Weekly (5-Day)", "col": "Target_Weekly_Close"},
    "monthly": {"days": 21, "label": "Monthly (21-Day)", "col": "Target_Monthly_Close"}
}

# Modeling parameters
TEST_SIZE = 0.2  # Most recent 20% for testing (chronological split)
RANDOM_STATE = 42

# Strategy backtesting parameters
INITIAL_CAPITAL = 10000.0  # $10,000 USD
PREDICTION_THRESHOLD = 0.001  # 0.1% expected price movement to trigger buy signal
