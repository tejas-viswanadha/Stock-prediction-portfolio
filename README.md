# 📈 Stock Portfolio & Hub

An end-to-end Machine Learning stock forecasting and portfolio analytics platform built in Python. The system fetches live market data with **automatic daily refresh on open**, computes technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, Volatility, Lags), trains return-based machine learning models (Random Forest, Gradient Boosting, Ridge Regression), forecasts **Tomorrow (1-Day)**, **Weekly (5-Day)**, and **Monthly (21-Day)** price targets, and tracks **overall portfolio gain or loss**.

---

## 🌟 Key Features

1. **Daily Auto-Refresh on Launch**:
   - Every time you open the dashboard or run the script, it automatically checks the latest market close and pulls fresh daily prices.
   - Provides a one-click manual force refresh button `🔄 Force Refresh Market Data`.

2. **Custom Owned Portfolio or Curated/Random Modes**:
   - **💼 My Custom Stocks (Owned Portfolio)**: Users can add any stocks they own (e.g. Apple, NVIDIA, Tesla, etc.) with custom share quantities and average purchase prices to track real dollar portfolio gains/losses and forecasts.
   - **⭐ Curated Top 10 Portfolio**: AAPL, MSFT, NVDA, AMZN, GOOGL, TSLA, JPM, JNJ, WMT, XOM.
   - **🎲 Random 10 Equities Mode**: Instantly draws randomized stocks from an expanded pool of 30+ S&P 500 giants.

3. **Multi-Horizon Predictions**:
   - **Tomorrow's Prediction (1-Day Ahead)**: Next-day forecast price, expected gain/loss ($, %), and Bullish/Bearish signal.
   - **Weekly Prediction (5-Day Ahead)**: 1-week price target, expected gain/loss ($, %), and medium-term trend.
   - **Monthly Prediction (21-Day Ahead)**: 1-month price target, expected gain/loss ($, %), and monthly outlook.

4. **Overall Portfolio Loss or Gain Tracking**:
   - **Realized Past Performance**: Past 1-Day, Past 1-Week (7D), and Past 1-Month (30D) realized profit/loss.
   - **Projected Future Net Gain/Loss**: Aggregate estimated portfolio return for Tomorrow, Next Week, and Next Month based on equal-weighted allocation ($10,000 base capital).

5. **Interactive Streamlit Web Dashboard (`app.py`)**:
   - **Executive Portfolio View**: Top KPI cards + Master 10-Stock multi-horizon comparison table with color heatmaps.
   - **Multi-Horizon Trajectory Cones**: Visualizes the path from Current Price ➔ Tomorrow ➔ 1-Week ➔ 1-Month.
   - **Interactive Technical Analysis**: Candlestick charts with Bollinger Bands, 20 SMA, RSI, and MACD.
   - **Feature Importance Drivers**: Top technical indicators influencing model predictions.
   - **Strategy Backtesting**: Model trading strategy vs. Buy & Hold benchmark with Sharpe ratio and max drawdown.

---

## 📁 Project Structure

```
stock_prediction/
├── config.py              # Stock universes, horizons (1d, 5d, 21d), technical configs
├── data_loader.py         # Daily yfinance fetcher, auto-updating cache & offline fallback
├── features.py            # Technical indicator & multi-horizon return calculations
├── models.py              # Return-based ML regressors (Tomorrow, Weekly, Monthly)
├── backtest.py            # Quantitative trading strategy simulator
├── main.py                # Command-line pipeline runner & visualizer
├── app.py                 # Streamlit interactive web dashboard
├── requirements.txt       # Python dependencies
├── data/                  # Cached stock datasets (auto-refreshed daily)
└── output/                # Generated reports & multi-horizon forecasts
    ├── multi_horizon_forecasts.csv
    ├── next_day_forecasts.csv
    ├── model_performance_metrics.csv
    └── charts/            # Stock prediction & backtest visual plots
```

---

## 🚀 How to Run

### 1. Launch the Streamlit Interactive Dashboard
Run from PowerShell in the project directory:
```powershell
cd C:\Users\Tejas\stock_prediction
.venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`. Every time you open it, it automatically refreshes for today's market data!

---

### 2. Run the Command-Line Pipeline
Run the multi-horizon pipeline on the curated 10 stocks:
```powershell
cd C:\Users\Tejas\stock_prediction
.venv\Scripts\python.exe main.py
```

Run on **10 randomly selected stocks**:
```powershell
.venv\Scripts\python.exe main.py --random
```

Run with a different algorithm (e.g., `gradient_boosting` or `ridge`):
```powershell
.venv\Scripts\python.exe main.py --random --model gradient_boosting
```

All CSV forecasts and comparison plots are automatically exported to `output/`.
