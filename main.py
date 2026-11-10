"""
Main Execution Script: Runs end-to-end stock prediction pipeline across 10 stocks.
"""

import os
import sys
import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_STOCKS
from data_loader import load_multiple_stocks, select_random_stocks
from models import StockPredictor
from backtest import run_backtest

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def plot_stock_predictions(ticker: str, pred_df: pd.DataFrame, save_path: str):
    """Plot Actual vs Predicted Close Prices with Trading Signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(pred_df.index, pred_df["Actual_Next_Close"], label="Actual Next Close", color="#1f77b4", linewidth=1.8)
    plt.plot(pred_df.index, pred_df["Predicted_Next_Close"], label="Predicted Next Close", color="#ff7f0e", linestyle="--", linewidth=1.5)
    
    plt.title(f"{ticker} - Actual vs Predicted Next-Day Close Price", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price (USD)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_backtest_curve(ticker: str, backtest_df: pd.DataFrame, save_path: str):
    """Plot Strategy Equity Curve vs Buy & Hold."""
    plt.figure(figsize=(12, 5))
    plt.plot(backtest_df.index, backtest_df["Strategy_Equity"], label="ML Strategy", color="#2ca02c", linewidth=2)
    plt.plot(backtest_df.index, backtest_df["BuyHold_Equity"], label="Buy & Hold Benchmark", color="#7f7f7f", linestyle="--", linewidth=1.5)
    
    plt.title(f"{ticker} - Strategy Cumulative Performance ($10,000 Initial)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Portfolio Value ($)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_summary_dashboard(all_metrics: List[Dict], save_path: str):
    """Generates a summary multi-panel chart comparing all 10 stocks."""
    df = pd.DataFrame(all_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.set_theme(style="whitegrid")

    # 1. Directional Accuracy
    sns.barplot(data=df, x="Ticker", y="Directional_Accuracy(%)", hue="Ticker", palette="viridis", legend=False, ax=axes[0, 0])
    axes[0, 0].axhline(50, color="red", linestyle="--", alpha=0.7, label="Random Guess (50%)")
    axes[0, 0].set_title("Directional Accuracy (%) [Higher is Better]", fontweight="bold")
    axes[0, 0].set_ylim(40, max(75, df["Directional_Accuracy(%)"].max() + 5))
    axes[0, 0].legend()

    # 2. MAPE (%)
    sns.barplot(data=df, x="Ticker", y="MAPE(%)", hue="Ticker", palette="crest", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("Mean Absolute Percentage Error (%) [Lower is Better]", fontweight="bold")

    # 3. Strategy Return vs Buy & Hold
    ret_df = df[["Ticker", "Strategy_Total_Return(%)", "BuyHold_Total_Return(%)"]].melt(
        id_vars="Ticker", var_name="Strategy", value_name="Return(%)"
    )
    ret_df["Strategy"] = ret_df["Strategy"].str.replace("_Total_Return(%)", "")
    sns.barplot(data=ret_df, x="Ticker", y="Return(%)", hue="Strategy", palette="Set2", ax=axes[1, 0])
    axes[1, 0].set_title("Strategy Return vs Buy & Hold (%)", fontweight="bold")

    # 4. Sharpe Ratio
    sns.barplot(data=df, x="Ticker", y="Sharpe_Ratio", hue="Ticker", palette="flare", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Annualized Sharpe Ratio", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_pipeline(tickers: List[str], model_type: str = "random_forest") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs data loading, training, forecasting, and backtesting for all tickers."""
    print("=" * 80)
    print(f" STOCK PREDICTION PIPELINE: 10 STOCKS | MODEL: {model_type.upper()} ")
    print("=" * 80)
    print(f"Target Tickers: {', '.join(tickers)}\n")

    # Step 1: Load Data
    print(">> Step 1/4: Fetching historical stock data...")
    stock_dfs = load_multiple_stocks(tickers)

    all_metrics = []
    all_forecasts = []
    all_multi_horizon = []

    # Step 2: Modeling & Prediction
    print(">> Step 2/4: Training models & generating test predictions...")
    for ticker in tickers:
        raw_df = stock_dfs[ticker]
        predictor = StockPredictor(ticker=ticker, model_type=model_type)
        metrics = predictor.train_and_evaluate(raw_df)

        # Backtesting
        bt_df, bt_metrics = run_backtest(predictor.predictions_df)
        combined_metrics = {**metrics, **bt_metrics}
        all_metrics.append(combined_metrics)

        # Forecasts
        all_forecasts.append(predictor.next_day_forecast)
        all_multi_horizon.append(predictor.multi_horizon_summary)

        # Step 3: Plotting individual stock charts
        pred_plot_path = os.path.join(CHARTS_DIR, f"{ticker}_predictions.png")
        plot_stock_predictions(ticker, predictor.predictions_df, pred_plot_path)

        bt_plot_path = os.path.join(CHARTS_DIR, f"{ticker}_backtest.png")
        plot_backtest_curve(ticker, bt_df, bt_plot_path)

        sum_row = predictor.multi_horizon_summary
        print(f"   [DONE] {ticker:<5} | Now: ${sum_row['Current_Price']} | 1D: {sum_row['Tomorrow_Expected_Change(%)']:+.2f}% | 1W: {sum_row['Weekly_Expected_Change(%)']:+.2f}% | 1M: {sum_row['Monthly_Expected_Change(%)']:+.2f}% | Signal: {sum_row['Tomorrow_Signal']}")

    # Step 4: Summary Analytics & Dashboard
    print("\n>> Step 3/4: Creating summary performance charts...")
    summary_chart_path = os.path.join(OUTPUT_DIR, "summary_dashboard.png")
    plot_summary_dashboard(all_metrics, summary_chart_path)

    metrics_df = pd.DataFrame(all_metrics)
    forecasts_df = pd.DataFrame(all_forecasts)
    multi_df = pd.DataFrame(all_multi_horizon)

    # Save to files
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_performance_metrics.csv"), index=False)
    forecasts_df.to_csv(os.path.join(OUTPUT_DIR, "next_day_forecasts.csv"), index=False)
    multi_df.to_csv(os.path.join(OUTPUT_DIR, "multi_horizon_forecasts.csv"), index=False)

    # Portfolio Overall Aggregation (Equal-weight $1,000 per stock = $10,000 total)
    avg_hist_week = multi_df["Hist_Week_Change(%)"].mean()
    avg_hist_month = multi_df["Hist_Month_Change(%)"].mean()
    avg_pred_tomorrow = multi_df["Tomorrow_Expected_Change(%)"].mean()
    avg_pred_week = multi_df["Weekly_Expected_Change(%)"].mean()
    avg_pred_month = multi_df["Monthly_Expected_Change(%)"].mean()

    print(">> Step 4/4: Complete! Saved results to output/\n")
    print("=" * 105)
    print(" 🌟 10-STOCK PORTFOLIO OVERALL PERFORMANCE & LOSS/GAIN SUMMARY 🌟 ")
    print("=" * 105)
    print(f"  • Past 1-Week Realized Portfolio Change:   {avg_hist_week:+.2f}%  (Net: ${avg_hist_week * 100:+.2f} on $10k)")
    print(f"  • Past 1-Month Realized Portfolio Change:  {avg_hist_month:+.2f}%  (Net: ${avg_hist_month * 100:+.2f} on $10k)")
    print(f"  • Tomorrow Projected Portfolio Gain/Loss:  {avg_pred_tomorrow:+.2f}%  (Est: ${avg_pred_tomorrow * 100:+.2f} on $10k)")
    print(f"  • Weekly Projected Portfolio Gain/Loss:    {avg_pred_week:+.2f}%  (Est: ${avg_pred_week * 100:+.2f} on $10k)")
    print(f"  • Monthly Projected Portfolio Gain/Loss:   {avg_pred_month:+.2f}%  (Est: ${avg_pred_month * 100:+.2f} on $10k)")
    print("=" * 105)

    print("\n" + "=" * 105)
    print(" 🔮 10 STOCKS: TOMORROW, WEEKLY & MONTHLY FORECAST TABLE ")
    print("=" * 105)
    display_multi = [
        "Ticker", "Current_Price", "Today_Change(%)",
        "Hist_Week_Change(%)", "Hist_Month_Change(%)",
        "Tomorrow_Forecast($)", "Tomorrow_Expected_Change(%)", "Tomorrow_Signal",
        "Weekly_Expected_Change(%)", "Weekly_Signal",
        "Monthly_Expected_Change(%)", "Monthly_Signal"
    ]
    print(multi_df[display_multi].to_string(index=False))
    print("=" * 105)

    print("\n" + "=" * 105)
    print(" 📈 OVERALL MODEL & STRATEGY EVALUATION METRICS ")
    print("=" * 105)
    display_cols = [
        "Ticker", "RMSE", "MAE", "MAPE(%)", "Directional_Accuracy(%)",
        "Strategy_Total_Return(%)", "BuyHold_Total_Return(%)", "Sharpe_Ratio", "Win_Rate(%)"
    ]
    print(metrics_df[display_cols].to_string(index=False))
    print("=" * 105)

    return metrics_df, multi_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Prediction on 10 Random Stocks")
    parser.add_argument("--random", action="store_true", help="Pick 10 random stocks from the expanded pool")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "gradient_boosting", "ridge"], help="ML Model algorithm")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatable random selection")
    args = parser.parse_args()

    if args.random:
        selected_tickers = select_random_stocks(n=10, seed=args.seed)
    else:
        selected_tickers = list(DEFAULT_STOCKS.keys())

    run_pipeline(selected_tickers, model_type=args.model)
