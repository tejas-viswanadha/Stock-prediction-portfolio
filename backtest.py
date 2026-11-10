"""
Backtesting Module: Simulates trading strategy using model predictions against Buy & Hold.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from config import INITIAL_CAPITAL, PREDICTION_THRESHOLD


def run_backtest(
    predictions_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    threshold: float = PREDICTION_THRESHOLD
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backtests a basic long-or-cash strategy based on next-day predictions.
    
    Rule:
    - If Predicted_Next_Close > Current_Close * (1 + threshold), position = 1 (Long)
    - Else, position = 0 (Cash)
    """
    df = predictions_df.copy()

    # Actual daily return of the stock from Close_t to Close_{t+1}
    df["Actual_Return"] = (df["Actual_Next_Close"] - df["Current_Close"]) / df["Current_Close"]

    # Model predicted expected return
    df["Predicted_Return"] = (df["Predicted_Next_Close"] - df["Current_Close"]) / df["Current_Close"]

    # Trading Signal: 1 (Long) if predicted return exceeds positive threshold, else 0 (Cash)
    df["Signal"] = (df["Predicted_Return"] > threshold).astype(int)

    # Strategy Return: When Signal=1, earn Actual_Return. When 0, earn 0%.
    df["Strategy_Return"] = df["Signal"] * df["Actual_Return"]

    # Cumulative Wealth
    df["Strategy_Equity"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()
    df["BuyHold_Equity"] = initial_capital * (1 + df["Actual_Return"]).cumprod()

    # Cumulative Returns (%)
    strategy_total_return = ((df["Strategy_Equity"].iloc[-1] - initial_capital) / initial_capital) * 100
    buyhold_total_return = ((df["BuyHold_Equity"].iloc[-1] - initial_capital) / initial_capital) * 100

    # Drawdown Calculation
    strat_rolling_max = df["Strategy_Equity"].cummax()
    strat_drawdowns = (df["Strategy_Equity"] - strat_rolling_max) / strat_rolling_max
    max_drawdown_strat = float(strat_drawdowns.min() * 100)

    bh_rolling_max = df["BuyHold_Equity"].cummax()
    bh_drawdowns = (df["BuyHold_Equity"] - bh_rolling_max) / bh_rolling_max
    max_drawdown_bh = float(bh_drawdowns.min() * 100)

    # Annualized Sharpe Ratio (assuming 252 trading days, 0% risk free rate)
    strat_mean = df["Strategy_Return"].mean()
    strat_std = df["Strategy_Return"].std()
    sharpe = float((strat_mean / (strat_std + 1e-10)) * np.sqrt(252)) if strat_std > 0 else 0.0

    # Win Rate of Trades taken (when Signal == 1)
    trades_taken = df[df["Signal"] == 1]
    if len(trades_taken) > 0:
        win_rate = float((trades_taken["Actual_Return"] > 0).mean() * 100)
    else:
        win_rate = 0.0

    metrics = {
        "Initial_Capital": initial_capital,
        "Final_Strategy_Equity": round(float(df["Strategy_Equity"].iloc[-1]), 2),
        "Final_BuyHold_Equity": round(float(df["BuyHold_Equity"].iloc[-1]), 2),
        "Strategy_Total_Return(%)": round(strategy_total_return, 2),
        "BuyHold_Total_Return(%)": round(buyhold_total_return, 2),
        "Alpha_vs_BuyHold(%)": round(strategy_total_return - buyhold_total_return, 2),
        "Sharpe_Ratio": round(sharpe, 2),
        "Max_Drawdown_Strategy(%)": round(max_drawdown_strat, 2),
        "Max_Drawdown_BuyHold(%)": round(max_drawdown_bh, 2),
        "Trades_Count": int(df["Signal"].sum()),
        "Win_Rate(%)": round(win_rate, 2)
    }

    return df, metrics
