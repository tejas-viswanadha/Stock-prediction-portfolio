"""
Machine Learning Models Module: Training, Cross-Validation, Evaluation, and Forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import TEST_SIZE, RANDOM_STATE
from features import extract_features, prepare_dataset


class StockPredictor:
    """
    Handles model training, evaluation, and next-day forecasting for a stock.
    """

    def __init__(self, ticker: str, model_type: str = "random_forest"):
        self.ticker = ticker
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._init_model(model_type)
        self.model_weekly = self._init_model(model_type)
        self.model_monthly = self._init_model(model_type)
        self.feature_names = []
        self.metrics = {}
        self.predictions_df = None
        self.latest_features = None
        self.next_day_forecast = None
        self.weekly_forecast = None
        self.monthly_forecast = None
        self.multi_horizon_summary = None

    def _init_model(self, model_type: str):
        if model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                random_state=RANDOM_STATE
            )
        elif model_type == "ridge":
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train_and_evaluate(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts features, performs chronological train-test split, trains model,
        and computes comprehensive performance metrics.
        """
        featured_df = extract_features(raw_df)

        # The last row has no Target_Next_Close (future day), perfect for live prediction
        self.latest_features = featured_df.iloc[[-1]].copy()

        # Training set: drop rows with NaNs
        valid_df = featured_df.dropna().copy()
        if len(valid_df) < 50:
            raise ValueError(f"Not enough clean data points ({len(valid_df)}) for {self.ticker}")

        exclude_cols = [
            "Target_Tomorrow_Close", "Target_Tomorrow_Return",
            "Target_Weekly_Close", "Target_Weekly_Return",
            "Target_Monthly_Close", "Target_Monthly_Return",
            "Target_Next_Close", "Target_Next_Return", "Target_Direction",
            "Hist_1d_Return(%)", "Hist_Weekly_Return(%)", "Hist_Monthly_Return(%)",
            "Open", "High", "Low", "Adj Close"
        ]
        self.feature_names = [c for c in featured_df.columns if c not in exclude_cols]

        # Valid data for Tomorrow target
        valid_tomorrow = featured_df.dropna(subset=["Target_Tomorrow_Return"] + self.feature_names).copy()
        if len(valid_tomorrow) < 50:
            raise ValueError(f"Not enough clean data points ({len(valid_tomorrow)}) for {self.ticker}")

        X = valid_tomorrow[self.feature_names]
        y_return = valid_tomorrow["Target_Tomorrow_Return"]
        actual_closes = valid_tomorrow["Close"]
        actual_next_closes = valid_tomorrow["Target_Tomorrow_Close"]

        # Chronological Split (No lookahead bias)
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_return.iloc[:split_idx], y_return.iloc[split_idx:]
        test_closes = actual_closes.iloc[split_idx:]
        test_next_closes = actual_next_closes.iloc[split_idx:].values
        test_dates = valid_tomorrow.index[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1. Train Tomorrow model (predicts next-day return)
        self.model.fit(X_train_scaled, y_train)

        # 2. Train Weekly model (predicts 5-day return)
        valid_weekly = featured_df.dropna(subset=["Target_Weekly_Return"] + self.feature_names).copy()
        if len(valid_weekly) >= 50:
            X_w = self.scaler.transform(valid_weekly[self.feature_names])
            y_w = valid_weekly["Target_Weekly_Return"]
            self.model_weekly.fit(X_w, y_w)

        # 3. Train Monthly model (predicts 21-day return)
        valid_monthly = featured_df.dropna(subset=["Target_Monthly_Return"] + self.feature_names).copy()
        if len(valid_monthly) >= 50:
            X_m = self.scaler.transform(valid_monthly[self.feature_names])
            y_m = valid_monthly["Target_Monthly_Return"]
            self.model_monthly.fit(X_m, y_m)

        # Predict returns on test set and convert to price
        pred_returns = self.model.predict(X_test_scaled)
        y_pred = test_closes.values * (1 + pred_returns)

        # Evaluate Metrics on predicted price vs actual price
        rmse = float(np.sqrt(mean_squared_error(test_next_closes, y_pred)))
        mae = float(mean_absolute_error(test_next_closes, y_pred))
        mape = float(np.mean(np.abs((test_next_closes - y_pred) / test_next_closes)) * 100)
        r2 = float(r2_score(test_next_closes, y_pred))

        # Directional Accuracy: Did the model correctly predict Up vs Down return?
        actual_direction = (y_test.values > 0).astype(int)
        predicted_direction = (pred_returns > 0).astype(int)
        directional_acc = float(np.mean(actual_direction == predicted_direction) * 100)

        self.metrics = {
            "Ticker": self.ticker,
            "Model": self.model_type,
            "Train_Samples": len(X_train),
            "Test_Samples": len(X_test),
            "Last_Actual_Price": float(test_closes.iloc[-1]),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE(%)": round(mape, 2),
            "R2_Score": round(r2, 4),
            "Directional_Accuracy(%)": round(directional_acc, 2)
        }

        # Save test predictions DataFrame
        self.predictions_df = pd.DataFrame({
            "Date": test_dates,
            "Current_Close": test_closes.values,
            "Actual_Next_Close": test_next_closes,
            "Predicted_Next_Close": y_pred,
            "Actual_Direction": actual_direction,
            "Predicted_Direction": predicted_direction
        }).set_index("Date")

        # Multi-Horizon Live Forecasts from latest row
        latest_X = self.latest_features[self.feature_names]
        latest_X_scaled = self.scaler.transform(latest_X)
        current_price = float(raw_df["Close"].iloc[-1])
        prev_price = float(raw_df["Close"].iloc[-2]) if len(raw_df) > 1 else current_price
        week_ago_price = float(raw_df["Close"].iloc[-6]) if len(raw_df) > 5 else float(raw_df["Close"].iloc[0])
        month_ago_price = float(raw_df["Close"].iloc[-22]) if len(raw_df) > 21 else float(raw_df["Close"].iloc[0])

        # Historical Loss / Gain
        today_diff = current_price - prev_price
        today_pct = ((current_price - prev_price) / prev_price) * 100
        hist_week_diff = current_price - week_ago_price
        hist_week_pct = ((current_price - week_ago_price) / week_ago_price) * 100
        hist_month_diff = current_price - month_ago_price
        hist_month_pct = ((current_price - month_ago_price) / month_ago_price) * 100

        # Forecast Calculations: Tomorrow (1d), Weekly (5d), Monthly (21d)
        pred_1d_ret = float(self.model.predict(latest_X_scaled)[0])
        next_1d_price = current_price * (1 + pred_1d_ret)
        next_1d_diff = next_1d_price - current_price
        next_1d_pct = pred_1d_ret * 100
        sig_1d = "BUY / BULLISH" if next_1d_pct > 0.1 else ("SELL / BEARISH" if next_1d_pct < -0.1 else "HOLD / NEUTRAL")

        pred_5d_ret = float(self.model_weekly.predict(latest_X_scaled)[0]) if len(valid_weekly) >= 50 else pred_1d_ret
        next_5d_price = current_price * (1 + pred_5d_ret)
        next_5d_diff = next_5d_price - current_price
        next_5d_pct = pred_5d_ret * 100
        sig_5d = "BULLISH ↗" if next_5d_pct > 0.5 else ("BEARISH ↘" if next_5d_pct < -0.5 else "NEUTRAL ➔")

        pred_21d_ret = float(self.model_monthly.predict(latest_X_scaled)[0]) if len(valid_monthly) >= 50 else pred_5d_ret
        next_21d_price = current_price * (1 + pred_21d_ret)
        next_21d_diff = next_21d_price - current_price
        next_21d_pct = pred_21d_ret * 100
        sig_21d = "BULLISH ↗" if next_21d_pct > 1.0 else ("BEARISH ↘" if next_21d_pct < -1.0 else "NEUTRAL ➔")

        # Standard next_day_forecast for backward compatibility
        self.next_day_forecast = {
            "Ticker": self.ticker,
            "Current_Price": round(current_price, 2),
            "Forecast_Next_Close": round(next_1d_price, 2),
            "Expected_Change(%)": round(next_1d_pct, 2),
            "Signal": sig_1d
        }

        self.weekly_forecast = {
            "Ticker": self.ticker,
            "Current_Price": round(current_price, 2),
            "Forecast_Weekly_Close": round(next_5d_price, 2),
            "Expected_Weekly_Change(%)": round(next_5d_pct, 2),
            "Expected_Weekly_Change($)": round(next_5d_diff, 2),
            "Signal": sig_5d
        }

        self.monthly_forecast = {
            "Ticker": self.ticker,
            "Current_Price": round(current_price, 2),
            "Forecast_Monthly_Close": round(next_21d_price, 2),
            "Expected_Monthly_Change(%)": round(next_21d_pct, 2),
            "Expected_Monthly_Change($)": round(next_21d_diff, 2),
            "Signal": sig_21d
        }

        self.multi_horizon_summary = {
            "Ticker": self.ticker,
            "Current_Price": round(current_price, 2),
            "Today_Change($)": round(today_diff, 2),
            "Today_Change(%)": round(today_pct, 2),
            "Hist_Week_Change(%)": round(hist_week_pct, 2),
            "Hist_Month_Change(%)": round(hist_month_pct, 2),
            "Tomorrow_Forecast($)": round(next_1d_price, 2),
            "Tomorrow_Expected_Change(%)": round(next_1d_pct, 2),
            "Tomorrow_Signal": sig_1d,
            "Weekly_Forecast($)": round(next_5d_price, 2),
            "Weekly_Expected_Change(%)": round(next_5d_pct, 2),
            "Weekly_Signal": sig_5d,
            "Monthly_Forecast($)": round(next_21d_price, 2),
            "Monthly_Expected_Change(%)": round(next_21d_pct, 2),
            "Monthly_Signal": sig_21d
        }

        return self.metrics

        return self.metrics

    def get_feature_importances(self, top_n: int = 10) -> pd.DataFrame:
        """Returns top feature importances for tree-based models."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            df_imp = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(top_n)
            return df_imp
        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_)
            df_imp = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": coefs
            }).sort_values(by="Importance", ascending=False).head(top_n)
            return df_imp
        return pd.DataFrame()
