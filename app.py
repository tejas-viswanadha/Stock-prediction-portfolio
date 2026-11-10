"""
Streamlit Web Dashboard for 10-Stock Prediction & Analytics.
Supports Daily Auto-Refresh, Multi-Horizon Forecasts (Tomorrow, Weekly, Monthly),
and Overall Loss or Gain Tracking.

Run with: streamlit run app.py
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_STOCKS, EXPANDED_STOCK_POOL, COMPANY_NAMES, DEFAULT_START_DATE, DEFAULT_END_DATE
from data_loader import fetch_stock_data, select_random_stocks
from features import extract_features
from models import StockPredictor
from backtest import run_backtest

# Streamlit Page Configuration
st.set_page_config(
    page_title="Stock Portfolio & Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-bullish {
        background-color: #1b5e20;
        color: #81c784;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-bearish {
        background-color: #b71c1c;
        color: #ef9a9a;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neutral {
        background-color: #37474f;
        color: #cfd8dc;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("⚙️ Portfolio & Controls")

portfolio_mode = st.sidebar.radio(
    "Select Portfolio Type",
    [
        "💼 My Custom Stocks (Owned Portfolio)",
        "⭐ Curated Top 10 Leaders",
        "🎲 Random 10 Equities"
    ],
    index=0,
    help="Choose between your own stocks (e.g. Apple, NVIDIA), curated market leaders, or random equities."
)

holdings = {}

if portfolio_mode == "💼 My Custom Stocks (Owned Portfolio)":
    st.sidebar.markdown("### 🏷️ Add or Choose Your Stocks")

    # Session state for user-added tickers
    if "user_stocks" not in st.session_state:
        st.session_state.user_stocks = ["AAPL", "NVDA"]  # Default: Apple & NVIDIA

    # Input to add ANY custom ticker
    col_add1, col_add2 = st.sidebar.columns([3, 1])
    with col_add1:
        new_ticker = st.text_input(
            "Add Ticker Symbol:",
            placeholder="e.g. TSLA, PLTR, AMD",
            label_visibility="collapsed",
            key="new_ticker_input"
        ).strip().upper()
    with col_add2:
        add_clicked = st.button("➕ Add", use_container_width=True)

    if add_clicked and new_ticker:
        if new_ticker not in st.session_state.user_stocks:
            st.session_state.user_stocks.append(new_ticker)
            st.sidebar.success(f"Added {new_ticker}!")
            st.rerun()
        else:
            st.sidebar.info(f"{new_ticker} already in list.")

    # Quick preset buttons
    col_p1, col_p2 = st.sidebar.columns(2)
    if col_p1.button("🍎 Apple & NVIDIA", use_container_width=True):
        st.session_state.user_stocks = ["AAPL", "NVDA"]
        st.rerun()
    if col_p2.button("🚀 Top 5 Tech", use_container_width=True):
        st.session_state.user_stocks = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL"]
        st.rerun()

    # Dynamic pool of choices
    pool_choices = sorted(list(set(st.session_state.user_stocks + list(COMPANY_NAMES.keys()))))
    tickers = st.sidebar.multiselect(
        "Active Stocks in Your Portfolio:",
        options=pool_choices,
        default=st.session_state.user_stocks,
        help="Select the stocks you own or wish to analyze."
    )

    # Optional holdings / shares owned input
    with st.sidebar.expander("📝 Enter Shares Owned (Optional)", expanded=False):
        st.caption("Specify shares owned to compute your exact portfolio value and dollar gain/loss:")
        for t in tickers:
            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                sh = st.number_input(f"{t} Shares", min_value=0.0, value=10.0, step=1.0, key=f"shares_{t}")
            with col_s2:
                bp = st.number_input(f"{t} Buy Price ($)", min_value=0.0, value=0.0, step=1.0, key=f"buy_price_{t}")
            holdings[t] = {"shares": sh, "buy_price": bp}

elif portfolio_mode == "⭐ Curated Top 10 Leaders":
    tickers = list(DEFAULT_STOCKS.keys())
else:
    rand_seed = st.sidebar.number_input("Random Seed", value=42, step=1, help="Change seed to draw a different 10-stock portfolio")
    tickers = select_random_stocks(n=10, seed=int(rand_seed))

model_type = st.sidebar.selectbox(
    "Prediction Algorithm",
    ["random_forest", "gradient_boosting", "ridge"],
    help="Machine learning model architecture used for return and price regression."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Active Portfolio ({len(tickers)} Stocks):")
for t in tickers:
    name = COMPANY_NAMES.get(t, DEFAULT_STOCKS.get(t, f"{t} Stock"))
    st.sidebar.markdown(f"- **{t}**: {name}")

# Force refresh button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Refresh Market Data"):
    st.cache_data.clear()
    st.sidebar.success("Market cache cleared! Re-fetching latest data...")
    st.rerun()


# Cached data and model training (TTL = 3600 seconds for daily auto-refresh)
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock(ticker: str):
    return fetch_stock_data(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_analysis(ticker: str, m_type: str):
    df = load_stock(ticker)
    predictor = StockPredictor(ticker=ticker, model_type=m_type)
    metrics = predictor.train_and_evaluate(df)
    bt_df, bt_metrics = run_backtest(predictor.predictions_df)
    imp_df = predictor.get_feature_importances()
    summary = predictor.multi_horizon_summary
    return predictor, metrics, bt_df, bt_metrics, imp_df, summary, df


# Page Header with Real-Time Daily Status
today_str = datetime.today().strftime("%A, %B %d, %Y")
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📈 Stock Portfolio & Hub")
    st.caption(f"End-to-End Quantitative Machine Learning | Multi-Horizon Forecasts (Tomorrow, Weekly, Monthly) | Auto-refreshed for {today_str}")

with col_h2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 15px;">
        <span style="color: #4caf50; font-weight: bold;">● Live Market Feed</span><br>
        <small style="color: #9e9e9e;">Auto-refreshes on open</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# Step 1: Pre-compute Multi-Horizon Analysis for all Stocks
if not tickers:
    st.warning("⚠️ Please select at least one stock to track in the sidebar.")
    st.stop()

all_summaries = []
model_dict = {}

with st.spinner(f"Analyzing {len(tickers)} stock(s) with {model_type.upper()}..."):
    for t in tickers:
        try:
            pred, met, bdf, bmet, imp, summ, raw = get_stock_analysis(t, model_type)
            all_summaries.append(summ)
            model_dict[t] = {
                "predictor": pred,
                "metrics": met,
                "bt_df": bdf,
                "bt_metrics": bmet,
                "imp_df": imp,
                "raw_df": raw
            }
        except Exception as e:
            st.error(f"Error analyzing {t}: {e}")

if not all_summaries:
    st.error("No stock data could be processed. Please check network connection.")
    st.stop()

summary_df = pd.DataFrame(all_summaries)

# Calculate Overall Portfolio Metrics (Custom holdings or $10,000 baseline)
has_custom_shares = bool(holdings) and any(h.get("shares", 0) > 0 for h in holdings.values())

if has_custom_shares:
    summary_df["Shares_Owned"] = summary_df["Ticker"].map(lambda t: holdings.get(t, {}).get("shares", 0.0))
    summary_df["Position_Value"] = summary_df["Shares_Owned"] * summary_df["Current_Price"]
    total_portfolio_value = summary_df["Position_Value"].sum()

    today_dollar_change = (summary_df["Position_Value"] * (summary_df["Today_Change(%)"] / 100)).sum()
    today_pct_change = (today_dollar_change / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

    week_dollar_hist = (summary_df["Position_Value"] * (summary_df["Hist_Week_Change(%)"] / 100)).sum()
    week_pct_hist = (week_dollar_hist / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

    month_dollar_hist = (summary_df["Position_Value"] * (summary_df["Hist_Month_Change(%)"] / 100)).sum()
    month_pct_hist = (month_dollar_hist / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

    tomorrow_proj_dollar = (summary_df["Shares_Owned"] * (summary_df["Tomorrow_Forecast($)"] - summary_df["Current_Price"])).sum()
    tomorrow_proj_pct = (tomorrow_proj_dollar / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

    weekly_proj_dollar = (summary_df["Shares_Owned"] * (summary_df["Weekly_Forecast($)"] - summary_df["Current_Price"])).sum()
    weekly_proj_pct = (weekly_proj_dollar / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

    monthly_proj_dollar = (summary_df["Shares_Owned"] * (summary_df["Monthly_Forecast($)"] - summary_df["Current_Price"])).sum()
    monthly_proj_pct = (monthly_proj_dollar / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
else:
    total_portfolio_value = 10000.0
    today_pct_change = summary_df["Today_Change(%)"].mean()
    today_dollar_change = (today_pct_change / 100) * total_portfolio_value

    week_pct_hist = summary_df["Hist_Week_Change(%)"].mean()
    week_dollar_hist = (week_pct_hist / 100) * total_portfolio_value

    month_pct_hist = summary_df["Hist_Month_Change(%)"].mean()
    month_dollar_hist = (month_pct_hist / 100) * total_portfolio_value

    tomorrow_proj_pct = summary_df["Tomorrow_Expected_Change(%)"].mean()
    tomorrow_proj_dollar = (tomorrow_proj_pct / 100) * total_portfolio_value

    weekly_proj_pct = summary_df["Weekly_Expected_Change(%)"].mean()
    weekly_proj_dollar = (weekly_proj_pct / 100) * total_portfolio_value

    monthly_proj_pct = summary_df["Monthly_Expected_Change(%)"].mean()
    monthly_proj_dollar = (monthly_proj_pct / 100) * total_portfolio_value

# ==============================================================================
# SECTION 1: EXECUTIVE PORTFOLIO LOSS / GAIN OVERVIEW
# ==============================================================================
st.subheader(f"💼 Overall Portfolio Loss & Gain Summary ({len(tickers)} Stocks)")
if has_custom_shares:
    st.markdown(f"Performance weighted by your exact holdings across **{len(tickers)} owned stock(s)** (Total Portfolio Value: **${total_portfolio_value:,.2f}**).")
    kpi0, kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(6)
    kpi0.metric(
        label="Total Portfolio Value",
        value=f"${total_portfolio_value:,.2f}",
        delta=f"{len(tickers)} Stocks Tracked"
    )
else:
    st.markdown(f"Performance based on an equal-weighted portfolio ($10,000 base capital / ${10000/len(tickers):,.0f} per stock).")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    label="Today's Realized Change",
    value=f"${today_dollar_change:+,.2f}",
    delta=f"{today_pct_change:+.2f}% Today"
)

kpi2.metric(
    label="Past 1-Week Gain / Loss",
    value=f"${week_dollar_hist:+,.2f}",
    delta=f"{week_pct_hist:+.2f}% 7D Realized"
)

kpi3.metric(
    label="Past 1-Month Gain / Loss",
    value=f"${month_dollar_hist:+,.2f}",
    delta=f"{month_pct_hist:+.2f}% 30D Realized"
)

kpi4.metric(
    label="🔮 Tomorrow Projected Gain",
    value=f"${tomorrow_proj_dollar:+,.2f}",
    delta=f"{tomorrow_proj_pct:+.2f}% Tomorrow",
    delta_color="normal"
)

kpi5.metric(
    label="📈 Monthly Projected Gain",
    value=f"${monthly_proj_dollar:+,.2f}",
    delta=f"{monthly_proj_pct:+.2f}% 30D Target",
    delta_color="normal"
)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: MULTI-HORIZON MASTER TABLE
# ==============================================================================
st.subheader(f"📊 {len(tickers)} Stocks: Tomorrow, Weekly & Monthly Forecast Table")

display_df = summary_df.copy()
display_df["Company"] = display_df["Ticker"].map(lambda x: COMPANY_NAMES.get(x, DEFAULT_STOCKS.get(x, f"{x} Equity")))

if has_custom_shares:
    cols_order = [
        "Ticker", "Company", "Shares_Owned", "Current_Price", "Position_Value", "Today_Change(%)",
        "Hist_Week_Change(%)", "Hist_Month_Change(%)",
        "Tomorrow_Forecast($)", "Tomorrow_Expected_Change(%)", "Tomorrow_Signal",
        "Weekly_Forecast($)", "Weekly_Expected_Change(%)", "Weekly_Signal",
        "Monthly_Forecast($)", "Monthly_Expected_Change(%)", "Monthly_Signal"
    ]
else:
    cols_order = [
        "Ticker", "Company", "Current_Price", "Today_Change(%)",
        "Hist_Week_Change(%)", "Hist_Month_Change(%)",
        "Tomorrow_Forecast($)", "Tomorrow_Expected_Change(%)", "Tomorrow_Signal",
        "Weekly_Forecast($)", "Weekly_Expected_Change(%)", "Weekly_Signal",
        "Monthly_Forecast($)", "Monthly_Expected_Change(%)", "Monthly_Signal"
    ]

display_table = display_df[[c for c in cols_order if c in display_df.columns]].copy()

format_dict = {
    "Current_Price": "${:,.2f}",
    "Today_Change(%)": "{:+.2f}%",
    "Hist_Week_Change(%)": "{:+.2f}%",
    "Hist_Month_Change(%)": "{:+.2f}%",
    "Tomorrow_Forecast($)": "${:,.2f}",
    "Tomorrow_Expected_Change(%)": "{:+.2f}%",
    "Weekly_Forecast($)": "${:,.2f}",
    "Weekly_Expected_Change(%)": "{:+.2f}%",
    "Monthly_Forecast($)": "${:,.2f}",
    "Monthly_Expected_Change(%)": "{:+.2f}%"
}
if has_custom_shares:
    format_dict["Shares_Owned"] = "{:,.1f}"
    format_dict["Position_Value"] = "${:,.2f}"

st.dataframe(
    display_table.style
        .format(format_dict)
        .background_gradient(subset=["Tomorrow_Expected_Change(%)", "Weekly_Expected_Change(%)", "Monthly_Expected_Change(%)"], cmap="RdYlGn", vmin=-5, vmax=5),
    use_container_width=True,
    height=380
)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 3: COMPARATIVE VISUALIZATIONS
# ==============================================================================
col_c1, col_c2 = st.columns([2, 1])

with col_c1:
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=summary_df["Ticker"],
        y=summary_df["Tomorrow_Expected_Change(%)"],
        name="Tomorrow (1-Day)",
        marker_color="#2962FF"
    ))
    fig_comp.add_trace(go.Bar(
        x=summary_df["Ticker"],
        y=summary_df["Weekly_Expected_Change(%)"],
        name="Weekly (5-Day)",
        marker_color="#00E676"
    ))
    fig_comp.add_trace(go.Bar(
        x=summary_df["Ticker"],
        y=summary_df["Monthly_Expected_Change(%)"],
        name="Monthly (21-Day)",
        marker_color="#FF9100"
    ))
    fig_comp.update_layout(
        title=f"Projected Expected Return (%) Across {len(summary_df)} Stocks by Time Horizon",
        barmode="group",
        yaxis_title="Expected Return (%)",
        xaxis_title="Stock",
        template="plotly_dark",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with col_c2:
    # Sentiment distribution
    signals = summary_df["Tomorrow_Signal"].value_counts()
    colors_map = {
        "BUY / BULLISH": "#00E676",
        "SELL / BEARISH": "#FF5252",
        "HOLD / NEUTRAL": "#90A4AE"
    }
    pie_colors = [colors_map.get(s, "#78909C") for s in signals.index]
    fig_pie = go.Figure(data=[go.Pie(
        labels=signals.index,
        values=signals.values,
        hole=0.45,
        marker=dict(colors=pie_colors)
    )])
    fig_pie.update_layout(
        title="Tomorrow's Market Sentiment",
        template="plotly_dark",
        height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECTION 4: SINGLE-STOCK DEEP DIVE DRILL-DOWN
# ==============================================================================
st.subheader("🔍 Stock Deep Dive & Interactive Projections")
valid_tickers = [t for t in tickers if t in model_dict]
if not valid_tickers:
    st.error("No valid analyzed stocks available.")
    st.stop()

selected_t = st.selectbox(
    "Choose a stock to inspect detailed indicators & forecasts:",
    valid_tickers,
    format_func=lambda t: f"{t} — {COMPANY_NAMES.get(t, DEFAULT_STOCKS.get(t, 'Equity'))}"
)

active = model_dict[selected_t]
predictor = active["predictor"]
metrics = active["metrics"]
bt_df = active["bt_df"]
bt_metrics = active["bt_metrics"]
imp_df = active["imp_df"]
raw_df = active["raw_df"]
s_row = predictor.multi_horizon_summary

# Deep-dive metric cards
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Current Price", f"${s_row['Current_Price']:,.2f}", f"{s_row['Today_Change(%)']:+.2f}% Today")
d2.metric("Tomorrow Forecast", f"${s_row['Tomorrow_Forecast($)']:,.2f}", f"{s_row['Tomorrow_Expected_Change(%)']:+.2f}%")
d3.metric("Weekly Forecast", f"${s_row['Weekly_Forecast($)']:,.2f}", f"{s_row['Weekly_Expected_Change(%)']:+.2f}%")
d4.metric("Monthly Forecast", f"${s_row['Monthly_Forecast($)']:,.2f}", f"{s_row['Monthly_Expected_Change(%)']:+.2f}%")
d5.metric("Model Directional Acc", f"{metrics['Directional_Accuracy(%)']:.1f}%", f"MAPE: {metrics['MAPE(%)']:.2f}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Multi-Horizon Trajectory & Predictions",
    "📊 Candlestick & Technical Indicators",
    "🧠 Key Predictive Drivers",
    "💼 Strategy Backtest & Equity Curve"
])

# Tab 1: Trajectory Cone & Predictions
with tab1:
    st.markdown("#### Projected Price Trajectory Cone (Current ➔ Tomorrow ➔ 1-Week ➔ 1-Month)")
    
    # Construct projection points
    proj_dates = [
        "Current Close",
        "Tomorrow (1D)",
        "Weekly (5D)",
        "Monthly (21D)"
    ]
    proj_prices = [
        s_row["Current_Price"],
        s_row["Tomorrow_Forecast($)"],
        s_row["Weekly_Forecast($)"],
        s_row["Monthly_Forecast($)"]
    ]
    
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=proj_dates,
        y=proj_prices,
        mode="lines+markers+text",
        name="Projected Target",
        text=[f"${p:,.2f}" for p in proj_prices],
        textposition="top center",
        line=dict(color="#00E676", width=3),
        marker=dict(size=10, color="#FFD600")
    ))
    fig_proj.update_layout(
        title=f"{selected_t} Multi-Horizon Price Forecast Trajectory",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # Actual vs Predicted on Test Set
    pred_df = predictor.predictions_df
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["Actual_Next_Close"],
        mode="lines", name="Actual Close",
        line=dict(color="#2962FF", width=2)
    ))
    fig_test.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["Predicted_Next_Close"],
        mode="lines", name="Predicted Close",
        line=dict(color="#FF6D00", width=1.5, dash="dash")
    ))
    fig_test.update_layout(
        title=f"Test Set Historical Backtest ({metrics['Test_Samples']} Days) | Directional Accuracy: {metrics['Directional_Accuracy(%)']}%",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_dark",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_test, use_container_width=True)

# Tab 2: Technical Indicators
with tab2:
    featured_df = extract_features(raw_df)

    fig_tech = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Price & Bollinger Bands", "Relative Strength Index (RSI - 14)", "MACD (12, 26, 9)"),
        row_heights=[0.55, 0.22, 0.23]
    )

    # Candlestick
    fig_tech.add_trace(go.Candlestick(
        x=featured_df.index,
        open=featured_df["Open"], high=featured_df["High"],
        low=featured_df["Low"], close=featured_df["Close"],
        name="OHLC"
    ), row=1, col=1)

    # Bollinger Bands & SMA
    fig_tech.add_trace(go.Scatter(x=featured_df.index, y=featured_df["SMA_20"], name="SMA 20", line=dict(color="orange", width=1.2)), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=featured_df.index, y=featured_df["BB_Upper"], name="BB Upper", line=dict(color="rgba(100,200,255,0.4)", width=1)), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=featured_df.index, y=featured_df["BB_Lower"], name="BB Lower", line=dict(color="rgba(100,200,255,0.4)", width=1), fill="tonexty", fillcolor="rgba(100,200,255,0.05)"), row=1, col=1)

    # RSI
    fig_tech.add_trace(go.Scatter(x=featured_df.index, y=featured_df["RSI"], name="RSI", line=dict(color="#9c27b0", width=1.5)), row=2, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig_tech.add_trace(go.Scatter(x=featured_df.index, y=featured_df["MACD"], name="MACD", line=dict(color="#00bcd4", width=1.2)), row=3, col=1)
    fig_tech.add_trace(go.Scatter(x=featured_df.index, y=featured_df["MACD_Signal"], name="Signal", line=dict(color="#ff9800", width=1.2)), row=3, col=1)
    fig_tech.add_trace(go.Bar(x=featured_df.index, y=featured_df["MACD_Hist"], name="Histogram", marker_color="gray"), row=3, col=1)

    fig_tech.update_layout(height=750, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_tech, use_container_width=True)

# Tab 3: Feature Importance
with tab3:
    st.subheader(f"Key Predictive Drivers for {selected_t}")
    if not imp_df.empty:
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker=dict(color=imp_df["Importance"], colorscale="Viridis")
        ))
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            title="Top 10 Technical Indicators by Model Importance",
            xaxis_title="Feature Weight",
            template="plotly_dark",
            height=450
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

# Tab 4: Strategy Backtesting
with tab4:
    st.subheader(f"{selected_t} — ML Strategy vs. Buy & Hold Benchmark")
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["Strategy_Equity"],
        mode="lines", name="ML Model Strategy",
        line=dict(color="#00E676", width=2.2)
    ))
    fig_bt.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["BuyHold_Equity"],
        mode="lines", name="Buy & Hold Benchmark",
        line=dict(color="#B0BEC5", width=1.5, dash="dot")
    ))
    fig_bt.update_layout(
        title=f"Cumulative Portfolio Growth ($10,000 Base Capital)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Strategy Total Return", f"{bt_metrics['Strategy_Total_Return(%)']:+.2f}%")
    b2.metric("Buy & Hold Return", f"{bt_metrics['BuyHold_Total_Return(%)']:+.2f}%")
    b3.metric("Sharpe Ratio", f"{bt_metrics['Sharpe_Ratio']:.2f}")
    b4.metric("Strategy Win Rate", f"{bt_metrics['Win_Rate(%)']:.2f}%")
