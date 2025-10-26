# ui_multi_stock.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from multi_stock_lstm import (
    df_feat, SEQ_LEN, FEATURE_COLS, scaler, le, model,
    create_sequences_multi, predict_for_stock, predict_and_plot_stock,
    prepare_test_df, simulate_strategy_date_range, plot_performance_date_range,
    strategy_metrics_tables
)

st.set_page_config(page_title="Multi-Stock LSTM Dashboard", layout="wide")

st.title("📈 Multi-Stock LSTM Prediction & Strategy Dashboard")

# --- Stock selection ---
ticker = st.selectbox("Select Stock", df_feat['Ticker'].unique())

# --- Date range selection ---
min_date = df_feat['Date'].min()
max_date = df_feat['Date'].max()
start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

st.markdown("---")

# --- Predictions ---
st.subheader(f"Predictions for {ticker}")
df_test = prepare_test_df(ticker, model, scaler, le)

# Filter by date range
df_test_range = df_test[(df_test['Date'] >= pd.to_datetime(start_date)) &
                        (df_test['Date'] <= pd.to_datetime(end_date))].reset_index(drop=True)

if df_test_range.empty:
    st.warning("⚠️ Not enough data for this date range.")
else:
    last_pred_class = df_test_range['pred'].iloc[-1]
    last_pred_dir = "UP" if last_pred_class == 1 else "DOWN"
    last_pred_ret = df_test_range['next_ret'].iloc[-1] * 100

    st.metric("Next Day Prediction", last_pred_dir)
    st.metric("Expected Return (%)", f"{last_pred_ret:.2f}")

    # --- Strategy Simulation ---
    st.subheader("Strategy Simulation")
    sim_df, stats = simulate_strategy_date_range(df_test_range, start_date=start_date, end_date=end_date)

    st.markdown("### 📊 Performance Metrics")
    perf_table, trade_table = strategy_metrics_tables(stats)
    st.dataframe(perf_table)

    st.markdown("### 📋 Trade Stats")
    st.dataframe(trade_table)

    # --- Plots ---
    st.subheader("Equity Curve, Drawdowns & Returns")
    st.pyplot(plot_performance_date_range(sim_df, stats, start_date=start_date, end_date=end_date))

    st.subheader("Stock Price, MAs, RSI & MACD")
    predict_and_plot_stock(ticker, start_date=start_date, end_date=end_date)
