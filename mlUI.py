# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ta
from datetime import datetime

# Import from your utils (make sure utils.py is in same folder)
from utils import feature_engineering, prepare_test_df, simulate_strategy, plot_performance, FEATURE_COLS, SEQ_LEN

# For model loading
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Multi-Stock LSTM Dashboard", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_data(csv_path='multi_stock_data.csv'):
    df = pd.read_csv(csv_path)
    # ensure Date is datetime and sorted
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker','Date']).reset_index(drop=True)
    df_feat = feature_engineering(df.copy())  # use copy to avoid SettingWithCopyWarning
    return df, df_feat

@st.cache_resource
def load_model_and_preproc(model_path='multi_stock_lstm_model.keras', scaler_path='scaler.pkl', le_path='le.pkl'):
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        # fallback: user might have saved only weights -> we'll raise a helpful error
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    # Load scaler & labelencoder
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

def compute_indicators_for_plot(df_plot):
    # expects df_plot sorted and containing 'Adj Close' and 'Date'
    out = df_plot.copy()
    out['MA_5'] = out['Adj Close'].rolling(5).mean()
    out['MA_10'] = out['Adj Close'].rolling(10).mean()
    out['MA_20'] = out['Adj Close'].rolling(20).mean()
    out['RSI_14'] = ta.momentum.RSIIndicator(out['Adj Close'], window=14).rsi()
    macd = ta.trend.MACD(out['Adj Close'])
    out['MACD'] = macd.macd()
    out['MACD_Signal'] = macd.macd_signal()
    return out

def plot_price_indicators(df_plot, ticker):
    fig, axes = plt.subplots(3,1, figsize=(10,8), gridspec_kw={'height_ratios':[3,1,1]})
    ax0, ax1, ax2 = axes
    ax0.plot(df_plot['Date'], df_plot['Adj Close'], label='Adj Close')
    ax0.plot(df_plot['Date'], df_plot['MA_5'], '--', label='MA 5')
    ax0.plot(df_plot['Date'], df_plot['MA_10'], '--', label='MA 10')
    ax0.plot(df_plot['Date'], df_plot['MA_20'], '--', label='MA 20')
    ax0.set_title(f"{ticker} Price & Moving Averages")
    ax0.legend(loc='upper left')
    ax0.grid(True)

    ax1.plot(df_plot['Date'], df_plot['RSI_14'], label='RSI(14)')
    ax1.axhline(70, color='red', linestyle='--')
    ax1.axhline(30, color='green', linestyle='--')
    ax1.set_title("RSI (14)")
    ax1.grid(True)

    ax2.plot(df_plot['Date'], df_plot['MACD'], label='MACD')
    ax2.plot(df_plot['Date'], df_plot['MACD_Signal'], '--', label='Signal')
    ax2.set_title("MACD")
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    return fig

def plot_strategy_charts(sim_df, stats):
    # Equity + B&H, Drawdown, Histogram, Scatter pred_prob vs next_ret
    fig, axes = plt.subplots(2,2, figsize=(12,8), gridspec_kw={'height_ratios':[2,1]})
    ax1 = axes[0,0]; ax2 = axes[0,1]; ax3 = axes[1,0]; ax4 = axes[1,1]

    ax1.plot(sim_df.index, sim_df['strategy_equity'], label='Strategy Equity', linewidth=2)
    ax1.plot(sim_df.index, sim_df['bh_equity'], label='Buy & Hold Equity', linewidth=2)
    ax1.set_title("Equity Curve")
    ax1.legend()
    ax1.grid(True)

    running_max = np.maximum.accumulate(sim_df['strategy_equity'].values)
    drawdown = (sim_df['strategy_equity'].values - running_max) / running_max
    ax2.plot(sim_df.index, drawdown, color='red')
    ax2.fill_between(sim_df.index, drawdown, 0, color='red', alpha=0.2)
    ax2.set_title(f"Drawdown (max dd = {stats['strategy']['max_drawdown']:.2%})")
    ax2.grid(True)

    ax3.hist(sim_df['strategy_daily_ret'].dropna(), bins=40)
    ax3.set_title("Strategy daily returns distribution")

    if 'pred_prob' in sim_df.columns:
        ax4.scatter(sim_df['pred_prob'], sim_df['next_ret'], alpha=0.6, s=20)
        ax4.set_xlabel("Predicted probability")
        ax4.set_ylabel("Actual next-day return")
        ax4.set_title("Predicted prob vs Actual next-day return")
    else:
        ax4.text(0.1,0.5,"No 'pred_prob' column present", fontsize=12)
        ax4.axis('off')

    plt.tight_layout()
    return fig

# -------------------------
# App: layout & logic
# -------------------------
st.title("📊 Multi-Stock LSTM Prediction Dashboard")

st.sidebar.header("Inputs & Model")
st.sidebar.markdown("Load model and choose a ticker + date range for prediction & backtest.")

# load data & model (cached)
df_raw, df_feat = load_data()
try:
    model, scaler, le = load_model_and_preproc(model_path='multi_stock_lstm_model.keras',
                                              scaler_path='scaler.pkl',
                                              le_path='le.pkl')
except Exception as e:
    st.sidebar.error(f"Unable to load model or preprocessing objects: {e}")
    st.stop()

# Sidebar controls
tickers = np.sort(df_feat['Ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Ticker", options=tickers)
min_date = df_feat[df_feat['Ticker']==selected_ticker]['Date'].min().date()
max_date = df_feat[df_feat['Ticker']==selected_ticker]['Date'].max().date()

start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start date must be <= end date")
    st.stop()

run_button = st.sidebar.button("Run Prediction & Backtest")

# Main display
st.subheader(f"Ticker: {selected_ticker}")

if run_button:
    with st.spinner("Generating predictions and running backtest..."):
        # prepare df_test using util function
        try:
            df_test = prepare_test_df(selected_ticker, df_feat, model, scaler, le, seq_len=SEQ_LEN)
        except Exception as e:
            st.error(f"Failed to prepare predictions: {e}")
            st.stop()

        # Filter to date range
        df_test['Date'] = pd.to_datetime(df_test['Date'])
        mask = (df_test['Date'] >= pd.to_datetime(start_date)) & (df_test['Date'] <= pd.to_datetime(end_date))
        df_range = df_test.loc[mask].reset_index(drop=True)
        if df_range.empty:
            st.warning("No predicted rows in the selected date range (maybe date range too small).")
        else:
            # Show last prediction (direction + expected return)
            last_row = df_range.iloc[-1]
            pred_dir = "UP" if int(last_row['pred'])==1 else "DOWN"
            pred_ret = float(last_row['next_ret']) * 100
            col1, col2 = st.columns([1,2])
            col1.metric("Last Prediction", pred_dir)
            col1.metric("Expected next-day return", f"{pred_ret:.2f}%")
            # Data preview
            with st.expander("Show prediction table (last 50 rows)"):
                st.dataframe(df_range.tail(50))

            # Compute indicators for plotting price chart
            df_plot = df_feat[df_feat['Ticker']==selected_ticker].copy()
            df_plot = df_plot[(df_plot['Date']>=pd.to_datetime(start_date)) & (df_plot['Date']<=pd.to_datetime(end_date))]
            if df_plot.empty:
                st.warning("No feature rows for plotting in selected range.")
            else:
                df_plot_ind = compute_indicators_for_plot(df_plot)
                fig1 = plot_price_indicators(df_plot_ind, selected_ticker)
                st.pyplot(fig1)

            # Run strategy simulation on df_range
            try:
                sim_df, stats = simulate_strategy(df_range)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

            # Show performance tables
            perf_table = {
                'Metric': ['Total Return','CAGR','Annualized Return','Annualized Volatility','Sharpe Ratio','Max Drawdown'],
                'Strategy': [
                    f"{stats['strategy']['total_return']*100:.2f}%",
                    f"{stats['strategy']['CAGR']*100:.2f}%",
                    f"{stats['strategy']['annualized_return']*100:.2f}%",
                    f"{stats['strategy']['annualized_vol']*100:.2f}%",
                    f"{stats['strategy']['sharpe']:.2f}",
                    f"{stats['strategy']['max_drawdown']*100:.2f}%"
                ],
                'Buy & Hold': [
                    f"{stats['buy_and_hold']['total_return']*100:.2f}%",
                    f"{stats['buy_and_hold']['CAGR']*100:.2f}%",
                    f"{stats['buy_and_hold']['annualized_return']*100:.2f}%",
                    f"{stats['buy_and_hold']['annualized_vol']*100:.2f}%",
                    f"{stats['buy_and_hold']['sharpe']:.2f}",
                    f"{stats['buy_and_hold']['max_drawdown']*100:.2f}%"
                ]
            }
            st.subheader("Performance Metrics")
            st.table(pd.DataFrame(perf_table))

            trade_stats = {
                'Metric': ['Number of Trades','Win Rate','Avg Return/Trade'],
                'Value': [
                    stats['n_trades'],
                    f"{stats['win_rate']*100:.2f}%",
                    f"{stats['avg_ret_per_trade']*100:.2f}%"
                ]
            }
            st.subheader("Trade Stats")
            st.table(pd.DataFrame(trade_stats))

            # Strategy charts
            sim_df_indexed, _ = simulate_strategy(df_range)  # simulate returns with index set
            fig2 = plot_strategy_charts(sim_df_indexed, stats)
            st.pyplot(fig2)

            st.success("Done ✅")

st.markdown("---")
st.caption("Notes: Model, scaler and label-encoder must be present in the working folder as: "
            "`multi_stock_lstm_model.keras`, `scaler.pkl`, `le.pkl`, and `multi_stock_data.csv`.")
