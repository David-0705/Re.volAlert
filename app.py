# main_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ta
from datetime import datetime
from gnews import GNews
from inference import predict_sentiment
from tensorflow.keras.models import load_model

# Import utils (must be in same folder)
from utils import feature_engineering, prepare_test_df, simulate_strategy, plot_performance, FEATURE_COLS, SEQ_LEN

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("🤖 AI-Powered Stock Dashboard")

# Sidebar navigation
page = st.sidebar.radio("📄 Select Page", ["News Sentiment Analysis", "Stock Market Prediction"])

# -----------------------------
# PAGE 1 — NEWS SENTIMENT ANALYSIS
# -----------------------------
if page == "News Sentiment Analysis":
    st.header("📰 Indian Stock News Sentiment (BERT Model - PyTorch)")
    st.info("Fetches latest Indian business news and analyzes sentiment using a fine-tuned BERT model.")

    stocks = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank"
    }
    choice = st.sidebar.selectbox("Select stock ticker", list(stocks.keys()))
    company_query = st.sidebar.text_input("Or enter company name / query", value=stocks[choice])
    num_articles = st.sidebar.slider("Number of articles", 3, 15, 8)
    analyze_btn = st.sidebar.button("Analyze News")

    st.info("⚠️ Make sure your model and tokenizer are saved in 'saved_model/'. Click 'Analyze News' to fetch and analyze recent news.")

    if analyze_btn:
        with st.spinner("Fetching latest news..."):
            google_news = GNews(language='en', country='IN', max_results=num_articles)
            articles = google_news.get_news(company_query)

        if not articles:
            st.warning("No recent news found for this query.")
        else:
            texts, rows = [], []
            for a in articles:
                title = a.get('title', '')
                desc = a.get('description', '') or ''
                published = a.get('date', 'N/A')
                source = a.get('publisher', {}).get('title', 'Unknown Source')
                url = a.get('url', '')
                text_for_pred = desc if len(desc.strip()) > 10 else title
                texts.append(text_for_pred)
                rows.append({
                    "title": title,
                    "description": desc,
                    "published": published,
                    "source": source,
                    "url": url
                })

            with st.spinner("Analyzing sentiment with BERT..."):
                results = predict_sentiment(texts)

            df_rows = []
            for r, row in zip(results, rows):
                df_rows.append({
                    "title": row["title"],
                    "summary": (row["description"][:240] + "...") if len(row["description"]) > 240 else row["description"],
                    "sentiment": r["label"],
                    "confidence": round(r["score"], 3),
                    "url": row["url"],
                    "published": row["published"],
                    "source": row["source"]
                })

            df = pd.DataFrame(df_rows)
            label_to_score = {"positive": 1, "neutral": 0, "negative": -1}
            avg_score = df['sentiment'].map(label_to_score).mean()

            st.metric("🧠 Average Sentiment (Signed)", round(avg_score, 3))
            st.write("### 📰 Recent News and Sentiments")

            for _, row in df.iterrows():
                sentiment_emoji = {
                    "positive": "🟢",
                    "neutral": "🟡",
                    "negative": "🔴"
                }.get(row["sentiment"], "⚪")

                st.markdown(f"""
                <div style="background-color:#1E1E1E;padding:15px;border-radius:10px;margin-bottom:15px;">
                    <h4 style="color:#58A6FF;margin-bottom:4px;">
                        <a href="{row['url']}" target="_blank" style="text-decoration:none;color:#58A6FF;">{row['title']}</a>
                    </h4>
                    <p style="font-size:13px;color:#aaa;margin-top:-5px;">
                        🗓️ {row['published']} &nbsp; | &nbsp; 🏢 {row['source']}
                    </p>
                    <p style="color:#ddd;">{row['summary']}</p>
                    <p style="font-size:15px;">{sentiment_emoji} <b>{row['sentiment'].capitalize()}</b> 
                    (Confidence: {row['confidence']})</p>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------
# PAGE 2 — STOCK MARKET PREDICTION
# -----------------------------
elif page == "Stock Market Prediction":
    st.header("📈 Multi-Stock LSTM Prediction Dashboard")

    # Cached data loader
    @st.cache_data
    def load_data(csv_path='multi_stock_data.csv'):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        df_feat = feature_engineering(df.copy())
        return df, df_feat

    @st.cache_resource
    def load_model_and_preproc(model_path='multi_stock_lstm_model.keras', scaler_path='scaler.pkl', le_path='le.pkl'):
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        return model, scaler, le

    def compute_indicators_for_plot(df_plot):
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

    # Load data and model
    df_raw, df_feat = load_data()
    try:
        model, scaler, le = load_model_and_preproc()
    except Exception as e:
        st.sidebar.error(f"Unable to load model: {e}")
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

    if run_button:
        with st.spinner("Generating predictions and running backtest..."):
            try:
                df_test = prepare_test_df(selected_ticker, df_feat, model, scaler, le, seq_len=SEQ_LEN)
            except Exception as e:
                st.error(f"Failed to prepare predictions: {e}")
                st.stop()

            df_test['Date'] = pd.to_datetime(df_test['Date'])
            mask = (df_test['Date'] >= pd.to_datetime(start_date)) & (df_test['Date'] <= pd.to_datetime(end_date))
            df_range = df_test.loc[mask].reset_index(drop=True)
            if df_range.empty:
                st.warning("No predicted rows in the selected date range.")
            else:
                last_row = df_range.iloc[-1]
                pred_dir = "UP" if int(last_row['pred'])==1 else "DOWN"
                pred_ret = float(last_row['next_ret']) * 100
                col1, col2 = st.columns([1,2])
                col1.metric("Last Prediction", pred_dir)
                col1.metric("Expected next-day return", f"{pred_ret:.2f}%")

                with st.expander("Show prediction table (last 50 rows)"):
                    st.dataframe(df_range.tail(50))

                df_plot = df_feat[df_feat['Ticker']==selected_ticker].copy()
                df_plot = df_plot[(df_plot['Date']>=pd.to_datetime(start_date)) & (df_plot['Date']<=pd.to_datetime(end_date))]
                if not df_plot.empty:
                    df_plot_ind = compute_indicators_for_plot(df_plot)
                    fig1 = plot_price_indicators(df_plot_ind, selected_ticker)
                    st.pyplot(fig1)

                sim_df, stats = simulate_strategy(df_range)
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

                fig2 = plot_strategy_charts(sim_df, stats)
                st.pyplot(fig2)

                st.success("Done ✅")

    st.markdown("---")
    st.caption("Notes: Model, scaler and label-encoder must be present in the working folder as: "
                "`multi_stock_lstm_model.keras`, `scaler.pkl`, `le.pkl`, and `multi_stock_data.csv`.")
