# main_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ta
from datetime import datetime, timedelta
from gnews import GNews
from inference import predict_sentiment
from tensorflow.keras.models import load_model
import plotly.express as px

# Import utils (must be in same folder)
from utils import feature_engineering, prepare_test_df, simulate_strategy, FEATURE_COLS, SEQ_LEN

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="AI Stock & News Dashboard", layout="wide")
st.title("🤖 AI Stock & News Intelligence Dashboard")

# Sidebar navigation
page = st.sidebar.radio("📄 Select Page", ["News & Sentiment Report", "Stock Market Prediction"])

# ====================================================
# PAGE 1 — NEWS SENTIMENT REPORT (with date filtering)
# ====================================================
if page == "News & Sentiment Report":
    st.header("📰 Company News Sentiment Analysis (Date-Filtered)")

    stocks = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank"
    }

    choice = st.sidebar.selectbox("Select Stock Ticker", list(stocks.keys()))
    company_query = st.sidebar.text_input("Or enter company name / query", value=stocks[choice])

    # Date range selector
    default_end = datetime.today()
    default_start = default_end - timedelta(days=7)
    start_date = st.sidebar.date_input("Start Date", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=default_end)
    analyze_btn = st.sidebar.button("Analyze News in Range")

    st.info("📅 The app will fetch and analyze news within the selected date range for the chosen company.")

    if analyze_btn:
        with st.spinner("Fetching news articles..."):
            google_news = GNews(language='en', country='IN', max_results=100)
            google_news.start_date = (start_date.year, start_date.month, start_date.day)
            google_news.end_date = (end_date.year, end_date.month, end_date.day)
            articles = google_news.get_news(company_query)

        if not articles:
            st.warning("⚠️ No recent news found for this company in the given date range.")
        else:
            rows = []
            for a in articles:
                title = a.get('title', '')
                desc = a.get('description', '') or ''
                pub_str = a.get('published date') or a.get('published') or ''
                try:
                    pub_date = datetime.strptime(pub_str[:10], "%Y-%m-%d").date()
                except:
                    pub_date = datetime.today().date()
                text_for_pred = desc if len(desc.strip()) > 10 else title
                rows.append({
                    "title": title,
                    "description": desc,
                    "published": pub_date,
                    "source": a.get('publisher', {}).get('title', 'Unknown'),
                    "url": a.get('url', ''),
                    "text_for_pred": text_for_pred
                })

            if not rows:
                st.warning("⚠️ No valid news within this date range.")
            else:
                texts = [r["text_for_pred"] for r in rows]
                with st.spinner("Analyzing sentiment using BERT..."):
                    results = predict_sentiment(texts)

                df_rows = []
                for r, row in zip(results, rows):
                    df_rows.append({
                        "Title": row["title"],
                        "Summary": (row["description"][:240] + "...") if len(row["description"]) > 240 else row["description"],
                        "Sentiment": r["label"],
                        "Confidence": round(r["score"], 3),
                        "Date": row["published"],
                        "Source": row["source"],
                        "URL": row["url"]
                    })

                df = pd.DataFrame(df_rows)
                label_to_score = {"positive": 1, "neutral": 0, "negative": -1}
                avg_score = df['Sentiment'].map(label_to_score).mean()

                st.metric("🧠 Average Sentiment", f"{avg_score:.2f}")
                st.write("### 📊 Sentiment Summary Report")

                # Charts
                fig_pie = px.pie(df, names='Sentiment', title='Sentiment Distribution')
                st.plotly_chart(fig_pie, use_container_width=True)

                sentiment_counts = df['Sentiment'].value_counts().reset_index()
                fig_bar = px.bar(sentiment_counts, x='index', y='Sentiment', color='index', title="Sentiment Count")
                st.plotly_chart(fig_bar, use_container_width=True)

                # Article cards
                st.write("### 📰 Articles within Date Range")
                for _, row in df.iterrows():
                    emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(row["Sentiment"], "⚪")
                    st.markdown(f"""
                    <div style="background-color:#1E1E1E;padding:15px;border-radius:10px;margin-bottom:15px;">
                        <h4 style="color:#58A6FF;margin-bottom:4px;">
                            <a href="{row['URL']}" target="_blank" style="text-decoration:none;color:#58A6FF;">{row['Title']}</a>
                        </h4>
                        <p style="font-size:13px;color:#aaa;margin-top:-5px;">
                            🗓️ {row['Date']} &nbsp; | &nbsp; 🏢 {row['Source']}
                        </p>
                        <p style="color:#ddd;">{row['Summary']}</p>
                        <p style="font-size:15px;">{emoji} <b>{row['Sentiment'].capitalize()}</b> 
                        (Confidence: {row['Confidence']})</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.success("✅ News Sentiment Analysis Completed")

                # --- Optional Financial Report ---
                st.subheader("📈 Generate Financial LSTM Report (for same company)")
                gen_fin = st.button("Run Stock Market Prediction for same period")
                if gen_fin:
                    st.session_state["company"] = company_query
                    st.session_state["start_date"] = start_date
                    st.session_state["end_date"] = end_date
                    st.experimental_rerun()

# ====================================================
# PAGE 2 — LSTM PREDICTION DASHBOARD
# ====================================================
elif page == "Stock Market Prediction":
    st.header("📈 Multi-Stock LSTM Prediction Dashboard")

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

    df_raw, df_feat = load_data()
    try:
        model, scaler, le = load_model_and_preproc()
    except Exception as e:
        st.sidebar.error(f"Unable to load model: {e}")
        st.stop()

    tickers = np.sort(df_feat['Ticker'].unique())
    selected_ticker = st.sidebar.selectbox("Select Ticker", options=tickers)

    min_date = df_feat[df_feat['Ticker']==selected_ticker]['Date'].min().date()
    max_date = df_feat[df_feat['Ticker']==selected_ticker]['Date'].max().date()
    default_start = st.session_state.get("start_date", min_date)
    default_end = st.session_state.get("end_date", max_date)

    start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=default_end, min_value=min_date, max_value=max_date)
    run_button = st.sidebar.button("Run Prediction & Backtest")

    if run_button:
        with st.spinner("Running model prediction and backtest..."):
            df_test = prepare_test_df(selected_ticker, df_feat, model, scaler, le, seq_len=SEQ_LEN)
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

                # Plot price chart
                fig_line = px.line(df_range, x='Date', y='Close', title=f"{selected_ticker} — Price Trend with Predictions")
                st.plotly_chart(fig_line, use_container_width=True)

                sim_df, stats = simulate_strategy(df_range)

                perf_table = {
                    'Metric': ['Total Return','CAGR','Annualized Return','Volatility','Sharpe Ratio','Max Drawdown'],
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

                # Strategy performance chart
                st.write("### 📈 Strategy vs Buy & Hold Performance")
                fig_perf = px.line(sim_df, x='Date', y=['Strategy Equity','Buy & Hold Equity'], title="Cumulative Performance")
                st.plotly_chart(fig_perf, use_container_width=True)

                st.success("✅ Financial Backtest Completed")

    st.markdown("---")
    st.caption("Notes: Model, scaler, label-encoder, and data must be in the working folder.")
