# streamlit_app.py
import streamlit as st
from gnews import GNews
from inference import predict_sentiment
import pandas as pd

st.set_page_config(page_title="Indian Stock News Sentiment", layout="wide")
st.title("📊 Indian Stock News Sentiment (BERT Model - PyTorch)")

# Sidebar inputs
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

        # --- Custom display for each article ---
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
