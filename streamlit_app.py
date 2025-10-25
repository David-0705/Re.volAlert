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
            published = a.get('published', '')
            text_for_pred = desc if len(desc.strip()) > 10 else title
            texts.append(text_for_pred)
            rows.append({
                "title": title,
                "description": desc,
                "published": published,
                "url": a.get('url', '')
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
                "published": row["published"]
            })

        df = pd.DataFrame(df_rows)
        label_to_score = {"positive": 1, "neutral": 0, "negative": -1}
        avg_score = df['sentiment'].map(label_to_score).mean()

        st.metric("🧠 Average Sentiment (Signed)", round(avg_score, 3))
        st.write("### 📰 Recent News and Sentiments")
        st.dataframe(df[["published", "title", "summary", "sentiment", "confidence"]], use_container_width=True)
