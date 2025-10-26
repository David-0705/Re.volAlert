import os
import requests
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TextClassificationPipeline,
    pipeline
)
from datetime import datetime
from dotenv import load_dotenv

# ======================
# Load environment vars
# ======================
load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

if not GNEWS_API_KEY:
    raise ValueError("❌ GNEWS_API_KEY not found in .env. Please add it before running the app.")

# ======================
# Load models
# ======================
print("🔹 Loading BERT sentiment model...")
MODEL_DIR = "saved_model/pt_model"
TOKENIZER_DIR = "saved_model/tokenizer"

model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
sentiment_analyzer = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

print("🔹 Loading summarization model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ======================
# Utility functions
# ======================
def fetch_stock_news(start_date, end_date, max_articles=30):
    """Fetch stock market related news within date range."""
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": "stock market OR nifty OR sensex OR shares OR trading OR finance",
        "from": start_date,
        "to": end_date,
        "lang": "en",
        "max": max_articles,
        "token": GNEWS_API_KEY
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if "articles" not in data:
        print("⚠️ Unexpected GNews API response:", data)
        return []
    return data["articles"]

def summarize_text(text):
    """Summarize article text."""
    if not text or len(text.split()) < 40:
        return text
    summary = summarizer(text, max_length=80, min_length=25, do_sample=False)
    return summary[0]["summary_text"]

def format_date(date_str):
    """Convert ISO 8601 date to readable format."""
    if not date_str:
        return "Unknown date"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return date_str

def analyze_articles(start_date, end_date):
    """Fetch, summarize, and analyze sentiment of stock news in date range."""
    articles = fetch_stock_news(start_date, end_date)
    results = []

    for a in articles:
        title = a.get("title", "No title")
        desc = a.get("description", "")
        url = a.get("url", "#")
        publisher = a.get("source", {}).get("name", "Unknown source")
        published_raw = a.get("publishedAt") or a.get("published") or a.get("pubDate")
        published = format_date(published_raw)

        # Summarize and analyze sentiment
        summary = summarize_text(desc or title)
        sentiment = sentiment_analyzer(summary)[0]["label"]

        results.append({
            "title": title,
            "summary": summary,
            "sentiment": sentiment,
            "published": published,
            "publisher": publisher,
            "url": url
        })

    return results


# Local test
if __name__ == "__main__":
    print("Testing article analysis...")
    out = analyze_articles("2025-10-01", "2025-10-05")
    print(out[:2])
