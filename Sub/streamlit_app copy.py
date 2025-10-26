import streamlit as st
from inference import analyze_articles

st.set_page_config(page_title="Stock Market News Sentiment Analyzer", page_icon="📈", layout="wide")

st.title("📈 Stock Market News Sentiment Analyzer")

st.markdown("""
Analyze real-world financial news between any two dates to see the summarized
content and the detected sentiment.  
**Powered by BERT + BART.**
""")

# ---- Date Range Input ----
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("🗓️ Start Date")
with col2:
    end_date = st.date_input("🗓️ End Date")

# ---- Button ----
if st.button("🔍 Analyze News"):
    if not start_date or not end_date:
        st.warning("Please select both start and end dates.")
    else:
        with st.spinner("Fetching and analyzing news... Please wait ⏳"):
            results = analyze_articles(str(start_date), str(end_date))

        if not results:
            st.error("No articles found for this date range. Try expanding the range.")
        else:
            st.success(f"✅ Found {len(results)} relevant articles between {start_date} and {end_date}.")

            # ---- Display results ----
            for r in results:
                st.markdown(f"### 📰 [{r['title']}]({r['url']})")
                st.caption(f"🗞️ **{r['publisher']}** | 🕒 {r['published']}")
                st.write("**🧩 Summary:**", r["summary"])
                if r["sentiment"].lower() == "positive":
                    st.write("**💬 Sentiment:**", f":green[{r['sentiment']}]")
                elif r["sentiment"].lower() == "negative":
                    st.write("**💬 Sentiment:**", f":red[{r['sentiment']}]")
                else:
                    st.write("**💬 Sentiment:**", f":orange[{r['sentiment']}]")
                st.divider()
