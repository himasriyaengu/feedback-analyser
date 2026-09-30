"""Streamlit dashboard for the Product Feedback Analyser.

Run from the repo root:
    streamlit run src/app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Product Feedback Analyser", layout="wide")
st.title("Product Feedback Analyser")
st.caption("NLP on real app-store reviews: sentiment, themes and pain points.")


datasets = [""] + sorted(
    p.stem.removeprefix("reviews_analyzed_")
    for p in (BASE / "data").glob("reviews_analyzed_*.csv")
    if (BASE / "data" / f"theme_summary_{p.stem.removeprefix('reviews_analyzed_')}.csv").exists()
    and (BASE / "data" / f"product_brief_{p.stem.removeprefix('reviews_analyzed_')}.md").exists()
)
tag = st.sidebar.selectbox("Analysis dataset", datasets, format_func=lambda x: x or "All reviews")
suffix = f"_{tag}" if tag else ""
st.caption(f"Selected analysis: {tag or 'All reviews'}")

@st.cache_data
def load(suffix, modified):
    df = pd.read_csv(BASE / "data" / f"reviews_analyzed{suffix}.csv")
    summary = pd.read_csv(BASE / "data" / f"theme_summary{suffix}.csv")
    brief = (BASE / "data" / f"product_brief{suffix}.md").read_text()
    return df, summary, brief


try:
    paths = [BASE / "data" / f"{name}{suffix}.{ext}" for name, ext in
             [("reviews_analyzed", "csv"), ("theme_summary", "csv"), ("product_brief", "md")]]
    df, summary, brief = load(suffix, tuple(p.stat().st_mtime_ns for p in paths))
except FileNotFoundError:
    st.error("No analysed data found. Run `python src/scrape_reviews.py` then "
             "`python src/analyze.py` first.")
    st.stop()

# ---- KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analysed", f"{len(df):,}")
c2.metric("% negative", f"{(df['sentiment_label'] == 'negative').mean() * 100:.1f}%")
c3.metric("Avg rating", f"{df['rating'].mean():.2f} / 5")
c4.metric("Themes found", f"{df['theme'].nunique()}")

# ---- Charts ----
left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    dist = df["sentiment_label"].value_counts().reset_index()
    dist.columns = ["sentiment", "count"]
    fig = px.bar(dist, x="sentiment", y="count", color="sentiment",
                 color_discrete_map={"positive": "#2ca02c", "neutral": "#7f7f7f",
                                     "negative": "#d62728"})
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Mentions per theme (colour = % negative)")
    fig2 = px.bar(summary, x="mentions", y="theme", orientation="h",
                  color="pct_negative", color_continuous_scale="Reds",
                  labels={"theme": "", "pct_negative": "% negative"})
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, width="stretch")

# ---- Theme table ----
st.subheader("Theme breakdown")
st.dataframe(
    summary.style.format({"avg_rating": "{:.1f}", "avg_sentiment": "{:.2f}",
                          "pct_negative": "{:.0f}%"}),
    width="stretch",
)

# ---- Review explorer ----
st.subheader("Explore reviews")
f1, f2 = st.columns(2)
theme_filter = f1.selectbox("Theme", ["All"] + sorted(df["theme"].unique()))
sent_filter = f2.selectbox("Sentiment", ["All", "negative", "neutral", "positive"])
view = df
if theme_filter != "All":
    view = view[view["theme"] == theme_filter]
if sent_filter != "All":
    view = view[view["sentiment_label"] == sent_filter]
st.dataframe(view[["rating", "sentiment_label", "theme", "text"]].head(100),
             width="stretch")

# ---- Product brief ----
with st.expander("Auto-generated product brief"):
    st.markdown(brief)
