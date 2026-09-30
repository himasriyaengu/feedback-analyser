#!/usr/bin/env python3
"""Analyse app reviews: sentiment + topic modelling -> theme summary + product brief.

Usage:
    python src/analyze.py --in data/reviews.csv --topics 6 --outdir data

Outputs (in --outdir):
    reviews_analyzed.csv  - every review with sentiment label and theme
    theme_summary.csv     - per-theme mentions, avg rating, % negative
    product_brief.md      - one-page brief: top pain points + what to build first

Sentiment uses a built-in lexicon scorer (no downloads needed). To upgrade it,
swap `lexicon_sentiment` for a Hugging Face pipeline, e.g.:
    from transformers import pipeline
    clf = pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english")
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Optional: give themes human-readable names after a first run, e.g.
# THEME_NAMES = {0: "Subscription / pricing", 1: "App crashes"}
THEME_NAMES: dict[int, str] = {}

# --- Compact sentiment lexicon (word -> weight) ---
POSITIVE = {
    "love": 2.0, "loved": 2.0, "amazing": 2.0, "awesome": 1.8, "excellent": 1.8,
    "fantastic": 1.8, "wonderful": 1.6, "perfect": 1.8, "best": 1.6, "great": 1.4,
    "brilliant": 1.6, "superb": 1.6, "outstanding": 1.6, "incredible": 1.6,
    "impressive": 1.4, "good": 1.0, "nice": 0.9, "easy": 1.0, "helpful": 1.2,
    "useful": 1.1, "fun": 1.2, "enjoyable": 1.3, "enjoy": 1.2, "enjoyed": 1.2,
    "beautiful": 1.2, "gorgeous": 1.2, "clean": 0.8, "simple": 0.8, "smooth": 1.0,
    "fast": 0.9, "quick": 0.8, "intuitive": 1.1, "powerful": 1.0, "reliable": 1.1,
    "solid": 0.9, "efficient": 1.0, "effective": 1.0, "recommend": 1.3,
    "favorite": 1.2, "favourite": 1.2, "happy": 1.2, "glad": 1.0, "pleased": 1.0,
    "satisfied": 1.1, "worth": 0.9, "convenient": 1.0, "addictive": 1.0,
    "motivating": 1.2, "educational": 1.0, "learned": 0.9, "improved": 0.9,
    "better": 0.8, "thanks": 0.8, "thank": 0.8, "flawless": 1.6, "seamless": 1.2,
}
NEGATIVE = {
    "hate": -2.0, "hated": -2.0, "terrible": -1.8, "awful": -1.8, "horrible": -1.8,
    "worst": -1.8, "useless": -1.6, "broken": -1.5, "crash": -1.5, "crashes": -1.5,
    "crashed": -1.5, "crashing": -1.5, "bug": -1.2, "bugs": -1.2, "buggy": -1.4,
    "glitch": -1.2, "glitches": -1.2, "glitchy": -1.3, "slow": -1.0, "lag": -1.1,
    "laggy": -1.2, "freeze": -1.3, "freezes": -1.3, "frozen": -1.3,
    "annoying": -1.3, "frustrating": -1.4, "frustrated": -1.3,
    "disappointed": -1.4, "disappointing": -1.4, "disappointment": -1.4,
    "waste": -1.4, "sucks": -1.5, "suck": -1.4, "stupid": -1.4, "dumb": -1.3,
    "ridiculous": -1.4, "pathetic": -1.5, "garbage": -1.5, "trash": -1.4,
    "scam": -1.8, "misleading": -1.3, "confusing": -1.1, "complicated": -1.0,
    "difficult": -0.9, "expensive": -1.1, "overpriced": -1.3, "pricey": -1.1,
    "bad": -1.0, "poor": -1.1, "error": -1.1, "errors": -1.1, "issue": -0.9,
    "issues": -0.9, "problem": -1.0, "problems": -1.0, "wrong": -1.0,
    "fail": -1.2, "failed": -1.2, "fails": -1.2, "missing": -1.0, "lost": -1.1,
    "drain": -1.0, "drains": -1.0, "battery": -0.4, "uninstall": -1.2,
    "uninstalled": -1.2, "delete": -0.8, "deleted": -0.9, "downgrade": -1.2,
    "worse": -1.0, "worsened": -1.1, "never": -0.6, "refund": -0.8,
}
LEXICON = {**POSITIVE, **NEGATIVE}
NEGATORS = {"not", "no", "never", "n't", "dont", "don't", "doesnt", "doesn't",
            "cant", "can't", "wont", "won't", "isnt", "isn't", "wasnt", "wasn't",
            "without", "hardly", "barely"}
INTENSIFIERS = {"very": 0.3, "really": 0.3, "extremely": 0.5, "super": 0.4,
                "so": 0.3, "too": 0.3, "quite": 0.2, "incredibly": 0.5}


def lexicon_sentiment(text: str) -> float:
    """Rule-based sentiment in [-1, 1]; handles negation and intensifiers."""
    words = text.split()
    score = 0.0
    for i, w in enumerate(words):
        base = LEXICON.get(w)
        if base is None:
            continue
        window = words[max(0, i - 2):i]
        if any(n in window for n in NEGATORS):
            base = -base * 0.8
        boost = sum(INTENSIFIERS.get(x, 0.0) for x in window)
        score += base * (1.0 + boost) if base > 0 else base * (1.0 + boost)
    # VADER-style normalisation into [-1, 1]
    return score / math.sqrt(score * score + 15.0)


def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse review sentiment and themes.")
    ap.add_argument("--in", dest="inp", default="data/reviews.csv")
    ap.add_argument("--topics", type=int, default=6, help="Number of LDA topics/themes")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--domain-stopwords", default="", help="Comma-separated words excluded from themes")
    ap.add_argument("--max-rating", type=int, choices=range(1, 6), help="Include ratings up to this value")
    ap.add_argument("--tag", default="", help="Output filename suffix, e.g. lowrated")
    args = ap.parse_args()
    if args.topics < 1:
        ap.error("--topics must be at least 1")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        ap.error("--tag may contain only letters, numbers, underscores and hyphens")
    suffix = f"_{args.tag}" if args.tag else ""

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.inp)
    if not {"text", "rating"}.issubset(df.columns):
        ap.error("Input CSV must contain text and rating columns")
    df = df.dropna(subset=["text"]).copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if args.max_rating is not None:
        df = df[df["rating"].between(1, args.max_rating)].copy()
    df["clean"] = df["text"].apply(clean)
    df = df[df["clean"].str.len() >= 4].reset_index(drop=True)
    if df.empty:
        ap.error("No usable reviews remain after filtering; no analysis files were written")
    print(f"Analysing {len(df)} reviews...")

    # ---- Sentiment ----
    df["sentiment"] = df["clean"].apply(lexicon_sentiment)
    df["sentiment_label"] = pd.cut(
        df["sentiment"], bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["negative", "neutral", "positive"],
    )

    # ---- Themes via LDA topic modelling ----
    stopwords = sorted(set(ENGLISH_STOP_WORDS) | {
        word for item in args.domain_stopwords.split(",") for word in clean(item).split()
    })
    vec = CountVectorizer(max_df=1.0, min_df=1 if len(df) < 20 else 3,
                          stop_words=stopwords, max_features=1500, ngram_range=(1, 2))
    try:
        X = vec.fit_transform(df["clean"])
    except ValueError:
        vec.set_params(min_df=1)
        try:
            X = vec.fit_transform(df["clean"])
        except ValueError:
            ap.error("No topic words remain; use fewer stopwords or more reviews")
    n_topics = max(1, min(args.topics, max(1, len(df) // 20), X.shape[1]))
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, learning_method="batch", max_iter=20
    )
    doc_topics = lda.fit_transform(X)
    df["theme_id"] = doc_topics.argmax(axis=1)

    vocab = np.array(vec.get_feature_names_out())
    theme_words = {
        i: [vocab[j] for j in comp.argsort()[-8:][::-1]]
        for i, comp in enumerate(lda.components_)
    }
    df["theme"] = df["theme_id"].apply(
        lambda i: THEME_NAMES.get(i, ", ".join(theme_words[i][:4]))
    )

    # ---- Per-theme summary ----
    summary = (
        df.groupby("theme")
        .agg(
            mentions=("theme", "size"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment", "mean"),
            pct_negative=("sentiment_label", lambda s: (s == "negative").mean() * 100),
        )
        .reset_index()
        .sort_values("mentions", ascending=False)
    )

    df.to_csv(outdir / f"reviews_analyzed{suffix}.csv", index=False)
    summary.to_csv(outdir / f"theme_summary{suffix}.csv", index=False)

    # ---- Product brief: top 3 pain points ----
    candidates = summary[(summary["mentions"] >= max(1, len(df) * 0.05)) & (summary["pct_negative"] > 0)]
    pain = candidates.sort_values("pct_negative", ascending=False).head(3)

    lines = ["# Product Brief (auto-generated)\n"]
    lines.append(f"Based on {len(df)} user reviews.\n")
    if args.max_rating is not None:
        lines.append(f"Scope: ratings 1–{args.max_rating}/5 only; this subset does not represent all reviews.\n")
    lines.append("Sentiment is a heuristic lexicon score; themes need human review.\n")
    lines.append("## Sentiment overview\n")
    dist = df["sentiment_label"].value_counts(normalize=True) * 100
    for label in ["positive", "neutral", "negative"]:
        lines.append(f"- {label.capitalize()}: {dist.get(label, 0):.1f}%")
    lines.append("\n## Top pain points\n")
    if pain.empty:
        lines.append("No qualifying themes with negative lexicon sentiment were found.")
    for rank, (_, row) in enumerate(pain.iterrows(), 1):
        lines.append(
            f"{rank}. **{row['theme']}** — {row['mentions']} mentions, "
            f"{row['pct_negative']:.0f}% negative, avg rating {row['avg_rating']:.1f}/5"
        )
        worst = df[(df["theme"] == row["theme"]) & (df["sentiment_label"] == "negative")]
        worst = worst.sort_values("sentiment").head(2)
        for _, w in worst.iterrows():
            quote = str(w["text"])[:220].replace("\n", " ")
            lines.append(f'   > "{quote}"')
    lines.append("\n## What to build first\n")
    lines.append(
        "Prioritise the themes above by (mentions x % negative): "
        "fix the highest-negativity, highest-volume theme first, "
        "then re-run this analysis after release to measure the shift."
    )
    (outdir / f"product_brief{suffix}.md").write_text("\n".join(lines))

    print("\nTheme summary:")
    print(summary.round(1).to_string(index=False))
    print(f"\nWrote: {outdir}/reviews_analyzed{suffix}.csv, theme_summary{suffix}.csv, product_brief{suffix}.md")


if __name__ == "__main__":
    main()
