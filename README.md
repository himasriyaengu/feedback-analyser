# Product Feedback Analyser

NLP on real app-store reviews: scrape Google Play reviews, run **sentiment analysis**
(lexicon-based, with negation + intensifier handling) and **LDA topic modelling**
(scikit-learn) to surface themes, then rank **pain points** and auto-generate a
one-page product brief. Served through an interactive **Streamlit** dashboard.

Built as a portfolio project: the same "product thinking" as a PM case study,
but engineered end-to-end with real code and real data.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Scrape reviews (any Play Store app; find the id in its store URL)
python src/scrape_reviews.py --app com.duolingo --count 500 --out data/reviews.csv

# 2. Analyse: sentiment + themes + product brief
python src/analyze.py --in data/reviews.csv --topics 6 --outdir data

# 3. Dashboard
streamlit run src/app.py
```

## What it does

| Step | How |
|---|---|
| Scrape | `google-play-scraper`, newest reviews, resumable batches |
| Clean | lowercasing, URL/punctuation stripping |
| Sentiment | built-in lexicon scorer: negation ("not good") and intensifiers ("very bad") handled, VADER-style normalisation to [-1, 1] |
| Themes | `CountVectorizer` (unigrams + bigrams) + `LatentDirichletAllocation` |
| Pain points | themes ranked by mentions x % negative, with worst-review quotes |
| Brief | `data/product_brief.md` — sentiment overview, top 3 pain points, what to build first |
| Dashboard | KPIs, sentiment distribution, theme chart, review explorer, brief |

## Project structure

```
feedback-analyser/
├── src/
│   ├── scrape_reviews.py   # Play Store -> CSV
│   ├── analyze.py          # sentiment + LDA + brief
│   └── app.py              # Streamlit dashboard
├── data/
│   ├── reviews.csv             # raw scraped reviews
│   ├── reviews_analyzed.csv    # + sentiment label + theme
│   ├── theme_summary.csv       # per-theme stats
│   └── product_brief.md        # auto-generated brief
└── requirements.txt
```

## Resume bullet

> Built an NLP pipeline analysing 500+ Google Play reviews: lexicon-based
> sentiment analysis and LDA topic modelling (scikit-learn) to extract themes,
> rank product pain points and auto-generate a product brief; shipped an
> interactive Streamlit dashboard with Plotly visualisations.

## Ideas to extend

- Swap the lexicon scorer for a Hugging Face transformer
  (`distilbert-base-uncased-finetuned-sst-2-english`) — the hook is marked in `analyze.py`
- Try BERTopic instead of LDA for sharper themes
- Track themes over time: re-scrape monthly and plot sentiment drift per theme
- Deploy the dashboard on Streamlit Community Cloud / Hugging Face Spaces (free)

## Analyse low-rated reviews separately

```bash
python src/analyze.py --in data/reviews.csv --topics 6 --outdir data \
  --domain-stopwords "duolingo,app,application,language" \
  --max-rating 2 --tag lowrated
streamlit run src/app.py
```

Select **lowrated** in the dashboard sidebar. Tagged outputs preserve the all-review analysis. Ratings filter the input; lexicon sentiment is a separate heuristic and may differ from star ratings. Small datasets automatically use fewer topics.
