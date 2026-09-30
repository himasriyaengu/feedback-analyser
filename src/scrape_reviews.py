#!/usr/bin/env python3
"""Scrape Google Play Store reviews into a CSV.

Usage:
    python src/scrape_reviews.py --app com.duolingo --count 500 --out data/reviews.csv

Find an app's package id in its Play Store URL:
    https://play.google.com/store/apps/details?id=com.duolingo
"""
import argparse

import pandas as pd
from google_play_scraper import Sort, reviews


def scrape(app_id: str, count: int = 500, lang: str = "en", country: str = "us") -> pd.DataFrame:
    all_reviews = []
    token = None
    while len(all_reviews) < count:
        batch, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=min(200, count - len(all_reviews)),
            continuation_token=token,
        )
        if not batch:
            break
        all_reviews.extend(batch)
        if token is None:
            break

    rows = [
        {
            "review_id": r["reviewId"],
            "user": r["userName"],
            "rating": r["score"],
            "date": r["at"],
            "text": r["content"],
            "thumbs_up": r["thumbsUpCount"],
        }
        for r in all_reviews
    ]
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape Play Store reviews to CSV.")
    p.add_argument("--app", default="com.duolingo",
                   help="Google Play package id, e.g. com.duolingo")
    p.add_argument("--count", type=int, default=500, help="Max reviews to fetch")
    p.add_argument("--out", default="data/reviews.csv", help="Output CSV path")
    args = p.parse_args()

    df = scrape(args.app, args.count)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} reviews for {args.app} -> {args.out}")


if __name__ == "__main__":
    main()
