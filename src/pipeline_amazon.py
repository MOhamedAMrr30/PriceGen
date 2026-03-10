"""
Step 2 — Amazon Pipeline
Loads Amazon-Products.csv, cleans prices/ratings,
engineers pricing features, and saves amazon_features.csv.
"""

import os
import re
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "amazon")
OUT_DIR = os.path.join(BASE_DIR, "data")


def _clean_price(val):
    """Remove currency symbols (₹) and commas, convert to float."""
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    cleaned = re.sub(r"[₹â‚¹,\s]", "", val)
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _clean_rating(val):
    """Strip 'out of 5' text, convert to float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip()
    val = val.replace("out of 5", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def _clean_rating_count(val):
    """Remove commas, convert to int."""
    if pd.isna(val):
        return 0
    val = str(val).replace(",", "").strip()
    try:
        return int(float(val))
    except ValueError:
        return 0


def load_and_clean():
    """Load Amazon-Products.csv and clean all columns."""
    print("[1/4] Loading Amazon-Products.csv ...")
    df = pd.read_csv(os.path.join(RAW_DIR, "Amazon-Products.csv"), low_memory=False)
    print(f"   Raw rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    # Clean prices
    print("[2/4] Cleaning price and rating columns ...")
    df["actual_price"] = df["actual_price"].apply(_clean_price)
    df["discounted_price"] = df["discount_price"].apply(_clean_price)

    # Clean ratings
    df["rating"] = df["ratings"].apply(_clean_rating)
    df["rating_count"] = df["no_of_ratings"].apply(_clean_rating_count)

    # Drop rows where both prices are missing
    df = df.dropna(subset=["actual_price", "discounted_price"], how="all")
    # Fill missing discounted_price with actual_price
    df["discounted_price"] = df["discounted_price"].fillna(df["actual_price"])
    df["actual_price"] = df["actual_price"].fillna(df["discounted_price"])

    print(f"   Rows after price cleaning: {len(df):,}")
    return df


def engineer_features(df):
    """Engineer pricing and popularity features."""
    print("[3/4] Engineering features ...")

    # Discount percentage
    df["discount_percentage"] = np.where(
        df["actual_price"] > 0,
        (df["actual_price"] - df["discounted_price"]) / df["actual_price"] * 100,
        0,
    )
    # Clip negative discounts (data errors)
    df["discount_percentage"] = df["discount_percentage"].clip(lower=0)

    # Price competitiveness score
    df["price_competitiveness_score"] = np.where(
        df["actual_price"] > 0,
        df["discounted_price"] / df["actual_price"],
        1.0,
    )

    # Popularity score = rating * log(rating_count + 1)
    df["popularity_score"] = df["rating"].fillna(0) * np.log1p(
        df["rating_count"].fillna(0)
    )

    # Clean category
    df["category"] = df["main_category"].fillna("Other").str.strip().str.title()

    # Select output columns
    out_cols = [
        "name",
        "category",
        "sub_category",
        "actual_price",
        "discounted_price",
        "discount_percentage",
        "price_competitiveness_score",
        "rating",
        "rating_count",
        "popularity_score",
    ]
    df = df[out_cols].copy()

    return df


def main():
    print("=" * 60)
    print("AMAZON PIPELINE")
    print("=" * 60)

    df = load_and_clean()
    df = engineer_features(df)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "amazon_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[4/4] Saved → {out_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Categories: {df['category'].nunique()}")
    print(f"   Price range: {df['actual_price'].min():.0f} — {df['actual_price'].max():.0f}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
