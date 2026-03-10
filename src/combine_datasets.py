"""
Step 3 — Combine Datasets
Normalizes categories, computes competitor stats from Amazon,
joins onto Retail Rocket features, and saves master_pricing_data.csv.
"""

import os
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "retail_rocket")


# ─── Category Mapping ───────────────────────────────────────────────

AMAZON_CATEGORY_MAP = {
    # Electronics
    "appliances": "Electronics",
    "electronics": "Electronics",
    "computers & accessories": "Electronics",
    "computers": "Electronics",
    "accessories": "Electronics",
    "musical instruments": "Electronics",
    "office products": "Electronics",
    "car & motorbike": "Electronics",
    # Fashion
    "women's clothing": "Fashion",
    "men's clothing": "Fashion",
    "women's shoes": "Fashion",
    "men's shoes": "Fashion",
    "kids' fashion": "Fashion",
    "clothing & accessories": "Fashion",
    "handbags & clutches": "Fashion",
    "fashion": "Fashion",
    "watches": "Fashion",
    "jewellery": "Fashion",
    "luggage & bags": "Fashion",
    # Home
    "home & kitchen": "Home",
    "home improvement": "Home",
    "home, kitchen, pets": "Home",
    "furniture": "Home",
    "garden & outdoors": "Home",
    "kitchen & dining": "Home",
    "home & garden": "Home",
    "pet supplies": "Home",
    # Beauty
    "beauty": "Beauty",
    "health & personal care": "Beauty",
    "beauty & health": "Beauty",
    # Sports
    "sports, fitness & outdoors": "Sports",
    "sports & fitness": "Sports",
    "sports": "Sports",
    # Other
    "toys & games": "Other",
    "grocery & gourmet foods": "Other",
    "baby": "Other",
    "books": "Other",
    "music": "Other",
    "movies & tv shows": "Other",
    "video games": "Other",
    "industrial & scientific": "Other",
    "stores": "Other",
}


def map_amazon_category(cat):
    """Map an Amazon main_category to one of 6 buckets."""
    if pd.isna(cat):
        return "Other"
    return AMAZON_CATEGORY_MAP.get(cat.strip().lower(), "Other")


def map_retail_rocket_categories(rr_df):
    """
    Retail Rocket has only numeric categoryids with no text labels.
    We load category_tree.csv to get parent hierarchy and assign
    top-level parent IDs to buckets heuristically.

    Since we have no text mapping, we distribute categories across
    buckets based on parent depth (top-level parents → different buckets).
    """
    tree = pd.read_csv(
        os.path.join(RAW_DIR, "category_tree.csv"),
        dtype={"categoryid": str, "parentid": str},
    )
    tree = tree.replace("", np.nan)

    # Build parent lookup
    parent_map = dict(zip(tree["categoryid"], tree["parentid"]))

    # Find root for each category
    def find_root(cid):
        visited = set()
        while cid in parent_map and pd.notna(parent_map.get(cid)) and cid not in visited:
            visited.add(cid)
            cid = parent_map[cid]
        return cid

    # Get unique roots
    all_cats = tree["categoryid"].unique()
    roots = set()
    for c in all_cats:
        roots.add(find_root(c))
    roots = sorted(roots)

    # Distribute roots across 6 buckets
    buckets = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Other"]
    root_to_bucket = {}
    for i, root in enumerate(roots):
        root_to_bucket[root] = buckets[i % len(buckets)]

    # Map each categoryid → bucket via its root
    def cat_to_bucket(cid):
        if pd.isna(cid):
            return "Other"
        root = find_root(str(int(float(cid))) if isinstance(cid, float) else str(cid))
        return root_to_bucket.get(root, "Other")

    rr_df["category_bucket"] = rr_df["categoryid"].apply(cat_to_bucket)
    return rr_df


def compute_competitor_stats(amazon_df):
    """Compute per-category competitor pricing stats from Amazon data."""
    print("[2/5] Computing competitor stats from Amazon ...")
    stats = (
        amazon_df.groupby("category_bucket")
        .agg(
            avg_competitor_price=("discounted_price", "mean"),
            min_competitor_price=("discounted_price", "min"),
            max_competitor_price=("discounted_price", "max"),
            avg_discount_percentage=("discount_percentage", "mean"),
            avg_popularity_score=("popularity_score", "mean"),
        )
        .reset_index()
    )
    print(f"   Competitor stats by category:\n{stats.to_string(index=False)}")
    return stats


def remove_outliers_iqr(df, column="item_price"):
    """Remove outliers using IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    print(f"   Outlier removal ({column}): {before:,} → {len(df):,} "
          f"(removed {before - len(df):,}, bounds: [{lower:.1f}, {upper:.1f}])")
    return df


def main():
    print("=" * 60)
    print("COMBINE DATASETS")
    print("=" * 60)

    # Load features
    print("[1/5] Loading feature CSVs ...")
    rr_df = pd.read_csv(os.path.join(DATA_DIR, "retail_rocket_features.csv"),
                         dtype={"itemid": str, "categoryid": str})
    amazon_df = pd.read_csv(os.path.join(DATA_DIR, "amazon_features.csv"))
    print(f"   Retail Rocket items: {len(rr_df):,}")
    print(f"   Amazon products: {len(amazon_df):,}")

    # Map categories
    print("[2/5] Normalizing categories ...")
    amazon_df["category_bucket"] = amazon_df["category"].apply(map_amazon_category)
    rr_df = map_retail_rocket_categories(rr_df)

    # Competitor stats
    comp_stats = compute_competitor_stats(amazon_df)

    # Join
    print("[3/5] Joining competitor signals ...")
    master = pd.merge(rr_df, comp_stats, on="category_bucket", how="left")
    print(f"   Rows after join: {len(master):,}")

    # Compute price_position
    master["price_position"] = np.where(
        master["avg_competitor_price"] > 0,
        master["item_price"] / master["avg_competitor_price"],
        np.nan,
    )

    # Drop rows without price
    before = len(master)
    master = master.dropna(subset=["item_price"])
    print(f"   Rows with valid price: {len(master):,} (dropped {before - len(master):,})")

    # Remove outliers
    print("[4/5] Removing outliers ...")
    master = remove_outliers_iqr(master, "item_price")

    # Final column order
    final_cols = [
        "itemid",
        "item_price",
        "demand_score",
        "conversion_rate",
        "view_to_cart_rate",
        "event_hour",
        "event_dayofweek",
        "event_month",
        "categoryid",
        "category_bucket",
        "avg_competitor_price",
        "min_competitor_price",
        "max_competitor_price",
        "avg_discount_percentage",
        "avg_popularity_score",
        "price_position",
    ]
    master = master[[c for c in final_cols if c in master.columns]]

    # Save
    out_path = os.path.join(DATA_DIR, "master_pricing_data.csv")
    master.to_csv(out_path, index=False)
    print(f"\n[5/5] Saved → {out_path}")
    print(f"   Shape: {master.shape}")
    print(f"   Category distribution:\n{master['category_bucket'].value_counts().to_string()}")
    print("=" * 60)

    return master


if __name__ == "__main__":
    main()
