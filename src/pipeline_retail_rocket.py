"""
Step 1 — Retail Rocket Pipeline
Loads events.csv + item_properties, engineers behavioral features,
aggregates to item level, and saves retail_rocket_features.csv.
"""

import os
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "retail_rocket")
OUT_DIR = os.path.join(BASE_DIR, "data")


def load_events():
    """Load events.csv and parse timestamps."""
    print("[1/6] Loading events.csv ...")
    df = pd.read_csv(
        os.path.join(RAW_DIR, "events.csv"),
        dtype={"visitorid": str, "itemid": str, "event": str},
    )
    # timestamp is in milliseconds since epoch
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    # Drop transactionid column if present (not needed)
    if "transactionid" in df.columns:
        df.drop(columns=["transactionid"], inplace=True)
    print(f"   Events loaded: {len(df):,} rows")
    return df


def load_item_properties():
    """Load and combine item_properties_part1 and part2, extract price & categoryid."""
    print("[2/6] Loading item properties ...")
    parts = []
    for fname in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        p = pd.read_csv(
            os.path.join(RAW_DIR, fname),
            dtype={"itemid": str, "property": str, "value": str},
        )
        parts.append(p)
    props = pd.concat(parts, ignore_index=True)
    print(f"   Raw properties: {len(props):,} rows")

    # --- Extract categoryid ---
    cat_df = props[props["property"] == "categoryid"][["itemid", "value"]].copy()
    cat_df.rename(columns={"value": "categoryid"}, inplace=True)
    # Keep last (most recent) per item
    cat_df = cat_df.drop_duplicates(subset="itemid", keep="last")

    # --- Extract price ---
    # Price is stored under property "790" with values like "n15360.000"
    # Some values have multiple tokens; we take the first one.
    price_df = props[props["property"] == "790"][["itemid", "value"]].copy()
    price_df["value"] = price_df["value"].apply(_parse_price)
    price_df = price_df.dropna(subset=["value"])
    price_df.rename(columns={"value": "item_price"}, inplace=True)
    price_df = price_df.drop_duplicates(subset="itemid", keep="last")

    # Merge
    item_df = pd.merge(cat_df, price_df, on="itemid", how="outer")
    print(f"   Items with properties: {len(item_df):,}")
    return item_df


def _parse_price(val):
    """Parse price value like 'n15360.000' or 'n552.000 639502 n720.000 424566'."""
    if not isinstance(val, str):
        return np.nan
    # Take the first token starting with 'n'
    for token in val.split():
        if token.startswith("n"):
            try:
                return float(token[1:])
            except ValueError:
                continue
    return np.nan


def load_category_tree():
    """Load category_tree.csv."""
    print("   Loading category_tree.csv ...")
    tree = pd.read_csv(
        os.path.join(RAW_DIR, "category_tree.csv"),
        dtype={"categoryid": str, "parentid": str},
    )
    return tree


def merge_events_properties(events, item_props):
    """Merge events with item properties on itemid."""
    print("[3/6] Merging events with item properties ...")
    merged = pd.merge(events, item_props, on="itemid", how="left")
    print(f"   Merged rows: {len(merged):,}")
    return merged


def engineer_features(df):
    """Engineer time features and demand metrics."""
    print("[4/6] Engineering features ...")

    # Time features
    df["event_hour"] = df["timestamp"].dt.hour
    df["event_dayofweek"] = df["timestamp"].dt.dayofweek
    df["event_month"] = df["timestamp"].dt.month

    return df


def aggregate_to_item_level(df):
    """Aggregate to one row per itemid with all features."""
    print("[5/6] Aggregating to item level ...")

    # --- Demand score ---
    weight_map = {"view": 1, "addtocart": 3, "transaction": 5}
    df["event_weight"] = df["event"].map(weight_map).fillna(0)

    # Count events per item per type
    event_counts = df.groupby(["itemid", "event"]).size().unstack(fill_value=0)
    for col in ["view", "addtocart", "transaction"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    event_counts["demand_score"] = (
        event_counts["view"] * 1
        + event_counts["addtocart"] * 3
        + event_counts["transaction"] * 5
    )
    event_counts["conversion_rate"] = np.where(
        event_counts["view"] > 0,
        event_counts["transaction"] / event_counts["view"],
        0,
    )
    event_counts["view_to_cart_rate"] = np.where(
        event_counts["view"] > 0,
        event_counts["addtocart"] / event_counts["view"],
        0,
    )

    # --- Average time features per item ---
    time_agg = df.groupby("itemid").agg(
        event_hour=("event_hour", "mean"),
        event_dayofweek=("event_dayofweek", "mean"),
        event_month=("event_month", "mean"),
        item_price=("item_price", "first"),
        categoryid=("categoryid", "first"),
    )

    # Combine
    item_df = event_counts.join(time_agg, how="left")
    item_df = item_df.reset_index()

    # Keep only needed columns
    item_df = item_df[
        [
            "itemid",
            "item_price",
            "demand_score",
            "conversion_rate",
            "view_to_cart_rate",
            "event_hour",
            "event_dayofweek",
            "event_month",
            "categoryid",
        ]
    ]

    print(f"   Item-level rows: {len(item_df):,}")
    return item_df


def main():
    print("=" * 60)
    print("RETAIL ROCKET PIPELINE")
    print("=" * 60)

    events = load_events()
    item_props = load_item_properties()
    merged = merge_events_properties(events, item_props)
    merged = engineer_features(merged)
    item_df = aggregate_to_item_level(merged)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "retail_rocket_features.csv")
    item_df.to_csv(out_path, index=False)
    print(f"\n[6/6] Saved → {out_path}")
    print(f"   Shape: {item_df.shape}")
    print(f"   Columns: {list(item_df.columns)}")
    print("=" * 60)

    return item_df


if __name__ == "__main__":
    main()
