"""
Steps 4-6 - EDA, Business Insights Report, Feature Importance
Generates 13 EDA plots, EDA_summary.md, and feature_importance_preview.png.
"""

import os
import sys
import warnings
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Force unbuffered output
def log(msg):
    print(msg, flush=True)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EDA_DIR = os.path.join(BASE_DIR, "outputs", "eda")

# Style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("muted", 8)
FIGSIZE = (10, 6)


def ensure_dirs():
    os.makedirs(EDA_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(EDA_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    log(f"   -> {name}")


# -----------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------

def load_events():
    """Load events with only needed columns."""
    log("[Load] events.csv ...")
    df = pd.read_csv(
        os.path.join(DATA_DIR, "raw", "retail_rocket", "events.csv"),
        usecols=["timestamp", "visitorid", "event", "itemid"],
        dtype={"visitorid": str, "itemid": str, "event": str},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    log(f"   Events: {len(df):,}")
    return df


def load_amazon_features():
    """Load pre-processed amazon features (not raw 190MB file)."""
    log("[Load] amazon_features.csv ...")
    df = pd.read_csv(os.path.join(DATA_DIR, "amazon_features.csv"))
    log(f"   Amazon features: {len(df):,}")
    return df


def load_master():
    log("[Load] master_pricing_data.csv ...")
    df = pd.read_csv(os.path.join(DATA_DIR, "master_pricing_data.csv"),
                      dtype={"itemid": str, "categoryid": str})
    log(f"   Master: {len(df):,}")
    return df


def load_rr_features():
    log("[Load] retail_rocket_features.csv ...")
    df = pd.read_csv(os.path.join(DATA_DIR, "retail_rocket_features.csv"),
                      dtype={"itemid": str, "categoryid": str})
    log(f"   RR features: {len(df):,}")
    return df


# -----------------------------------------------------------
# DEMAND ANALYSIS (5 plots)
# -----------------------------------------------------------

def plot_event_distribution(events):
    """1. Event distribution pie chart."""
    counts = events["event"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=COLORS[:3], startangle=140, explode=[0.03]*len(counts),
        textprops={"fontsize": 13},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title("Event Type Distribution", fontsize=16, fontweight="bold", pad=20)
    save(fig, "event_distribution.png")


def plot_demand_score_distribution(rr_features):
    """2. Demand score distribution histogram."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    data = rr_features["demand_score"].clip(upper=rr_features["demand_score"].quantile(0.99))
    ax.hist(data, bins=50, color=COLORS[0], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Demand Score", fontsize=13)
    ax.set_ylabel("Number of Items", fontsize=13)
    ax.set_title("Demand Score Distribution (per Item)", fontsize=16, fontweight="bold")
    ax.axvline(data.median(), color="red", linestyle="--", label=f"Median: {data.median():.0f}")
    ax.legend(fontsize=12)
    save(fig, "demand_score_distribution.png")


def plot_conversion_by_hour(events):
    """3. Conversion rate by hour of day - line chart."""
    events["hour"] = events["timestamp"].dt.hour
    hourly = events.groupby(["hour", "event"]).size().unstack(fill_value=0)
    for col in ["view", "transaction"]:
        if col not in hourly.columns:
            hourly[col] = 0
    hourly["conversion_rate"] = hourly["transaction"] / hourly["view"].replace(0, np.nan)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hourly.index, hourly["conversion_rate"], marker="o", color=COLORS[2],
            linewidth=2.5, markersize=7)
    ax.fill_between(hourly.index, hourly["conversion_rate"], alpha=0.15, color=COLORS[2])
    ax.set_xlabel("Hour of Day", fontsize=13)
    ax.set_ylabel("Conversion Rate", fontsize=13)
    ax.set_title("Conversion Rate by Hour of Day", fontsize=16, fontweight="bold")
    ax.set_xticks(range(24))
    save(fig, "conversion_by_hour.png")
    return hourly


def plot_conversion_by_dayofweek(events):
    """4. Conversion rate by day of week - bar chart."""
    events["dayofweek"] = events["timestamp"].dt.dayofweek
    daily = events.groupby(["dayofweek", "event"]).size().unstack(fill_value=0)
    for col in ["view", "transaction"]:
        if col not in daily.columns:
            daily[col] = 0
    daily["conversion_rate"] = daily["transaction"] / daily["view"].replace(0, np.nan)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(range(7), daily["conversion_rate"], color=COLORS[1], edgecolor="white")
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names, fontsize=12)
    ax.set_xlabel("Day of Week", fontsize=13)
    ax.set_ylabel("Conversion Rate", fontsize=13)
    ax.set_title("Conversion Rate by Day of Week", fontsize=16, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.4f}",
                ha="center", va="bottom", fontsize=10)
    save(fig, "conversion_by_dayofweek.png")
    return daily


def plot_top20_demand(rr_features):
    """5. Top 20 highest demand items - horizontal bar chart."""
    top20 = rr_features.nlargest(20, "demand_score")[["itemid", "demand_score"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["itemid"].astype(str), top20["demand_score"],
            color=COLORS[3], edgecolor="white")
    ax.set_xlabel("Demand Score", fontsize=13)
    ax.set_ylabel("Item ID", fontsize=13)
    ax.set_title("Top 20 Highest Demand Items", fontsize=16, fontweight="bold")
    ax.invert_yaxis()
    save(fig, "top20_demand_items.png")


# -----------------------------------------------------------
# PRICING ANALYSIS (4 plots)
# -----------------------------------------------------------

def plot_price_by_category(amazon):
    """6. Price distribution by category bucket - boxplot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_data = amazon[amazon["actual_price"] < amazon["actual_price"].quantile(0.95)].copy()
    order = plot_data.groupby("category")["actual_price"].median().sort_values(ascending=False).index
    sns.boxplot(data=plot_data, x="category", y="actual_price", order=order[:10],
                palette="muted", ax=ax)
    ax.set_xlabel("Category", fontsize=13)
    ax.set_ylabel("Actual Price", fontsize=13)
    ax.set_title("Price Distribution by Category", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    save(fig, "price_by_category.png")


def plot_discount_by_category(amazon):
    """7. Discount percentage by category - bar chart."""
    cat_disc = amazon.groupby("category")["discount_percentage"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(cat_disc[:10])), cat_disc.values[:10],
                  color=COLORS[4], edgecolor="white")
    ax.set_xticks(range(len(cat_disc[:10])))
    ax.set_xticklabels(cat_disc.index[:10], rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("Category", fontsize=13)
    ax.set_ylabel("Average Discount %", fontsize=13)
    ax.set_title("Average Discount Percentage by Category", fontsize=16, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=10)
    save(fig, "discount_by_category.png")


def plot_rating_vs_price(amazon):
    """8. Rating vs price - scatter plot colored by category."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_data = amazon.dropna(subset=["rating", "actual_price"])
    plot_data = plot_data[plot_data["actual_price"] < plot_data["actual_price"].quantile(0.95)]
    top_cats = plot_data["category"].value_counts().head(6).index
    plot_data = plot_data[plot_data["category"].isin(top_cats)]
    sns.scatterplot(data=plot_data, x="actual_price", y="rating",
                    hue="category", alpha=0.4, s=15, ax=ax)
    ax.set_xlabel("Actual Price", fontsize=13)
    ax.set_ylabel("Rating", fontsize=13)
    ax.set_title("Rating vs Price (by Category)", fontsize=16, fontweight="bold")
    ax.legend(title="Category", fontsize=9, title_fontsize=11, loc="lower right")
    save(fig, "rating_vs_price.png")


def plot_popularity_vs_price(amazon):
    """9. Popularity score vs price - scatter plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_data = amazon.dropna(subset=["popularity_score", "actual_price"])
    plot_data = plot_data[plot_data["actual_price"] < plot_data["actual_price"].quantile(0.95)]
    # Sample down for scatter performance
    if len(plot_data) > 50000:
        plot_data = plot_data.sample(50000, random_state=42)
    ax.scatter(plot_data["actual_price"], plot_data["popularity_score"],
               alpha=0.3, s=12, color=COLORS[5])
    ax.set_xlabel("Actual Price", fontsize=13)
    ax.set_ylabel("Popularity Score", fontsize=13)
    ax.set_title("Popularity Score vs Price", fontsize=16, fontweight="bold")
    save(fig, "popularity_vs_price.png")


# -----------------------------------------------------------
# COMBINED ANALYSIS (4 plots)
# -----------------------------------------------------------

def plot_price_position_distribution(master):
    """10. Price position distribution histogram."""
    data = master["price_position"].dropna()
    data = data[(data > 0) & (data < data.quantile(0.99))]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(data, bins=50, color=COLORS[0], edgecolor="white", alpha=0.85)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Market Price (1.0)")
    above = (data > 1.0).sum()
    below = (data <= 1.0).sum()
    ax.annotate(f"Above market: {above:,}", xy=(0.72, 0.9), xycoords="axes fraction",
                fontsize=12, color="darkred")
    ax.annotate(f"Below market: {below:,}", xy=(0.05, 0.9), xycoords="axes fraction",
                fontsize=12, color="darkgreen")
    ax.set_xlabel("Price Position (item_price / avg_competitor_price)", fontsize=12)
    ax.set_ylabel("Number of Items", fontsize=13)
    ax.set_title("Price Position Distribution", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    save(fig, "price_position_distribution.png")


def plot_demand_vs_price_position(master):
    """11. Demand score vs price position - KEY scatter plot."""
    data = master.dropna(subset=["price_position", "demand_score"])
    data = data[(data["price_position"] > 0) &
                (data["price_position"] < data["price_position"].quantile(0.99))]
    data = data[data["demand_score"] < data["demand_score"].quantile(0.99)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(data["price_position"], data["demand_score"],
               alpha=0.3, s=15, c=COLORS[2])
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Market Price", alpha=0.7)
    ax.set_xlabel("Price Position (1.0 = market price)", fontsize=13)
    ax.set_ylabel("Demand Score", fontsize=13)
    ax.set_title("Demand Score vs Price Position (KEY CHART)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    save(fig, "demand_vs_price_position.png")


def plot_conversion_vs_price_position(master):
    """12. Conversion rate vs price position - KEY scatter plot."""
    data = master.dropna(subset=["price_position", "conversion_rate"])
    data = data[(data["price_position"] > 0) &
                (data["price_position"] < data["price_position"].quantile(0.99))]
    data = data[data["conversion_rate"] > 0]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(data["price_position"], data["conversion_rate"],
               alpha=0.3, s=15, color=COLORS[4])
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Market Price", alpha=0.7)
    ax.set_xlabel("Price Position (1.0 = market price)", fontsize=13)
    ax.set_ylabel("Conversion Rate", fontsize=13)
    ax.set_title("Conversion Rate vs Price Position (KEY CHART)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    save(fig, "conversion_vs_price_position.png")


def plot_correlation_heatmap(master):
    """13. Correlation heatmap of all numerical features."""
    num_cols = master.select_dtypes(include=[np.number]).columns
    corr = master[num_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    save(fig, "correlation_heatmap.png")


# -----------------------------------------------------------
# STEP 5 - BUSINESS INSIGHTS REPORT
# -----------------------------------------------------------

def generate_eda_summary(events, amazon, master, rr_features):
    """Generate EDA_summary.md with business insights."""
    log("\n[Step 5] Generating EDA_summary.md ...")

    lines = []
    lines.append("# EDA Summary - Dynamic Pricing Engine\n")

    # Data Overview
    lines.append("## Data Overview\n")
    lines.append("### Retail Rocket")
    lines.append(f"- **Total events:** {len(events):,}")
    lines.append(f"- **Unique items:** {events['itemid'].nunique():,}")
    lines.append(f"- **Unique visitors:** {events['visitorid'].nunique():,}")
    event_dist = events["event"].value_counts()
    for ev, cnt in event_dist.items():
        lines.append(f"- **{ev} events:** {cnt:,} ({cnt/len(events)*100:.1f}%)")

    lines.append("\n### Amazon")
    lines.append(f"- **Total products:** {len(amazon):,}")
    lines.append(f"- **Categories:** {amazon['category'].nunique()}")
    price_min = amazon['actual_price'].min()
    price_max = amazon['actual_price'].max()
    lines.append(f"- **Price range:** {price_min:.0f} - {price_max:.0f}")

    lines.append("\n### Master Dataset")
    lines.append(f"- **Final shape:** {master.shape[0]:,} rows x {master.shape[1]} columns")
    lines.append(f"- **Category buckets:** {master['category_bucket'].nunique()}")
    buck_dist = master["category_bucket"].value_counts()
    for b, cnt in buck_dist.items():
        lines.append(f"  - {b}: {cnt:,} items ({cnt/len(master)*100:.1f}%)")

    # Demand Insights
    lines.append("\n---\n")
    lines.append("## Demand Insights\n")

    events_copy = events.copy()
    events_copy["hour"] = events_copy["timestamp"].dt.hour
    hourly = events_copy.groupby(["hour", "event"]).size().unstack(fill_value=0)
    if "transaction" in hourly.columns and "view" in hourly.columns:
        hourly["conv"] = hourly["transaction"] / hourly["view"].replace(0, np.nan)
        peak_hour = hourly["conv"].idxmax()
        lines.append(f"- **Peak hour for transactions:** {peak_hour}:00 "
                      f"(conversion rate: {hourly['conv'].max():.4f})")

    events_copy["dayofweek"] = events_copy["timestamp"].dt.dayofweek
    daily = events_copy.groupby(["dayofweek", "event"]).size().unstack(fill_value=0)
    if "transaction" in daily.columns and "view" in daily.columns:
        daily["conv"] = daily["transaction"] / daily["view"].replace(0, np.nan)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        peak_day = day_names[int(daily["conv"].idxmax())]
        lines.append(f"- **Peak day for transactions:** {peak_day} "
                      f"(conversion rate: {daily['conv'].max():.4f})")

    del events_copy
    gc.collect()

    cat_demand = master.groupby("category_bucket")["demand_score"].sum().sort_values(ascending=False)
    lines.append("- **Top 3 categories by demand score:**")
    for i, (cat, score) in enumerate(cat_demand.head(3).items()):
        lines.append(f"  {i+1}. {cat}: {score:,.0f}")

    # Pricing Insights
    lines.append("\n---\n")
    lines.append("## Pricing Insights\n")

    cat_price = master.groupby("category_bucket")["avg_competitor_price"].first().sort_values(ascending=False)
    lines.append(f"- **Highest avg competitor price:** {cat_price.index[0]} "
                  f"({cat_price.iloc[0]:,.0f})")

    cat_disc = master.groupby("category_bucket")["avg_discount_percentage"].first().sort_values(ascending=False)
    lines.append(f"- **Highest avg discount:** {cat_disc.index[0]} "
                  f"({cat_disc.iloc[0]:.1f}%)")

    valid_pp = master["price_position"].dropna()
    above = (valid_pp > 1.0).sum()
    below = (valid_pp < 1.0).sum()
    at_market = (valid_pp == 1.0).sum()
    total = len(valid_pp)
    lines.append(f"- **Items above market (price_position > 1.0):** "
                  f"{above:,} ({above/total*100:.1f}%)")
    lines.append(f"- **Items below market (price_position < 1.0):** "
                  f"{below:,} ({below/total*100:.1f}%)")
    lines.append(f"- **Items at market price:** "
                  f"{at_market:,} ({at_market/total*100:.1f}%)")

    # Key Pricing Opportunities
    lines.append("\n---\n")
    lines.append("## Key Pricing Opportunities\n")

    for cat in cat_demand.head(5).index:
        cat_data = master[master["category_bucket"] == cat]
        if len(cat_data) == 0:
            continue
        avg_pp = cat_data["price_position"].mean()
        avg_demand = cat_data["demand_score"].mean()
        avg_conv = cat_data["conversion_rate"].mean()

        if avg_pp < 1.0:
            diff_pct = (1.0 - avg_pp) * 100
            lines.append(
                f"- **{cat}** items show high demand (avg score: {avg_demand:,.0f}) "
                f"but priced {diff_pct:.0f}% below market — revenue uplift opportunity."
            )
        elif avg_pp > 1.0:
            diff_pct = (avg_pp - 1.0) * 100
            lines.append(
                f"- **{cat}** items priced {diff_pct:.0f}% above market with "
                f"conversion rate {avg_conv:.4f} — monitor for price elasticity."
            )

    if "transaction" in hourly.columns:
        low_hour = hourly["conv"].idxmin()
        lines.append(
            f"- Conversion drops significantly after {low_hour}:00 — "
            f"consider time-based discounting for off-peak hours."
        )

    avg_conv = master["conversion_rate"].mean()
    lines.append(
        f"- Overall conversion rate: {avg_conv:.4f} — items with above-average "
        f"conversion and below-market price are strongest optimization candidates."
    )

    report_path = os.path.join(EDA_DIR, "EDA_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"   -> EDA_summary.md")
    return report_path


# -----------------------------------------------------------
# STEP 6 - FEATURE IMPORTANCE
# -----------------------------------------------------------

def run_feature_importance(master):
    """Random Forest on master data to predict item_price."""
    log("\n[Step 6] Running feature importance ...")

    exclude = ["item_price", "itemid", "categoryid"]
    feature_cols = [c for c in master.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    df = master[feature_cols + ["item_price"]].dropna()
    X = df[feature_cols]
    y = df["item_price"]

    log(f"   Features: {feature_cols}")
    log(f"   Samples: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    train_r2 = rf.score(X_train, y_train)
    test_r2 = rf.score(X_test, y_test)
    log(f"   R2 - Train: {train_r2:.4f} | Test: {test_r2:.4f}")

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    log("\n   Top 5 Features:")
    for feat, imp in importances.head(5).items():
        log(f"     {feat}: {imp:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    imp_plot = importances.sort_values(ascending=True)
    ax.barh(imp_plot.index, imp_plot.values, color=COLORS[2], edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=13)
    ax.set_title("Random Forest - Feature Importance for Price Prediction",
                 fontsize=15, fontweight="bold")
    for i, (feat, val) in enumerate(imp_plot.items()):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=10)
    save(fig, "feature_importance_preview.png")

    return importances


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    ensure_dirs()

    log("=" * 60)
    log("EDA - DYNAMIC PRICING ENGINE")
    log("=" * 60)

    # Load data
    events = load_events()
    amazon = load_amazon_features()
    master = load_master()
    rr_features = load_rr_features()

    # Step 4: Plots
    log("\n[Step 4] Generating EDA plots ...\n")

    log("  >> Demand Analysis")
    plot_event_distribution(events)
    plot_demand_score_distribution(rr_features)
    hourly = plot_conversion_by_hour(events)
    daily = plot_conversion_by_dayofweek(events)
    plot_top20_demand(rr_features)

    log("\n  >> Pricing Analysis")
    plot_price_by_category(amazon)
    plot_discount_by_category(amazon)
    plot_rating_vs_price(amazon)
    plot_popularity_vs_price(amazon)

    log("\n  >> Combined Analysis")
    plot_price_position_distribution(master)
    plot_demand_vs_price_position(master)
    plot_conversion_vs_price_position(master)
    plot_correlation_heatmap(master)

    # Step 5: Report
    generate_eda_summary(events, amazon, master, rr_features)

    # Step 6: Feature Importance
    run_feature_importance(master)

    log("\n" + "=" * 60)
    log("EDA COMPLETE - All outputs saved to outputs/eda/")
    log("=" * 60)


if __name__ == "__main__":
    main()
