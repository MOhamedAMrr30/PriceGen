"""
Steps 5-7 runner — SHAP + Recommender + Report
Bypasses cv2/numpy compatibility issue by mocking cv2 before SHAP import.
Reuses saved best model from Step 4.
"""

import os
import sys
import types
import warnings
import pickle
import gc

import numpy as np
import pandas as pd

# Mock cv2 to bypass NumPy 2.x / OpenCV incompatibility
cv2_mock = types.ModuleType("cv2")
cv2_mock.__version__ = "4.0.0"
sys.modules["cv2"] = cv2_mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "model")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("muted", 8)


def log(msg):
    print(msg, flush=True)


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_artifacts():
    """Load model, data, and metadata."""
    log("[Load] Loading model and data ...")

    with open(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"), "rb") as f:
        best_model = f.read()
    # Need to reload properly with xgboost available
    import io
    best_model = pickle.loads(best_model) if isinstance(best_model, bytes) else best_model

    with open(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"), "rb") as f:
        best_model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "model_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    df = pd.read_csv(os.path.join(DATA_DIR, "model_ready_data.csv"),
                      dtype={"itemid": str, "categoryid": str})

    log(f"   Model loaded. Metadata keys: {list(metadata.keys())}")
    log(f"   Data: {df.shape}")
    return best_model, metadata, df


def prepare_data(df):
    """Prepare X, y for modeling."""
    exclude_cols = ["itemid", "categoryid", "category_bucket", "item_price_usd"]
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, int, float]]

    clean = df[feature_cols + ["item_price_usd"]].dropna()
    X = clean[feature_cols]
    y = clean["item_price_usd"]
    return X, y, feature_cols


# ═══════════════════════════════════════════════════════════════
# STEP 5 — SHAP Explainability
# ═══════════════════════════════════════════════════════════════

def shap_analysis(best_model, X_test):
    """Compute SHAP values and generate plots."""
    log("\n" + "=" * 60)
    log("STEP 5: SHAP Explainability")
    log("=" * 60)

    log("   Computing SHAP values ...")
    if len(X_test) > 5000:
        X_sample = X_test.sample(5000, random_state=42)
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)

    # Top 5 features by mean |SHAP|
    mean_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.Series(mean_shap, index=X_sample.columns).sort_values(ascending=False)
    log("\n   Top 5 SHAP Features:")
    for feat, val in feat_importance.head(5).items():
        log(f"     {feat}: {val:.4f}")

    # 1. Summary plot (beeswarm)
    log("\n   Generating SHAP plots ...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("SHAP Summary Plot (Beeswarm)", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close("all")
    log("   -> shap_summary_plot.png")

    # 2. Bar plot (mean |SHAP|)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_plot.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close("all")
    log("   -> shap_bar_plot.png")

    # 3. Waterfall for single prediction
    plt.figure(figsize=(12, 8))
    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_sample.iloc[0].values,
        feature_names=list(X_sample.columns),
    )
    shap.waterfall_plot(shap_explanation, show=False, max_display=12)
    plt.title("SHAP Waterfall - Sample Prediction", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_waterfall_sample.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close("all")
    log("   -> shap_waterfall_sample.png")

    gc.collect()
    return feat_importance, explainer


# ═══════════════════════════════════════════════════════════════
# STEP 6 — Pricing Recommender Test
# ═══════════════════════════════════════════════════════════════

def _make_item_features(df, columns, demand_tier, category, demand_score_pct, price_position):
    """Create a synthetic item feature vector."""
    cat_data = df[df["category_bucket"] == category]
    if len(cat_data) == 0:
        cat_data = df

    row = {}
    for col in columns:
        if col == "demand_tier":
            row[col] = demand_tier
        elif col == "price_position":
            row[col] = price_position
        elif col == "is_peak_hour":
            row[col] = 1 if demand_tier == 2 else 0
        elif col == "is_peak_day":
            row[col] = 1 if demand_tier >= 1 else 0
        elif col.startswith("cat_"):
            cat_name = col.replace("cat_", "")
            row[col] = 1 if cat_name == category else 0
        elif col == "log_demand_score":
            row[col] = np.log1p(cat_data["demand_score"].quantile(demand_score_pct))
        elif col == "log_competitor_price":
            row[col] = cat_data["log_competitor_price"].median() if "log_competitor_price" in cat_data.columns else 0
        elif col in cat_data.columns:
            row[col] = cat_data[col].quantile(demand_score_pct)
        else:
            row[col] = 0
    return row


def test_recommender(best_model, df, feature_cols, explainer, r2_score_val):
    """Test pricing recommendations for 3 example items."""
    log("\n" + "=" * 60)
    log("STEP 6: Pricing Recommender Test")
    log("=" * 60)

    test_items = [
        {
            "label": "High Demand Electronics Item",
            "features": _make_item_features(
                df, feature_cols, demand_tier=2, category="Electronics",
                demand_score_pct=0.9, price_position=1.2
            ),
        },
        {
            "label": "Low Demand Fashion Item",
            "features": _make_item_features(
                df, feature_cols, demand_tier=0, category="Fashion",
                demand_score_pct=0.1, price_position=0.8
            ),
        },
        {
            "label": "Medium Demand Sports Item",
            "features": _make_item_features(
                df, feature_cols, demand_tier=1, category="Sports",
                demand_score_pct=0.5, price_position=1.0
            ),
        },
    ]

    recommendations = []
    for item in test_items:
        item_df = pd.DataFrame([item["features"]])[feature_cols]
        pred = best_model.predict(item_df)[0]
        sv = explainer.shap_values(item_df)[0]
        top_shap_idx = np.argsort(np.abs(sv))[::-1][:3]
        top_factors = [feature_cols[i] for i in top_shap_idx]

        pp_val = item_df["price_position"].values[0] if "price_position" in item_df.columns else 1.0
        if pp_val > 1.05:
            pos = "above"
        elif pp_val < 0.95:
            pos = "below"
        else:
            pos = "at"

        tier_map = {0: "Low", 1: "Medium", 2: "High"}
        dt = item_df["demand_tier"].values[0] if "demand_tier" in item_df.columns else 1
        tier_label = tier_map.get(int(dt), "Medium")

        result = {
            "recommended_price_usd": float(pred),
            "price_vs_market": pos,
            "demand_tier": tier_label,
            "confidence_score": r2_score_val,
            "top_factors": top_factors,
        }
        recommendations.append({"label": item["label"], "result": result})

        log(f"\n   {item['label']}:")
        log(f"     Recommended: ${pred:.2f} USD")
        log(f"     Market pos:  {pos}")
        log(f"     Demand tier: {tier_label}")
        log(f"     Top factors: {top_factors}")

    return recommendations


# ═══════════════════════════════════════════════════════════════
# STEP 7 — Model Report
# ═══════════════════════════════════════════════════════════════

def generate_report(baseline_m, xgb_m, tuned_m, shap_feats, recommendations):
    """Generate outputs/model/model_report.md."""
    log("\n" + "=" * 60)
    log("STEP 7: Model Performance Report")
    log("=" * 60)

    interpretations = {
        "price_position": "Direct market positioning signal — how price compares to competitors",
        "avg_competitor_price_usd": "Market reference price drives model's understanding of fair value",
        "log_competitor_price": "Log-transformed competitor price capturing diminishing marginal effects",
        "min_competitor_price_usd": "Lowest competitor price sets the floor for pricing decisions",
        "max_competitor_price_usd": "Highest competitor price indicates the ceiling of the market",
        "demand_score": "Raw demand signal — higher demand may justify higher prices",
        "log_demand_score": "Log demand captures non-linear demand-price relationships",
        "conversion_rate": "Purchase probability — key price elasticity signal",
        "view_to_cart_rate": "Intent signal — high add-to-cart indicates price acceptability",
        "price_elasticity_proxy": "Category-level price sensitivity measure",
        "avg_discount_percentage": "Market-wide discounting behavior in the category",
        "avg_popularity_score": "Category popularity drives competitive intensity",
        "event_hour": "Time-of-day demand pattern influences optimal pricing",
        "event_dayofweek": "Day-of-week seasonality affects purchase behavior",
        "event_month": "Monthly seasonality signal",
        "is_peak_hour": "Binary peak hour indicator (5 PM)",
        "is_peak_day": "Binary peak day indicator (Wednesday)",
        "demand_tier": "Demand segmentation bin (Low/Medium/High)",
    }

    lines = []
    lines.append("# Model Performance Report — Dynamic Pricing Engine\n")

    # Comparison table
    lines.append("## Model Comparison\n")
    lines.append("| Metric | Linear Regression | XGBoost (Initial) | XGBoost (Tuned) |")
    lines.append("|--------|-------------------|-------------------|-----------------|")
    lines.append(f"| MAE    | {baseline_m['mae']:.4f} | {xgb_m['mae']:.4f} | {tuned_m['mae']:.4f} |")
    lines.append(f"| RMSE   | {baseline_m['rmse']:.4f} | {xgb_m['rmse']:.4f} | {tuned_m['rmse']:.4f} |")
    lines.append(f"| R²     | {baseline_m['r2']:.4f} | {xgb_m['r2']:.4f} | {tuned_m['r2']:.4f} |")

    # Best model
    lines.append("\n## Best Model (Tuned XGBoost)\n")
    lines.append(f"- **MAE:** {tuned_m['mae']:.4f} USD")
    lines.append(f"- **RMSE:** {tuned_m['rmse']:.4f} USD")
    lines.append(f"- **R²:** {tuned_m['r2']:.4f}")
    if "best_params" in tuned_m:
        lines.append(f"- **Best Parameters:** `{tuned_m['best_params']}`")
    if "best_cv_score" in tuned_m:
        lines.append(f"- **Best CV Score (neg MAE):** {tuned_m['best_cv_score']:.4f}")

    # SHAP
    lines.append("\n## Top 5 SHAP Features — Business Interpretation\n")
    lines.append("| Rank | Feature | Mean |SHAP| | Interpretation |")
    lines.append("|------|---------|------------|----------------|")
    for i, (feat, val) in enumerate(shap_feats.head(5).items()):
        interp = interpretations.get(feat, "Engineered feature contributing to price prediction")
        lines.append(f"| {i+1} | {feat} | {val:.4f} | {interp} |")

    # Recommendations
    lines.append("\n## Sample Pricing Recommendations\n")
    for rec in recommendations:
        lines.append(f"### {rec['label']}")
        lines.append(f"- **Recommended Price:** ${rec['result']['recommended_price_usd']:.2f} USD")
        lines.append(f"- **Price vs Market:** {rec['result']['price_vs_market']}")
        lines.append(f"- **Demand Tier:** {rec['result']['demand_tier']}")
        lines.append(f"- **Confidence (R²):** {rec['result']['confidence_score']:.4f}")
        lines.append(f"- **Top Factors:** {', '.join(rec['result']['top_factors'])}")
        lines.append("")

    # Business Impact
    lines.append("## Business Impact\n")
    lines.append(
        "This XGBoost-based dynamic pricing model enables e-commerce platforms like noon or Jumia "
        "to move from static, manual pricing to data-driven, automated price optimization. "
        "By combining real-time demand signals (view/cart/transaction behavior) with competitor "
        "pricing intelligence, the model predicts optimal prices that balance revenue maximization "
        f"with conversion rate optimization. With an R² of {tuned_m['r2']:.4f} and MAE of "
        f"${tuned_m['mae']:.2f}, the model captures the key pricing dynamics and can serve as "
        "the core of an automated pricing pipeline — recommending price adjustments based on "
        "demand tier, market position, and time-based patterns, while SHAP explainability "
        "ensures pricing decisions are transparent and auditable for business stakeholders."
    )

    report_path = os.path.join(OUTPUT_DIR, "model_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"   Saved → {report_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    # Load
    best_model, metadata, df = load_artifacts()

    # Prepare data split (same seed as training)
    X, y, feature_cols = prepare_data(df)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log(f"   Test set: {len(X_test):,} samples")

    # Compute metrics for already-trained models
    # Load baseline and initial xgb if available, else recompute
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor

    log("\n   Recomputing baseline metrics ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    baseline_m = {
        "mae": mean_absolute_error(y_test, lr_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, lr_preds)),
        "r2": r2_score(y_test, lr_preds),
    }
    log(f"   Baseline: MAE={baseline_m['mae']:.4f}, R²={baseline_m['r2']:.4f}")

    log("   Recomputing initial XGBoost metrics ...")
    with open(os.path.join(MODEL_DIR, "xgboost_pricing_model.pkl"), "rb") as f:
        xgb_initial = pickle.load(f)
    xgb_preds = xgb_initial.predict(X_test)
    xgb_m = {
        "mae": mean_absolute_error(y_test, xgb_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, xgb_preds)),
        "r2": r2_score(y_test, xgb_preds),
    }
    log(f"   XGBoost initial: MAE={xgb_m['mae']:.4f}, R²={xgb_m['r2']:.4f}")

    best_preds = best_model.predict(X_test)
    tuned_m = {
        "mae": mean_absolute_error(y_test, best_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, best_preds)),
        "r2": r2_score(y_test, best_preds),
    }
    # Try to get best params from model
    try:
        tuned_m["best_params"] = best_model.get_params()
    except:
        tuned_m["best_params"] = {}
    log(f"   XGBoost tuned: MAE={tuned_m['mae']:.4f}, R²={tuned_m['r2']:.4f}")

    # Step 5: SHAP
    shap_feats, explainer = shap_analysis(best_model, X_test)

    # Step 6: Recommender test
    recommendations = test_recommender(best_model, df, feature_cols, explainer, tuned_m["r2"])

    # Step 7: Report
    generate_report(baseline_m, xgb_m, tuned_m, shap_feats, recommendations)

    log("\n" + "=" * 60)
    log("STEPS 5-7 COMPLETE!")
    log("=" * 60)


if __name__ == "__main__":
    main()
