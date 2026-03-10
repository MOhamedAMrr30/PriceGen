# pricing recommender - loads model from MLflow registry or local pkl

import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TRACKING_DIR = os.path.join(BASE_DIR, "mlflow_tracking")


def _load_model_from_registry():
    try:
        import mlflow
        import mlflow.xgboost

        tracking_uri = f"file:///{TRACKING_DIR.replace(os.sep, '/')}"
        mlflow.set_tracking_uri(tracking_uri)

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='pricing-model-production'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            model_uri = f"models:/pricing-model-production/{latest.version}"
            model = mlflow.xgboost.load_model(model_uri)
            print(f"[Recommender] Loaded from MLflow registry v{latest.version}", flush=True)
            return model
    except Exception as e:
        print(f"[Recommender] MLflow load failed: {e}", flush=True)
    return None


def _load_model_from_pkl():
    model_path = os.path.join(MODEL_DIR, "xgboost_best_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("[Recommender] Loaded from local pkl", flush=True)
    return model


def _load_metadata():
    meta_path = os.path.join(MODEL_DIR, "model_metadata.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def load_production_model():
    # try registry first, fallback to pkl
    model = _load_model_from_registry()
    if model is None:
        model = _load_model_from_pkl()
    metadata = _load_metadata()
    return model, metadata


ELASTICITY_MAP = {
    "Electronics": -0.015, "Fashion": -0.008, "Home": -0.012,
    "Beauty": -0.005, "Sports": -0.010, "Other": -0.007,
}


def recommend_price(item_features: dict) -> dict:
    model, metadata = load_production_model()
    feature_cols = metadata["feature_cols"]
    r2 = metadata["r2_score"]

    feature_values = [item_features.get(col, 0) for col in feature_cols]
    X = pd.DataFrame([feature_values], columns=feature_cols)

    predicted_price = max(float(model.predict(X)[0]), 0.01)

    # top factors weighted by input values
    importances = model.feature_importances_
    vals = X.iloc[0].values.astype(float)
    weighted = importances * np.abs(vals)
    top_idx = np.argsort(weighted)[::-1][:3]
    top_factors = [feature_cols[i] for i in top_idx]

    pp = item_features.get("price_position", 1.0)
    if pp > 1.05:
        price_vs_market = "above"
    elif pp < 0.95:
        price_vs_market = "below"
    else:
        price_vs_market = "at"

    tier_map = {0: "Low", 1: "Medium", 2: "High"}
    demand_tier = tier_map.get(int(item_features.get("demand_tier", 1)), "Medium")

    return {
        "recommended_price_usd": round(predicted_price, 2),
        "price_vs_market": price_vs_market,
        "demand_tier": demand_tier,
        "confidence_score": round(r2, 4),
        "top_factors": top_factors,
    }


if __name__ == "__main__":
    print("Testing pricing recommender...")
    model, metadata = load_production_model()
    feature_cols = metadata["feature_cols"]

    sample = {col: 0.0 for col in feature_cols}
    sample["demand_score"] = 10
    sample["demand_tier"] = 1
    sample["price_position"] = 1.0
    sample["conversion_rate"] = 0.005
    sample["cat_Electronics"] = 1
    sample["log_demand_score"] = np.log1p(10)
    sample["avg_competitor_price_usd"] = 50.0
    sample["log_competitor_price"] = np.log1p(50.0)

    result = recommend_price(sample)
    print(f"\nRecommendation:")
    for k, v in result.items():
        print(f"  {k}: {v}")
