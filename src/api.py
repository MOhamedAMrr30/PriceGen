# FastAPI pricing API for Dynamic Pricing Engine

import os
import sys
import types
import pickle
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# load model + metadata
print("[API] Loading model ...", flush=True)
with open(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model_metadata.pkl"), "rb") as f:
    METADATA = pickle.load(f)

FEATURE_COLS = METADATA["feature_cols"]
R2_SCORE = METADATA["r2_score"]

# pre-compute top features
_importances = MODEL.feature_importances_
_global_top_idx = np.argsort(_importances)[::-1]
GLOBAL_TOP_FEATURES = [FEATURE_COLS[i] for i in _global_top_idx[:5]]
print(f"[API] Model loaded. Features: {len(FEATURE_COLS)}, R²: {R2_SCORE:.4f}", flush=True)
print(f"[API] Top features: {GLOBAL_TOP_FEATURES}", flush=True)

VALID_CATEGORIES = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Other"]

ELASTICITY_MAP = {
    "Electronics": -0.015, "Fashion": -0.008, "Home": -0.012,
    "Beauty": -0.005, "Sports": -0.010, "Other": -0.007,
}


# request/response schemas

class ItemInput(BaseModel):
    itemid: str = Field(..., description="Unique item identifier")
    category_bucket: str = Field(..., description="Product category")
    demand_score: float = Field(..., ge=0)
    conversion_rate: float = Field(..., ge=0, le=1)
    event_hour: int = Field(..., ge=0, le=23)
    avg_competitor_price_usd: float = Field(..., ge=0)
    min_competitor_price_usd: float = Field(0.0, ge=0)
    max_competitor_price_usd: float = Field(0.0, ge=0)
    avg_discount_percentage: float = Field(0.0, ge=0, le=100)
    view_to_cart_rate: float = Field(0.0, ge=0, le=1)
    event_dayofweek: int = Field(2, ge=0, le=6)
    event_month: int = Field(6, ge=1, le=12)
    avg_popularity_score: float = Field(0.0, ge=0)

    @validator("category_bucket")
    @classmethod
    def validate_category(cls, v):
        if v not in VALID_CATEGORIES:
            raise ValueError(f"Must be one of: {VALID_CATEGORIES}")
        return v


class PricingResult(BaseModel):
    itemid: str
    recommended_price_usd: float
    price_vs_market: str
    demand_tier: str
    confidence_score: float
    top_factors: List[str]
    timestamp: str


class BatchInput(BaseModel):
    items: List[ItemInput]


class HealthResponse(BaseModel):
    status: str
    model: str


# prediction helpers

def _demand_tier(score):
    if score < 5:
        return 0
    elif score < 20:
        return 1
    return 2


def _build_features(item):
    dt = _demand_tier(item.demand_score)
    return {
        "demand_score": item.demand_score,
        "conversion_rate": item.conversion_rate,
        "view_to_cart_rate": item.view_to_cart_rate,
        "event_hour": float(item.event_hour),
        "event_dayofweek": float(item.event_dayofweek),
        "event_month": float(item.event_month),
        "avg_discount_percentage": item.avg_discount_percentage,
        "avg_popularity_score": item.avg_popularity_score,
        "price_position": 1.0,
        "avg_competitor_price_usd": item.avg_competitor_price_usd,
        "min_competitor_price_usd": item.min_competitor_price_usd,
        "max_competitor_price_usd": item.max_competitor_price_usd,
        "price_elasticity_proxy": ELASTICITY_MAP.get(item.category_bucket, -0.007),
        "demand_tier": dt,
        "is_peak_hour": 1 if item.event_hour == 17 else 0,
        "is_peak_day": 1 if item.event_dayofweek == 2 else 0,
        "log_demand_score": float(np.log1p(item.demand_score)),
        "log_competitor_price": float(np.log1p(item.avg_competitor_price_usd)),
        "cat_Beauty": 1 if item.category_bucket == "Beauty" else 0,
        "cat_Electronics": 1 if item.category_bucket == "Electronics" else 0,
        "cat_Fashion": 1 if item.category_bucket == "Fashion" else 0,
        "cat_Home": 1 if item.category_bucket == "Home" else 0,
        "cat_Other": 1 if item.category_bucket == "Other" else 0,
        "cat_Sports": 1 if item.category_bucket == "Sports" else 0,
    }


def _get_top_factors(X):
    # weight importances by actual input values
    vals = X.iloc[0].values.astype(float)
    weighted = _importances * np.abs(vals)
    top_idx = np.argsort(weighted)[::-1][:3]
    return [FEATURE_COLS[i] for i in top_idx]


def _predict_item(item):
    features = _build_features(item)
    X = pd.DataFrame([features])[FEATURE_COLS]

    pred = max(float(MODEL.predict(X)[0]), 0.01)

    # market position
    pp = pred / item.avg_competitor_price_usd if item.avg_competitor_price_usd > 0 else 1.0
    if pp > 1.05:
        market_pos = "above"
    elif pp < 0.95:
        market_pos = "below"
    else:
        market_pos = "at"

    tier_map = {0: "Low", 1: "Medium", 2: "High"}
    tier_label = tier_map.get(_demand_tier(item.demand_score), "Medium")
    top_factors = _get_top_factors(X)

    return PricingResult(
        itemid=item.itemid,
        recommended_price_usd=round(pred, 2),
        price_vs_market=market_pos,
        demand_tier=tier_label,
        confidence_score=round(R2_SCORE, 4),
        top_factors=top_factors,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# app

app = FastAPI(
    title="Dynamic Pricing Engine API",
    description="Pricing recommendations for e-commerce using XGBoost",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(status="ok", model="xgboost_best_model")


@app.get("/categories", tags=["Reference"])
def get_categories():
    return {"categories": VALID_CATEGORIES}


@app.post("/predict", response_model=PricingResult, tags=["Pricing"])
def predict(item: ItemInput):
    try:
        return _predict_item(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=List[PricingResult], tags=["Pricing"])
def batch_predict(batch: BatchInput):
    if len(batch.items) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100")
    try:
        return [_predict_item(item) for item in batch.items]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
