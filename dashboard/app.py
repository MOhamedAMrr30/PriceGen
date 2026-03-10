# PriceGen - intelligent pricing dashboard

import streamlit as st

st.set_page_config(page_title="PriceGen", page_icon="💹", layout="wide", initial_sidebar_state="expanded")

import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
EDA_DIR = os.path.join(BASE_DIR, "outputs", "eda")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "outputs", "model")

VALID_CATEGORIES = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Other"]
ELASTICITY_MAP = {
    "Electronics": -0.015, "Fashion": -0.008, "Home": -0.012,
    "Beauty": -0.005, "Sports": -0.010, "Other": -0.007,
}
CATEGORY_EMOJIS = {
    "Electronics": "💻", "Fashion": "👗", "Home": "🏠",
    "Beauty": "💄", "Sports": "⚽", "Other": "📦",
}

# inject global css
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --primary: #1E3A5F;
        --accent: #00C9A7;
        --bg-dark: #0F1923;
        --card-bg: #1A2B3C;
        --text-main: #FFFFFF;
        --text-muted: #A0AEC0;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0F1923 0%, #1a2332 50%, #0F1923 100%);
        color: var(--text-main);
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #132238 50%, #0a1628 100%) !important;
        border-right: 1px solid rgba(0, 201, 167, 0.15);
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(0, 201, 167, 0.06);
        border: 1px solid rgba(0, 201, 167, 0.12);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 3px 0;
        transition: all 0.3s ease;
        font-size: 0.92rem;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0, 201, 167, 0.15);
        border-color: rgba(0, 201, 167, 0.3);
        transform: translateX(4px);
    }

    /* buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00C9A7 0%, #00A389 100%) !important;
        color: #0F1923 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 201, 167, 0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00A389 0%, #008B73 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 201, 167, 0.4) !important;
    }

    /* input fields */
    .stSelectbox > div > div, .stSlider, .stNumberInput > div > div > input {
        background: var(--card-bg) !important;
        color: var(--text-main) !important;
        border-color: rgba(0, 201, 167, 0.2) !important;
    }

    /* hero section */
    .hero-container {
        text-align: center;
        padding: 60px 20px 40px;
    }
    .hero-logo {
        font-size: 4rem; font-weight: 900;
        background: linear-gradient(135deg, #00C9A7 0%, #00E5BE 50%, #00C9A7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
    }
    .hero-tagline {
        font-size: 1.4rem; font-weight: 300;
        color: #A0AEC0; margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 0.95rem; color: #718096;
        margin-bottom: 32px;
    }

    /* metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A2B3C 0%, #1f3347 100%);
        border: 1px solid rgba(0, 201, 167, 0.15);
        border-radius: 14px;
        padding: 22px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 201, 167, 0.4);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 201, 167, 0.1);
    }
    .metric-icon { font-size: 1.5rem; margin-bottom: 6px; }
    .metric-value {
        font-size: 1.8rem; font-weight: 800; color: #00C9A7;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.78rem; font-weight: 500; color: #A0AEC0;
        text-transform: uppercase; letter-spacing: 0.06em;
    }

    /* price hero */
    .price-hero {
        background: linear-gradient(135deg, #1E3A5F 0%, #2a4a6f 50%, #1E3A5F 100%);
        border: 1px solid rgba(0, 201, 167, 0.25);
        border-radius: 20px; padding: 36px; text-align: center;
        box-shadow: 0 12px 40px rgba(0, 201, 167, 0.15);
    }
    .price-amount {
        font-size: 3.5rem; font-weight: 900;
        color: #00C9A7; letter-spacing: -0.02em;
    }
    .price-label {
        font-size: 0.85rem; font-weight: 500; color: #A0AEC0;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
    }

    /* badges */
    .badge {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem; letter-spacing: 0.02em;
    }
    .badge-above { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .badge-below { background: rgba(0,201,167,0.2); color: #00C9A7; border: 1px solid rgba(0,201,167,0.3); }
    .badge-at { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .badge-high { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
    .badge-medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
    .badge-low { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }

    /* factor card */
    .factor-card {
        background: rgba(0, 201, 167, 0.06);
        border-left: 3px solid #00C9A7;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    }

    /* step card */
    .step-card {
        background: linear-gradient(135deg, #1A2B3C 0%, #1f3347 100%);
        border: 1px solid rgba(0, 201, 167, 0.12);
        border-radius: 14px; padding: 28px 20px; text-align: center;
        min-height: 180px;
    }
    .step-num {
        font-size: 2.5rem; margin-bottom: 10px;
    }
    .step-title {
        font-size: 1rem; font-weight: 700; color: #00C9A7;
        margin-bottom: 8px;
    }
    .step-desc {
        font-size: 0.85rem; color: #A0AEC0; line-height: 1.5;
    }

    /* section */
    .section-title {
        font-size: 1.6rem; font-weight: 700; color: #ffffff;
        margin-bottom: 4px;
    }
    .section-sub {
        font-size: 0.95rem; color: #718096; margin-bottom: 20px;
    }
    .divider {
        height: 2px; border: none; margin: 30px 0;
        background: linear-gradient(90deg, transparent, rgba(0,201,167,0.3), transparent);
    }

    /* info panel */
    .info-panel {
        background: linear-gradient(135deg, #1A2B3C 0%, #1f3347 100%);
        border: 1px dashed rgba(0, 201, 167, 0.25);
        border-radius: 14px; padding: 24px;
    }

    /* page header */
    .page-header {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #00C9A7 0%, #00E5BE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .page-sub {
        color: #718096; font-size: 1rem; margin-top: -4px; margin-bottom: 20px;
    }

    /* table in dark theme */
    .dark-table {
        width: 100%; border-collapse: collapse; margin: 16px 0;
    }
    .dark-table th {
        background: rgba(0,201,167,0.1); color: #00C9A7;
        padding: 10px 16px; text-align: left; font-weight: 600;
        border-bottom: 2px solid rgba(0,201,167,0.2);
    }
    .dark-table td {
        padding: 10px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #A0AEC0;
    }
    .dark-table tr:hover td { background: rgba(0,201,167,0.04); }

    /* footer */
    .footer-text {
        text-align: center; color: #4A5568; font-size: 0.85rem;
        padding: 30px 0 10px; border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 40px;
    }
    .footer-text a { color: #00C9A7; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# load model
@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return model, metadata, model.feature_importances_

MODEL, METADATA, IMPORTANCES = load_model()
FEATURE_COLS = METADATA["feature_cols"]
R2_SCORE = METADATA["r2_score"]


def demand_tier(score):
    if score < 5: return 0
    elif score < 20: return 1
    return 2


def predict_item(cat, demand, conv, hour, avg_c, min_c, max_c, disc,
                 vtc=0.0, dow=2, month=6, pop=0.0):
    dt = demand_tier(demand)
    features = {
        "demand_score": demand, "conversion_rate": conv, "view_to_cart_rate": vtc,
        "event_hour": float(hour), "event_dayofweek": float(dow), "event_month": float(month),
        "avg_discount_percentage": disc, "avg_popularity_score": pop, "price_position": 1.0,
        "avg_competitor_price_usd": avg_c, "min_competitor_price_usd": min_c,
        "max_competitor_price_usd": max_c,
        "price_elasticity_proxy": ELASTICITY_MAP.get(cat, -0.007),
        "demand_tier": dt, "is_peak_hour": 1 if hour == 17 else 0,
        "is_peak_day": 1 if dow == 2 else 0,
        "log_demand_score": float(np.log1p(demand)),
        "log_competitor_price": float(np.log1p(avg_c)),
        "cat_Beauty": 1 if cat == "Beauty" else 0,
        "cat_Electronics": 1 if cat == "Electronics" else 0,
        "cat_Fashion": 1 if cat == "Fashion" else 0,
        "cat_Home": 1 if cat == "Home" else 0,
        "cat_Other": 1 if cat == "Other" else 0,
        "cat_Sports": 1 if cat == "Sports" else 0,
    }
    X = pd.DataFrame([features])[FEATURE_COLS]
    pred = max(float(MODEL.predict(X)[0]), 0.01)
    pp = pred / avg_c if avg_c > 0 else 1.0
    market_pos = "above" if pp > 1.05 else ("below" if pp < 0.95 else "at")
    tier_map = {0: "Low", 1: "Medium", 2: "High"}
    vals = X.iloc[0].values.astype(float)
    weighted = IMPORTANCES * np.abs(vals)
    top_idx = np.argsort(weighted)[::-1][:3]
    top_factors = [FEATURE_COLS[i] for i in top_idx]
    return {
        "recommended_price_usd": round(pred, 2), "price_vs_market": market_pos,
        "demand_tier": tier_map.get(dt, "Medium"), "confidence_score": round(R2_SCORE, 4),
        "top_factors": top_factors, "price_position": round(pp, 4),
        "min_comp": min_c, "avg_comp": avg_c, "max_comp": max_c,
    }


# sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 5px;">
        <div style="font-size:2rem;font-weight:900;background:linear-gradient(135deg,#00C9A7,#00E5BE);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">💹 PriceGen</div>
        <div style="color:#718096;font-size:0.75rem;font-weight:500;letter-spacing:0.1em;text-transform:uppercase;">
            Intelligent Pricing Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:2px;background:linear-gradient(90deg,transparent,rgba(0,201,167,0.4),transparent);margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#718096 !important;font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;">NAVIGATION</div>', unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠 Home",
        "🎯 Single Item Pricer",
        "📊 Batch Analyzer",
        "🌍 Market Intelligence",
        "🧪 MLflow Experiments",
    ], label_visibility="collapsed")

    st.markdown('<div style="height:2px;background:linear-gradient(90deg,transparent,rgba(0,201,167,0.4),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)

    # model status card
    st.markdown(f"""
    <div style="background:rgba(0,201,167,0.06);border:1px solid rgba(0,201,167,0.15);border-radius:10px;padding:14px;margin-top:8px;">
        <div style="font-size:0.7rem;font-weight:600;color:#718096;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Model Status</div>
        <div style="display:flex;align-items:center;gap:8px;margin:6px 0;">
            <span style="color:#00C9A7;">🟢</span>
            <span style="color:#A0AEC0;font-size:0.85rem;">Model: Online</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin:6px 0;">
            <span>📦</span>
            <span style="color:#A0AEC0;font-size:0.85rem;">Version: Production</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin:6px 0;">
            <span>🎯</span>
            <span style="color:#00C9A7;font-size:0.85rem;font-weight:700;">R²: {R2_SCORE:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:10px 0;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:8px 0;">
        <div style="color:#A0AEC0;font-size:0.85rem;font-weight:600;">Mohamed Amr</div>
        <div style="margin-top:6px;">
            <a href="https://github.com/MOhamedAMrr30" style="color:#00C9A7;text-decoration:none;font-size:0.8rem;margin:0 6px;">GitHub</a>
            <a href="https://www.linkedin.com/in/mohameddamrr/" style="color:#00C9A7;text-decoration:none;font-size:0.8rem;margin:0 6px;">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE: HOME ──

def page_home():
    # hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-logo">💹 PriceGen</div>
        <div class="hero-tagline">Intelligent Pricing for Modern E-Commerce</div>
        <div class="hero-subtitle">Powered by XGBoost · SHAP Explainability · Real-Time Market Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # CTA buttons
    c1, c2, c3, c4 = st.columns([2, 3, 3, 2])
    with c2:
        st.button("🎯 Try Single Item Pricer", use_container_width=True, key="cta1")
    with c3:
        st.button("📊 Batch Analyze", use_container_width=True, key="cta2")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # metrics bar
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown("""<div class="metric-card">
        <div class="metric-icon">💰</div>
        <div class="metric-value">166,961</div>
        <div class="metric-label">Items Analyzed</div>
    </div>""", unsafe_allow_html=True)
    m2.markdown("""<div class="metric-card">
        <div class="metric-icon">🎯</div>
        <div class="metric-value">99.97%</div>
        <div class="metric-label">Model Accuracy (R²)</div>
    </div>""", unsafe_allow_html=True)
    m3.markdown("""<div class="metric-card">
        <div class="metric-icon">⚡</div>
        <div class="metric-value">&lt;6ms</div>
        <div class="metric-label">API Latency</div>
    </div>""", unsafe_allow_html=True)
    m4.markdown("""<div class="metric-card">
        <div class="metric-icon">📦</div>
        <div class="metric-value">6</div>
        <div class="metric-label">Categories</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # how it works
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Three steps from input to explainable pricing</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.markdown("""<div class="step-card">
        <div class="step-num">🔍</div>
        <div class="step-title">Step 1 — Input</div>
        <div class="step-desc">Provide item features and competitor pricing context</div>
    </div>""", unsafe_allow_html=True)
    s2.markdown("""<div class="step-card">
        <div class="step-num">🧠</div>
        <div class="step-title">Step 2 — Predict</div>
        <div class="step-desc">XGBoost model calculates the optimal price point</div>
    </div>""", unsafe_allow_html=True)
    s3.markdown("""<div class="step-card">
        <div class="step-num">💡</div>
        <div class="step-title">Step 3 — Explain</div>
        <div class="step-desc">SHAP shows the top 3 factors driving the price</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # model performance
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Comparing baseline vs tuned XGBoost</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="dark-table">
        <tr><th>Model</th><th>MAE (USD)</th><th>R² Score</th><th>Status</th></tr>
        <tr><td>Linear Regression</td><td>$218.17</td><td>0.8488</td><td style="color:#718096;">Baseline</td></tr>
        <tr><td>XGBoost (default)</td><td>$5.96</td><td>0.9996</td><td style="color:#718096;">Initial</td></tr>
        <tr><td style="color:#00C9A7;font-weight:700;">XGBoost (tuned)</td><td style="color:#00C9A7;font-weight:700;">$5.31</td><td style="color:#00C9A7;font-weight:700;">0.9997</td><td><span class="badge badge-below">Production ⭐</span></td></tr>
    </table>
    """, unsafe_allow_html=True)

    # footer
    st.markdown("""
    <div class="footer-text">
        Built by <strong>Mohamed Amr</strong> ·
        <a href="https://github.com/MOhamedAMrr30">GitHub</a> ·
        <a href="https://www.linkedin.com/in/mohameddamrr/">LinkedIn</a><br>
        Trained on Retail Rocket + Amazon datasets
    </div>
    """, unsafe_allow_html=True)


# ── PAGE: SINGLE ITEM PRICER ──

def page_single_pricer():
    st.markdown('<p class="page-header">🎯 Single Item Pricer</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Get an AI-powered pricing recommendation in seconds</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_form, _, col_result = st.columns([4, 0.5, 5])

    with col_form:
        st.markdown("#### 📝 Item Configuration")
        category = st.selectbox("Category", VALID_CATEGORIES,
                                format_func=lambda x: f"{CATEGORY_EMOJIS.get(x, '📦')} {x}")
        st.markdown("##### 📊 Demand Signals")
        demand_score = st.slider("Demand Score", 0.0, 500.0, 25.0, 1.0)
        conv = st.slider("Conversion Rate", 0.0, 0.1, 0.005, 0.001, format="%.4f")
        hour = st.slider("Peak Event Hour", 0, 23, 14)

        st.markdown("##### 💲 Competitor Prices (USD)")
        c1, c2, c3 = st.columns(3)
        min_c = c1.number_input("Floor", value=10.0, min_value=0.0, step=1.0)
        avg_c = c2.number_input("Average", value=50.0, min_value=0.01, step=1.0)
        max_c = c3.number_input("Ceiling", value=100.0, min_value=0.0, step=1.0)
        disc = st.slider("Avg Discount %", 0.0, 80.0, 35.0, 1.0)

        st.markdown("")
        btn = st.button("🚀 Generate Pricing Recommendation", type="primary", use_container_width=True)

    with col_result:
        if btn:
            with st.spinner("Running model..."):
                r = predict_item(category, demand_score, conv, hour, avg_c, min_c, max_c, disc)

            # price hero
            st.markdown(f"""
            <div class="price-hero">
                <div class="price-label">AI Recommended Price</div>
                <div class="price-amount">${r["recommended_price_usd"]:.2f}</div>
                <div style="margin-top:14px;">
                    <span class="badge badge-{r['price_vs_market']}">{r["price_vs_market"].upper()} MARKET</span>
                    &nbsp;
                    <span class="badge badge-{r['demand_tier'].lower()}">{r["demand_tier"]} DEMAND</span>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("")

            # stats
            s1, s2, s3 = st.columns(3)
            # confidence badge color
            conf_color = "#00C9A7" if r["confidence_score"] > 0.99 else ("#fbbf24" if r["confidence_score"] > 0.95 else "#f87171")
            s1.markdown(f"""<div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color:{conf_color};">{r["confidence_score"]:.4f}</div>
                <div style="color:#718096;font-size:0.75rem;">R² Score</div>
            </div>""", unsafe_allow_html=True)

            delta_pct = ((r["recommended_price_usd"] - avg_c) / avg_c * 100) if avg_c > 0 else 0
            delta_color = "#00C9A7" if delta_pct <= 0 else "#f87171"
            s2.markdown(f"""<div class="metric-card">
                <div class="metric-label">vs Market Avg</div>
                <div class="metric-value" style="color:{delta_color};">{delta_pct:+.1f}%</div>
                <div style="color:#718096;font-size:0.75rem;">vs ${avg_c:.0f}</div>
            </div>""", unsafe_allow_html=True)

            s3.markdown(f"""<div class="metric-card">
                <div class="metric-label">Price Position</div>
                <div class="metric-value">{r['price_position']:.2f}x</div>
                <div style="color:#718096;font-size:0.75rem;">of market avg</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("")

            # top drivers
            st.markdown("#### 🎯 Top Price Drivers")
            for i, f in enumerate(r["top_factors"], 1):
                st.markdown(f"""<div class="factor-card">
                    <span style="font-weight:700;color:#00C9A7;">#{i}</span>&nbsp;&nbsp;
                    <span style="color:#E2E8F0;">{f.replace('_',' ').title()}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # gauge
            st.markdown("#### 📊 Competitive Positioning")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=r["recommended_price_usd"],
                delta={"reference": avg_c, "relative": True, "valueformat": ".1%"},
                number={"prefix": "$", "font": {"size": 34, "color": "#00C9A7"}},
                gauge={
                    "axis": {"range": [0, max(max_c * 1.3, r["recommended_price_usd"] * 1.3)],
                             "tickcolor": "#4A5568"},
                    "bar": {"color": "#00C9A7", "thickness": 0.7},
                    "bgcolor": "#1A2B3C",
                    "bordercolor": "rgba(0,201,167,0.2)",
                    "steps": [
                        {"range": [0, min_c], "color": "rgba(0,201,167,0.08)"},
                        {"range": [min_c, avg_c], "color": "rgba(251,191,36,0.08)"},
                        {"range": [avg_c, max_c], "color": "rgba(239,68,68,0.08)"},
                    ],
                    "threshold": {"line": {"color": "#f87171", "width": 3}, "thickness": 0.8, "value": avg_c},
                },
            ))
            fig.update_layout(height=260, margin=dict(t=30, b=10, l=30, r=30),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family": "Inter", "color": "#A0AEC0"})
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div class="info-panel" style="text-align:center;padding:80px 30px;">
                <div style="font-size:3.5rem;margin-bottom:16px;">🤖</div>
                <div style="font-size:1.2rem;font-weight:600;color:#00C9A7;margin-bottom:8px;">Ready to Predict</div>
                <div style="color:#718096;">Configure item details and click <strong style="color:#00C9A7;">Generate Pricing Recommendation</strong></div>
            </div>""", unsafe_allow_html=True)


# ── PAGE: BATCH ANALYZER ──

def page_batch():
    st.markdown('<p class="page-header">📊 Batch Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Upload items in bulk for pricing analysis</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.expander("📋 Required CSV Columns", expanded=False):
        st.code("category_bucket, demand_score, conversion_rate, event_hour,\n"
                "avg_competitor_price_usd, min_competitor_price_usd,\n"
                "max_competitor_price_usd, avg_discount_percentage")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(8), use_container_width=True, height=250)

        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{len(df.columns)}</div></div>', unsafe_allow_html=True)
        cats = df.get("category_bucket", pd.Series()).nunique()
        m3.markdown(f'<div class="metric-card"><div class="metric-label">Categories</div><div class="metric-value">{cats}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
            results = []
            bar = st.progress(0, text="Analyzing...")
            for i, row in df.iterrows():
                try:
                    r = predict_item(
                        row.get("category_bucket", "Other"), float(row.get("demand_score", 10)),
                        float(row.get("conversion_rate", 0.005)), int(row.get("event_hour", 14)),
                        float(row.get("avg_competitor_price_usd", 50)),
                        float(row.get("min_competitor_price_usd", 10)),
                        float(row.get("max_competitor_price_usd", 100)),
                        float(row.get("avg_discount_percentage", 30)))
                    r["itemid"] = row.get("itemid", str(i))
                    results.append(r)
                except Exception as e:
                    results.append({"itemid": str(i), "error": str(e)})
                bar.progress((i + 1) / len(df), text=f"Analyzing... {i+1}/{len(df)}")
            bar.empty()
            res_df = pd.DataFrame(results)

            # summary cards
            if "price_vs_market" in res_df.columns:
                above = (res_df["price_vs_market"] == "above").sum()
                below = (res_df["price_vs_market"] == "below").sum()
                avg_p = res_df["recommended_price_usd"].mean()
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.markdown(f'<div class="metric-card"><div class="metric-label">Total Items</div><div class="metric-value">{len(res_df)}</div></div>', unsafe_allow_html=True)
                sm2.markdown(f'<div class="metric-card"><div class="metric-label">Avg Price</div><div class="metric-value" style="color:#00C9A7;">${avg_p:.2f}</div></div>', unsafe_allow_html=True)
                sm3.markdown(f'<div class="metric-card"><div class="metric-label">Above Market</div><div class="metric-value" style="color:#f87171;">{above}</div></div>', unsafe_allow_html=True)
                sm4.markdown(f'<div class="metric-card"><div class="metric-label">Below Market</div><div class="metric-value" style="color:#00C9A7;">{below}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            st.markdown("#### 📊 Results")

            def color_market(val):
                if val == "below": return "background-color: rgba(0,201,167,0.15); color: #00C9A7"
                elif val == "above": return "background-color: rgba(239,68,68,0.15); color: #f87171"
                return "background-color: rgba(251,191,36,0.15); color: #fbbf24"

            display_cols = [c for c in ["itemid", "recommended_price_usd", "price_vs_market",
                                         "demand_tier", "price_position"] if c in res_df.columns]
            styled = res_df[display_cols].style.applymap(
                color_market, subset=["price_vs_market"] if "price_vs_market" in display_cols else [])
            st.dataframe(styled, use_container_width=True, height=350)

            csv_out = res_df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv_out, "pricing_results.csv", "text/csv", use_container_width=True)
    else:
        st.markdown("""
        <div class="info-panel" style="text-align:center;padding:70px 30px;">
            <div style="font-size:3rem;margin-bottom:14px;">📁</div>
            <div style="font-size:1.1rem;font-weight:600;color:#00C9A7;margin-bottom:8px;">Upload Your Dataset</div>
            <div style="color:#718096;">Drag and drop a CSV file to begin batch analysis</div>
        </div>""", unsafe_allow_html=True)


# ── PAGE: MARKET INTELLIGENCE ──

def page_market():
    st.markdown('<p class="page-header">🌍 Market Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Explore insights from 167K items across 6 categories</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    master_path = os.path.join(DATA_DIR, "master_pricing_data_usd.csv")
    if os.path.exists(master_path):
        master = pd.read_csv(master_path, dtype={"itemid": str, "categoryid": str})

        # category filter
        cats = ["All"] + sorted(master["category_bucket"].dropna().unique().tolist())
        selected = st.selectbox("🔍 Filter by Category", cats,
                                format_func=lambda x: f"{CATEGORY_EMOJIS.get(x, '🌐')} {x}")
        if selected != "All":
            master = master[master["category_bucket"] == selected]

        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        avg_p = master["item_price_usd"].mean()
        med_p = master["item_price_usd"].median()

        m1.markdown(f'<div class="metric-card"><div class="metric-icon">📦</div><div class="metric-value">{len(master):,}</div><div class="metric-label">Items</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-icon">💰</div><div class="metric-value">${avg_p:.2f}</div><div class="metric-label">Avg Price</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-icon">📊</div><div class="metric-value">${med_p:.2f}</div><div class="metric-label">Median Price</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-icon">⏰</div><div class="metric-value">17:00</div><div class="metric-label">Peak Hour</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # category chart
        cat_stats = master.groupby("category_bucket").agg(
            count=("item_price_usd", "count"), mean_price=("item_price_usd", "mean")).round(2).reset_index()

        fig = px.bar(cat_stats, x="category_bucket", y="mean_price", color="category_bucket",
                     color_discrete_sequence=["#00C9A7", "#1E3A5F", "#00A389", "#2a5a8f", "#00E5BE", "#4A90C9"],
                     text="mean_price", hover_data=["count"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"family": "Inter", "color": "#A0AEC0"}, height=380,
                          showlegend=False, xaxis_title="", yaxis_title="Avg Price (USD)",
                          xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Master dataset not found.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📉 Visualizations")

    eda_charts = {
        "Event Distribution": "event_distribution.png",
        "Demand Score": "demand_score_distribution.png",
        "Conversion by Hour": "conversion_by_hour.png",
        "Conversion by Day": "conversion_by_dayofweek.png",
        "Top 20 Demand": "top20_demand_items.png",
        "Price by Category": "price_by_category.png",
        "Discount by Category": "discount_by_category.png",
        "Correlation Heatmap": "correlation_heatmap.png",
    }
    model_charts = {
        "SHAP Summary": "shap_summary_plot.png",
        "SHAP Importance": "shap_bar_plot.png",
        "SHAP Waterfall": "shap_waterfall_sample.png",
    }

    tab1, tab2 = st.tabs(["📊 EDA Charts", "🤖 Model Explainability"])
    with tab1:
        name = st.selectbox("Select chart", list(eda_charts.keys()), key="eda")
        path = os.path.join(EDA_DIR, eda_charts[name])
        if os.path.exists(path):
            st.image(Image.open(path), caption=name, use_column_width=True)
        else:
            st.warning(f"Not found: {eda_charts[name]}")

    with tab2:
        name2 = st.selectbox("Select chart", list(model_charts.keys()), key="model")
        path2 = os.path.join(MODEL_OUT_DIR, model_charts[name2])
        if os.path.exists(path2):
            st.image(Image.open(path2), caption=name2, use_column_width=True)
        else:
            st.warning(f"Not found: {model_charts[name2]}")


# ── PAGE: MLFLOW EXPERIMENTS ──

def page_mlflow():
    st.markdown('<p class="page-header">🧪 MLflow Experiments</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Track, compare, and manage model experiments</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # try iframe
    st.markdown("""
    <div style="background:#1A2B3C;border:1px solid rgba(0,201,167,0.15);border-radius:12px;overflow:hidden;">
        <iframe src="http://127.0.0.1:5000" width="100%" height="650" style="border:none;border-radius:12px;"></iframe>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚠️ MLflow UI not loading? Click here", expanded=False):
        st.markdown("""
        **To start the MLflow UI, run this command in your terminal:**
        ```bash
        mlflow ui --backend-store-uri mlflow_tracking/ --port 5000
        ```
        Then refresh this page.
        """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📋 Experiment Summary (Static Fallback)")

    st.markdown("""
    <table class="dark-table">
        <tr><th>Run</th><th>Model</th><th>MAE (USD)</th><th>RMSE (USD)</th><th>R²</th><th>Time</th></tr>
        <tr><td>1</td><td>Linear Regression</td><td>218.17</td><td>303.09</td><td>0.8488</td><td>0.14s</td></tr>
        <tr><td>2</td><td>XGBoost Default</td><td>5.96</td><td>15.32</td><td>0.9996</td><td>14.93s</td></tr>
        <tr><td>3</td><td>XGBoost Underfit (d=4)</td><td>75.30</td><td>102.93</td><td>0.9826</td><td>8.12s</td></tr>
        <tr style="color:#00C9A7;font-weight:600;"><td>4 ⭐</td><td>XGBoost Tuned (d=8)</td><td>5.31</td><td>14.52</td><td>0.9997</td><td>12.63s</td></tr>
        <tr><td>5</td><td>XGBoost Overfit (d=10)</td><td>9.03</td><td>20.54</td><td>0.9993</td><td>39.66s</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(0,201,167,0.06);border:1px solid rgba(0,201,167,0.15);border-radius:10px;padding:16px;margin-top:16px;">
        <strong style="color:#00C9A7;">Why Run 4?</strong><br>
        <span style="color:#A0AEC0;">Best MAE ($5.31) and R² (0.9997) with reasonable complexity (max_depth=8, lr=0.1). Run 3 underfits, Run 5 overfits with 3x training time for worse results.</span>
    </div>
    """, unsafe_allow_html=True)


# routing
if page == "🏠 Home":
    page_home()
elif page == "🎯 Single Item Pricer":
    page_single_pricer()
elif page == "📊 Batch Analyzer":
    page_batch()
elif page == "🌍 Market Intelligence":
    page_market()
elif page == "🧪 MLflow Experiments":
    page_mlflow()
