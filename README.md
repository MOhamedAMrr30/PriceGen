# 💰 Dynamic Pricing Engine for E-Commerce

> AI-powered dynamic pricing using XGBoost, SHAP explainability, and real-time competitor intelligence — built for platforms like noon and Jumia.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?logo=xgboost)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)

---

## 🏗️ Architecture

```
Retail Rocket Data ──┐
  (2.7M events)      │
                     ├──► Data Pipeline ──► Master Dataset ──► XGBoost Model ──┬──► FastAPI (port 8000)
Amazon Data ─────────┘    (ETL + FE)       (167K items)       (R² = 0.9997)   ├──► Streamlit (port 8501)
  (534K products)                                                              └──► MLflow UI (port 5000)
```

## 📊 Business Problem

E-commerce platforms need to price millions of products optimally — balancing revenue maximization with conversion rates. This engine:

- **Ingests** behavioral signals (views, add-to-cart, transactions) + competitor pricing data
- **Predicts** optimal price per item using gradient-boosted trees
- **Explains** every recommendation via SHAP feature attribution
- **Tracks** all experiments via MLflow with model registry
- **Serves** predictions through a REST API and interactive dashboard

## 🚀 Quick Start

```bash
git clone https://github.com/MOhamedAMrr30/dynamic-pricing-engine.git
docker-compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

**Without Docker:**

```bash
pip install -r requirements.txt

# Run data pipeline (if raw data available)
python src/pipeline_retail_rocket.py
python src/pipeline_amazon.py
python src/combine_datasets.py
python src/eda.py

# Train model (logs 5 experiments to MLflow)
python src/train_model.py

# Start API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Start Dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8501

# Start MLflow UI (new terminal)
mlflow ui --backend-store-uri mlflow_tracking/ --port 5000
```

## 📈 Model Performance

| Model | MAE (USD) | RMSE (USD) | R² |
|-------|-----------|------------|-----|
| Linear Regression (baseline) | 218.17 | 303.09 | 0.8488 |
| XGBoost (initial) | 5.96 | 15.32 | 0.9996 |
| **XGBoost (tuned)** | **5.31** | **14.52** | **0.9997** |

**Top SHAP features:** `price_position`, `avg_competitor_price_usd`, `avg_discount_percentage`

## 🧪 Experiment Tracking (MLflow)

All training runs are tracked in MLflow with parameters, metrics, and artifacts.

| Run | Model | MAE | R² | Purpose |
|-----|-------|-----|----|---------|
| 1 | Linear Regression | 218.17 | 0.8488 | Baseline |
| 2 | XGBoost (d=6, lr=0.05) | 5.96 | 0.9996 | Default XGBoost |
| 3 | XGBoost (d=4, lr=0.01) | ~20 | ~0.99 | Underfitting test |
| **4** | **XGBoost (d=8, lr=0.1)** | **5.31** | **0.9997** | **Production ⭐** |
| 5 | XGBoost (d=10, lr=0.05) | ~5 | ~0.999 | Overfitting test |

**Why Run 4 was selected:** Best balance of low MAE, high R², and reasonable model complexity. Runs 3/5 tested under/overfitting boundaries — Run 4 sits at the sweet spot with `max_depth=8` and `learning_rate=0.1`.

**Launch MLflow UI:**

```bash
mlflow ui --backend-store-uri mlflow_tracking/ --port 5000
# Open http://localhost:5000
```

The UI shows: all 5 runs with metrics, parameter comparison, artifact viewer (plots + reports), and the registered production model `pricing-model-production`.

## 🔌 API Usage

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "itemid": "item_001",
    "category_bucket": "Electronics",
    "demand_score": 50,
    "conversion_rate": 0.008,
    "event_hour": 17,
    "avg_competitor_price_usd": 65.0,
    "min_competitor_price_usd": 30.0,
    "max_competitor_price_usd": 120.0,
    "avg_discount_percentage": 40.0
  }'
```

**Response:**

```json
{
  "itemid": "item_001",
  "recommended_price_usd": 65.24,
  "price_vs_market": "at",
  "demand_tier": "High",
  "confidence_score": 0.9997,
  "top_factors": ["price_position", "avg_competitor_price_usd", "avg_discount_percentage"],
  "timestamp": "2026-03-10T02:00:00Z"
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/categories` | GET | Valid category list |
| `/predict` | POST | Single item pricing |
| `/batch_predict` | POST | Batch pricing (max 100) |
| `/docs` | GET | Swagger UI (auto-generated) |

## 📁 Project Structure

```
dynamic-pricing-engine/
├── data/
│   ├── raw/                          # Raw datasets
│   ├── retail_rocket_features.csv    # Step 1 output
│   ├── amazon_features.csv           # Step 2 output
│   ├── master_pricing_data.csv       # Step 3 output
│   ├── master_pricing_data_usd.csv   # Currency-normalized
│   └── model_ready_data.csv          # Feature-engineered
├── src/
│   ├── pipeline_retail_rocket.py     # Behavioral ETL
│   ├── pipeline_amazon.py            # Competitor ETL
│   ├── combine_datasets.py           # Dataset merger
│   ├── eda.py                        # EDA + plots
│   ├── train_model.py                # MLflow-instrumented training
│   ├── pricing_recommender.py        # MLflow registry-enabled recommender
│   └── api.py                        # FastAPI endpoints
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── models/
│   ├── xgboost_best_model.pkl        # Tuned model
│   └── model_metadata.pkl            # Feature columns + R²
├── mlflow_tracking/                  # MLflow experiment store
├── outputs/
│   ├── eda/                          # 13 EDA plots + report
│   └── model/                        # SHAP plots + report
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🖼️ Screenshots

*Dashboard, API, and MLflow UI screenshots coming soon*

## 📊 Datasets

1. [Retail Rocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — 2.7M behavioral events
2. [Amazon Products Dataset](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset) — 534K product listings

## 👤 Author

**Mohamed Amr**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohameddamrr/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/MOhamedAMrr30)

---




```
Retail Rocket Data ──┐
  (2.7M events)      │
                     ├──► Data Pipeline ──► Master Dataset ──► XGBoost Model ──┬──► FastAPI (port 8000)
Amazon Data ─────────┘    (ETL + FE)       (167K items)       (R² = 0.9997)   │
  (534K products)                                                              └──► Streamlit (port 8501)
```

## 📊 Business Problem

E-commerce platforms need to price millions of products optimally — balancing revenue maximization with conversion rates. This engine:

- **Ingests** behavioral signals (views, add-to-cart, transactions) + competitor pricing data
- **Predicts** optimal price per item using gradient-boosted trees
- **Explains** every recommendation via SHAP feature attribution
- **Serves** predictions through a REST API and interactive dashboard

## 🚀 Quick Start

```bash
git clone https://github.com/MOhamedAMrr30/dynamic-pricing-engine.git
docker-compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

**Without Docker:**

```bash
pip install -r requirements.txt

# Run data pipeline (if raw data available)
python src/pipeline_retail_rocket.py
python src/pipeline_amazon.py
python src/combine_datasets.py
python src/eda.py

# Train model
python src/train_model.py

# Start API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Start Dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8501
```

## 📈 Model Performance

| Model | MAE (USD) | RMSE (USD) | R² |
|-------|-----------|------------|-----|
| Linear Regression (baseline) | 218.17 | 303.09 | 0.8488 |
| XGBoost (initial) | 5.96 | 15.32 | 0.9996 |
| **XGBoost (tuned)** | **5.31** | **14.52** | **0.9997** |

**Top SHAP features:** `price_position`, `avg_competitor_price_usd`, `avg_discount_percentage`

## 🔌 API Usage

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "itemid": "item_001",
    "category_bucket": "Electronics",
    "demand_score": 50,
    "conversion_rate": 0.008,
    "event_hour": 17,
    "avg_competitor_price_usd": 65.0,
    "min_competitor_price_usd": 30.0,
    "max_competitor_price_usd": 120.0,
    "avg_discount_percentage": 40.0
  }'
```

**Response:**

```json
{
  "itemid": "item_001",
  "recommended_price_usd": 65.24,
  "price_vs_market": "at",
  "demand_tier": "High",
  "confidence_score": 0.9997,
  "top_factors": ["price_position", "avg_competitor_price_usd", "avg_discount_percentage"],
  "timestamp": "2026-03-10T02:00:00Z"
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/categories` | GET | Valid category list |
| `/predict` | POST | Single item pricing |
| `/batch_predict` | POST | Batch pricing (max 100) |
| `/docs` | GET | Swagger UI (auto-generated) |

## 📁 Project Structure

```
dynamic-pricing-engine/
├── data/
│   ├── raw/                          # Raw datasets
│   ├── retail_rocket_features.csv    # Step 1 output
│   ├── amazon_features.csv           # Step 2 output
│   ├── master_pricing_data.csv       # Step 3 output
│   ├── master_pricing_data_usd.csv   # Currency-normalized
│   └── model_ready_data.csv          # Feature-engineered
├── src/
│   ├── pipeline_retail_rocket.py     # Behavioral ETL
│   ├── pipeline_amazon.py            # Competitor ETL
│   ├── combine_datasets.py           # Dataset merger
│   ├── eda.py                        # EDA + plots
│   ├── train_model.py                # XGBoost training
│   ├── pricing_recommender.py        # Recommender function
│   └── api.py                        # FastAPI endpoints
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── models/
│   ├── xgboost_best_model.pkl        # Tuned model
│   ├── xgboost_pricing_model.pkl     # Initial model
│   └── model_metadata.pkl            # Feature columns + R²
├── outputs/
│   ├── eda/                          # 13 EDA plots + report
│   └── model/                        # SHAP plots + report
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🖼️ Screenshots

*Dashboard and API screenshots coming soon*

## 📊 Datasets

1. [Retail Rocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — 2.7M behavioral events
2. [Amazon Products Dataset](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset) — 534K product listings

## 👤 Author

**Mohamed Amr**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohamedamrr30/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/MOhamedAMrr30)

---

*Built as a portfolio project demonstrating end-to-end ML engineering: data pipelines → feature engineering → model training → explainability → API deployment → interactive dashboard.*
