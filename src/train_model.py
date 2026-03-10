# MLflow experiment tracking - runs 5 training configs and registers best model

import os
import sys
import types
import pickle
import time
import warnings

# fix cv2/numpy conflict
cv2_mock = types.ModuleType("cv2")
cv2_mock.__version__ = "4.0.0"
sys.modules["cv2"] = cv2_mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.xgboost

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "model")
TRACKING_DIR = os.path.join(BASE_DIR, "mlflow_tracking")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def log(msg):
    print(msg, flush=True)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_ready_data.csv"),
                     dtype={"itemid": str, "categoryid": str})
    log(f"Loaded: {df.shape}")

    exclude_cols = ["itemid", "categoryid", "category_bucket", "item_price_usd"]
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, int, float]]

    clean = df[feature_cols + ["item_price_usd"]].dropna()
    X = clean[feature_cols]
    y = clean["item_price_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


def plot_feature_importance(model, feature_cols, title, filepath):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return None

    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(idx)), importances[idx][::-1], color="#667eea")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


def plot_actual_vs_predicted(y_true, y_pred, filepath):
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_idx = np.random.choice(len(y_true), min(2000, len(y_true)), replace=False)
    ax.scatter(y_true.values[sample_idx], y_pred[sample_idx], alpha=0.3, s=10, color="#764ba2")
    lims = [0, max(y_true.values[sample_idx].max(), y_pred[sample_idx].max()) * 1.1]
    ax.plot(lims, lims, "--", color="red", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Price (USD)")
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title("Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


def run_experiment(run_name, model, model_type, params,
                   X_train, X_test, y_train, y_test, feature_cols,
                   register_as_production=False):
    log(f"\n{'='*60}")
    log(f"RUN: {run_name}")
    log(f"{'='*60}")

    with mlflow.start_run(run_name=run_name) as run:
        # log params
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("random_state", 42)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("dataset_version", "master_pricing_data_usd")

        for k, v in params.items():
            mlflow.log_param(k, v)

        # train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # predict
        preds_train = model.predict(X_train)
        t1 = time.time()
        preds_test = model.predict(X_test)
        pred_time = (time.time() - t1) / len(X_test) * 1000

        # metrics
        metrics = {
            "mae_train": mean_absolute_error(y_train, preds_train),
            "mae_test": mean_absolute_error(y_test, preds_test),
            "rmse_train": np.sqrt(mean_squared_error(y_train, preds_train)),
            "rmse_test": np.sqrt(mean_squared_error(y_test, preds_test)),
            "r2_train": r2_score(y_train, preds_train),
            "r2_test": r2_score(y_test, preds_test),
            "training_time_seconds": round(train_time, 2),
            "prediction_latency_ms": round(pred_time, 4),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        log(f"   MAE test:  {metrics['mae_test']:.4f}")
        log(f"   RMSE test: {metrics['rmse_test']:.4f}")
        log(f"   R² test:   {metrics['r2_test']:.4f}")
        log(f"   Train time: {metrics['training_time_seconds']:.2f}s")
        log(f"   Pred latency: {metrics['prediction_latency_ms']:.4f}ms")

        # save plots
        tmp_fi = os.path.join(OUTPUT_DIR, f"fi_{run_name.replace(' ', '_')}.png")
        fi_path = plot_feature_importance(model, feature_cols, f"Feature Importance — {run_name}", tmp_fi)
        if fi_path:
            mlflow.log_artifact(fi_path, "plots")

        tmp_avp = os.path.join(OUTPUT_DIR, f"avp_{run_name.replace(' ', '_')}.png")
        avp_path = plot_actual_vs_predicted(y_test, preds_test, tmp_avp)
        if avp_path:
            mlflow.log_artifact(avp_path, "plots")

        # log existing artifacts
        for artifact_file in ["model_report.md"]:
            p = os.path.join(OUTPUT_DIR, artifact_file)
            if os.path.exists(p):
                mlflow.log_artifact(p, "reports")

        req_path = os.path.join(BASE_DIR, "requirements.txt")
        if os.path.exists(req_path):
            mlflow.log_artifact(req_path, "config")

        for shap_file in ["shap_summary_plot.png", "shap_bar_plot.png", "shap_waterfall_sample.png"]:
            sp = os.path.join(OUTPUT_DIR, shap_file)
            if os.path.exists(sp):
                mlflow.log_artifact(sp, "shap")

        # log model
        tags = {
            "dataset": "retail_rocket + amazon",
            "currency": "USD",
            "categories": "6 buckets",
            "author": "Mohamed Amr",
        }

        if model_type.startswith("xgboost"):
            mlflow.xgboost.log_model(
                model, "model",
                registered_model_name="pricing-model-production" if register_as_production else None,
            )
        else:
            mlflow.sklearn.log_model(
                model, "model",
                registered_model_name="pricing-model-production" if register_as_production else None,
            )

        mlflow.set_tags(tags)
        log(f"   Run ID: {run.info.run_id}")

        # save best model locally too
        if register_as_production:
            with open(os.path.join(MODEL_DIR, "xgboost_best_model.pkl"), "wb") as f:
                pickle.dump(model, f)
            log(f"   Registered as production model")

            metadata = {"feature_cols": list(X_train.columns), "r2_score": metrics["r2_test"]}
            with open(os.path.join(MODEL_DIR, "model_metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)

        return run.info.run_id, metrics


def main():
    os.makedirs(TRACKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tracking_uri = f"file:///{TRACKING_DIR.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment("dynamic-pricing-engine")
    log(f"MLflow tracking: {tracking_uri}")
    log(f"Experiment: {experiment.name} (ID: {experiment.experiment_id})")

    X_train, X_test, y_train, y_test, feature_cols = load_data()
    results = []

    # run 1: baseline
    rid, m = run_experiment(
        "Run 1 - Linear Regression", LinearRegression(),
        "linear_regression", {},
        X_train, X_test, y_train, y_test, feature_cols)
    results.append(("Run 1", "Linear Regression", m))

    # run 2: xgboost default
    rid, m = run_experiment(
        "Run 2 - XGBoost Default",
        XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0),
        "xgboost", {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8},
        X_train, X_test, y_train, y_test, feature_cols)
    results.append(("Run 2", "XGBoost Default", m))

    # run 3: underfit test
    rid, m = run_experiment(
        "Run 3 - XGBoost Underfit",
        XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.01,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0),
        "xgboost_underfit", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.01,
                             "subsample": 0.8, "colsample_bytree": 0.8},
        X_train, X_test, y_train, y_test, feature_cols)
    results.append(("Run 3", "XGBoost Underfit (d=4, lr=0.01)", m))

    # run 4: best params (production)
    rid, m = run_experiment(
        "Run 4 - XGBoost Best",
        XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1,
                     subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, verbosity=0),
        "xgboost_tuned", {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1,
                          "subsample": 0.9, "colsample_bytree": 0.9},
        X_train, X_test, y_train, y_test, feature_cols,
        register_as_production=True)
    results.append(("Run 4", "XGBoost Best (d=8, lr=0.1) ⭐", m))

    # run 5: overfit test
    rid, m = run_experiment(
        "Run 5 - XGBoost Overfit",
        XGBRegressor(n_estimators=700, max_depth=10, learning_rate=0.05,
                     subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1, verbosity=0),
        "xgboost_overfit", {"n_estimators": 700, "max_depth": 10, "learning_rate": 0.05,
                            "subsample": 0.7, "colsample_bytree": 0.7},
        X_train, X_test, y_train, y_test, feature_cols)
    results.append(("Run 5", "XGBoost Overfit (d=10, lr=0.05)", m))

    # comparison
    log(f"\n{'='*80}")
    log("EXPERIMENT COMPARISON")
    log(f"{'='*80}")
    log(f"{'Run':<8} {'Model':<35} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time(s)':>8}")
    log("-" * 80)
    for run_label, model_name, m in results:
        log(f"{run_label:<8} {model_name:<35} {m['mae_test']:>8.4f} {m['rmse_test']:>8.4f} "
            f"{m['r2_test']:>8.4f} {m['training_time_seconds']:>8.2f}")
    log("-" * 80)
    log("⭐ Run 4 selected as production model")

    log(f"\n{'='*60}")
    log("ALL 5 EXPERIMENTS COMPLETE!")
    log(f"MLflow UI: mlflow ui --backend-store-uri {tracking_uri} --port 5000")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
