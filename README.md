# Retail Sales Forecasting Platform

An end-to-end MLOps platform for multi-store retail sales forecasting, built on Astronomer (Apache Airflow), MLflow, MinIO, and Streamlit.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Apache Airflow                           │
│  generate_data → validate → feature_engineering → train →      │
│  evaluate → register_model → promote_to_production             │
└────────────────────────────┬────────────────────────────────────┘
                             │ logs metrics & artifacts
                    ┌────────▼────────┐
                    │     MLflow      │  ← model registry + experiment tracking
                    │  + MinIO (S3)   │  ← artifact storage
                    └────────┬────────┘
                             │ loads production model
                    ┌────────▼────────┐
                    │  Streamlit UI   │  ← live inference + model insights
                    └─────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | Astronomer (Airflow 2.x) | Pipeline scheduling and monitoring |
| ML Models | XGBoost + LightGBM Ensemble | Gradient boosting ensemble |
| Hyperparameter Tuning | Optuna (50 trials) | Automated HPO |
| Experiment Tracking | MLflow 2.9 | Metrics, artifacts, model registry |
| Artifact Storage | MinIO (S3-compatible) | Model and data artifact storage |
| Database | PostgreSQL | MLflow backend store |
| Inference UI | Streamlit | Interactive forecasting dashboard |
| Containerization | Docker + Docker Compose | Reproducible environments |

## Features

- **Automated ML Pipeline**: 7-task Airflow DAG from data generation to model promotion
- **XGBoost + LightGBM Ensemble**: Weighted average ensemble with Optuna hyperparameter tuning (50 trials each)
- **50 Engineered Features**: Date features, lag features (7 lags), rolling statistics (25 features), cyclical encodings
- **Iterative Forecasting**: Day-by-day prediction feeding predictions back as lag features — avoids data leakage
- **Model Registry**: MLflow `production` alias for zero-downtime model promotion
- **Live Inference UI**: Streamlit dashboard with real-time metrics pulled from MLflow (R², MAPE, RMSE, MAE)
- **Confidence Intervals**: Historical-volatility-based prediction bands

## Quick Start

### Prerequisites

- Docker Desktop (8 GB+ RAM allocated)
- [Astronomer CLI](https://www.astronomer.io/docs/astro/cli/install-cli/) (`brew install astro` on macOS)
- Ports 8080, 5001, 8501, 9000, 9001 available

### 1. Clone and configure

```bash
git clone <repo-url>
cd sales_forecast

cp .env.example .env   # review defaults (no changes needed for local dev)
```

### 2. Start all services

```bash
astro dev start
```

This starts:
- **Airflow UI**: http://localhost:8080 (admin / admin)
- **MLflow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001 (minioadmin / minioadmin)
- **Streamlit UI**: http://localhost:8501

> **Network note**: Astronomer generates a project-specific Docker network name. If services can't reach each other, update the network names in `docker-compose.override.yml`. See the comment at the bottom of that file.

### 3. Run the training pipeline

1. Open Airflow UI → enable `sales_forecast_training` DAG → trigger manually
2. Monitor the 7-task pipeline (~3-5 minutes)
3. Open Streamlit at http://localhost:8501 to run forecasts

## Project Structure

```
sales_forecast/
├── dags/
│   └── sales_forecast_training.py   # Main Airflow DAG (7 tasks)
├── include/
│   ├── config/
│   │   └── ml_config.yaml           # Feature and model configuration
│   ├── data_validation/
│   │   └── validators.py            # Data quality checks
│   ├── feature_engineering/
│   │   └── feature_pipeline.py      # 50-feature engineering pipeline
│   ├── ml_models/
│   │   ├── train_models.py          # XGBoost + LightGBM training with Optuna
│   │   ├── ensemble_model.py        # Ensemble logic and weighting
│   │   ├── advanced_ensemble.py     # Advanced ensemble strategies
│   │   ├── diagnostics.py           # Model diagnostics
│   │   └── model_visualization.py  # Training charts and reports
│   └── utils/
│       ├── data_generator.py        # Synthetic multi-store data generation
│       ├── mlflow_utils.py          # MLflow helpers
│       └── mlflow_s3_utils.py       # MinIO/S3 artifact helpers
├── ui/
│   ├── inference_app_v2.py          # Streamlit inference application
│   ├── utils/
│   │   └── improved_predictor.py   # Iterative forecasting engine
│   ├── requirements.txt
│   └── entrypoint.sh
├── tests/
│   ├── dags/                        # DAG integrity tests
│   └── test_feature_engineering.py
├── docker-compose.override.yml      # Custom services (MLflow, MinIO, Streamlit)
├── .env.example                     # Environment variable template
├── Dockerfile                       # Airflow scheduler image
└── requirements.txt                 # Airflow Python dependencies
```

## ML Pipeline Details

### DAG Tasks

1. **`generate_data`** — Synthetic multi-store daily sales with seasonality, promotions, holidays
2. **`validate_data`** — Schema validation, outlier detection, completeness checks
3. **`feature_engineering`** — 50 features: date, lag (1/7/14/21/28/35/42 days), rolling stats, cyclical
4. **`train_models`** — XGBoost and LightGBM with Optuna HPO (50 trials each), logged to MLflow
5. **`evaluate_models`** — R², MAPE, RMSE, MAE on held-out test set
6. **`register_model`** — Best ensemble saved to MLflow Model Registry
7. **`promote_model`** — Assign `production` alias; Streamlit picks this up automatically

### Ensemble Strategy

The ensemble weights are optimized on the validation set to minimize RMSE:

```
ensemble_prediction = w₁ × xgboost + w₂ × lightgbm
where w₁ + w₂ = 1, weights optimized per validation performance
```

### Forecasting Approach

Standard batch prediction would set future lag features to zero, producing flat/zero forecasts. Instead, the `ImprovedPredictor` uses **iterative forecasting**: predict day 1 → append prediction to history → predict day 2 using day 1's prediction as lag feature → repeat.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Services can't resolve each other (DNS error) | Update network names in `docker-compose.override.yml` to match `docker network ls` output |
| Streamlit shows "No production model found" | Run the Airflow DAG first to train and register a model |
| Docker memory issues | Allocate 8+ GB to Docker Desktop in Settings → Resources |
| Port conflicts | Modify port mappings in `docker-compose.override.yml` |

```bash
# Check service logs
astro dev logs
docker compose -f docker-compose.override.yml logs mlflow
```

## License

MIT
