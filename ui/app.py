import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import mlflow
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shutil
import uuid

# Add include path
sys.path.append("/usr/local/airflow/include")
sys.path.append("/app/utils")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.predictor import ImprovedPredictor

sys.path.insert(0, "/usr/local/airflow/include/utils")
from data_generator import RealisticSalesDataGenerator

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ['model_loaded', 'model', 'scalers', 'encoders', 'feature_cols', 'model_info']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'model_loaded' else (None if key != 'model_info' else {})


# ── Model loading ─────────────────────────────────────────────────────────────
def load_production_model():
    try:
        with st.spinner("Loading production model from MLflow..."):
            client = mlflow.MlflowClient()
            model_name = "sales_forecast_models_ensemble"

            try:
                model_version = client.get_model_version_by_alias(model_name, "production")
            except Exception:
                versions = client.search_model_versions(f"name='{model_name}'")
                prod_versions = [v for v in versions if v.current_stage == "Production"]
                if not prod_versions:
                    return False, f"No production version found for {model_name}"
                model_version = prod_versions[0]

            run_id = model_version.run_id
            logger.info(f"Loading production model from run {run_id}")

            model_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/models/ensemble/ensemble_model.pkl")
            model = joblib.load(model_path)

            scalers_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/scalers.pkl")
            encoders_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/encoders.pkl")
            features_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/feature_cols.pkl")

            scalers = joblib.load(scalers_path)
            encoders = joblib.load(encoders_path)
            feature_cols = joblib.load(features_path)

            # Pull metrics and params from the MLflow run
            run = client.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params

            st.session_state.model = model
            st.session_state.scalers = scalers
            st.session_state.encoders = encoders
            st.session_state.feature_cols = feature_cols
            st.session_state.model_loaded = True
            st.session_state.model_info = {
                'name': model_name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': run_id,
                'metrics': metrics,
                'params': params,
                'loaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return True, f"Model loaded successfully! (run: {run_id[:8]}...)"

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False, f"Error loading model: {str(e)}"


# ── Forecast runner ───────────────────────────────────────────────────────────
def run_forecast(historical_data, forecast_days):
    try:
        predictor = ImprovedPredictor(
            model=st.session_state.model,
            scalers=st.session_state.scalers,
            encoders=st.session_state.encoders,
            feature_cols=st.session_state.feature_cols
        )
        predictions_df, error = predictor.predict_iterative(
            historical_data=historical_data,
            forecast_days=forecast_days,
            include_confidence=True
        )
        if error:
            return None, error
        return predictions_df, None
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=60)
st.sidebar.title("Sales Forecast Platform")
st.sidebar.caption("MLOps · XGBoost · LightGBM · Airflow · MLflow")
st.sidebar.markdown("---")
st.sidebar.header("Model Management")

if st.sidebar.button("🔄 Load Production Model", type="primary", use_container_width=True):
    success, message = load_production_model()
    if success:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

if st.session_state.model_loaded:
    info = st.session_state.model_info
    st.sidebar.success("✅ Model Ready")
    st.sidebar.markdown("### Active Model")
    st.sidebar.write(f"**Model:** `{info['name']}`")
    st.sidebar.write(f"**Version:** v{info['version']} · {info['stage']}")
    st.sidebar.write(f"**Run:** `{info.get('run_id','')[:8]}...`")
    st.sidebar.write(f"**Loaded:** {info['loaded_at']}")

    m = info.get('metrics', {})
    if m:
        st.sidebar.markdown("### Live Metrics")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("R²", f"{m.get('ensemble_r2', 0):.3f}")
        col2.metric("MAPE", f"{m.get('ensemble_mape', 0):.1f}%")
        col1.metric("RMSE", f"${m.get('ensemble_rmse', 0):.1f}")
        col2.metric("MAE", f"${m.get('ensemble_mae', 0):.1f}")
else:
    st.sidebar.warning("⚠️ No model loaded")
    st.sidebar.info("Click **Load Production Model** to begin")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Stack:** Airflow · MLflow · MinIO · XGBoost · LightGBM · Streamlit\n\n"
    "**Data:** 10 stores · 20 SKUs · 4 categories · 2021"
)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📈 Retail Sales Forecasting Platform")
st.caption("End-to-end MLOps pipeline · Ensemble model (XGBoost + LightGBM) · Automated retraining via Apache Airflow")
st.markdown("---")

if not st.session_state.model_loaded:
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1**\nClick **Load Production Model** in the sidebar to fetch the latest registered model from MLflow.")
    col2.info("**Step 2**\nGenerate synthetic historical sales data using the same logic as the training pipeline.")
    col3.info("**Step 3**\nRun the iterative day-by-day ensemble forecast with 95% confidence intervals.")
    st.stop()

st.success("✅ Production model loaded and ready for predictions!")

with st.expander("📋 Model Details", expanded=False):
    info = st.session_state.model_info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", info['name'].replace("sales_forecast_models_", ""))
    c2.metric("Version", f"v{info['version']}")
    c3.metric("Stage", info['stage'])
    c4.metric("Features", len(st.session_state.feature_cols))

tab1, tab2 = st.tabs(["📊 Batch Prediction", "📈 Model Insights"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Batch Prediction")

    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = None
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None

    st.subheader("⚙️ Configuration")
    forecast_days = st.slider(
        "Forecast horizon (days)",
        min_value=30, max_value=90, value=30, step=5,
        help="Number of future days to predict"
    )
    st.markdown("---")

    # ── Step 1: Generate data ──────────────────────────────────────────────────
    st.subheader("1️⃣ Generate Historical Sample Data")
    st.info("Generates synthetic sales data using the same `RealisticSalesDataGenerator` as the training pipeline — ensuring consistent feature distributions at inference time.")

    col1, col2 = st.columns([1, 3])
    with col1:
        sample_days = st.number_input("Historical Days", value=60, min_value=30, max_value=365)

    if st.button("🎲 Generate Sample Data", type="primary"):
        with st.spinner("Generating data..."):
            temp_dir = os.path.join("/tmp", f"streamlit_gen_{uuid.uuid4().hex}")
            os.makedirs(temp_dir, exist_ok=True)
            try:
                end_date = datetime(2021, 10, 31)
                start_date = end_date - timedelta(days=sample_days)

                generator = RealisticSalesDataGenerator(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                file_paths = generator.generate_sales_data(output_dir=temp_dir)

                sales_dfs = []
                for f in file_paths.get('sales', []):
                    df = pd.read_parquet(f)
                    df = df[df['store_id'] == 'store_001']
                    if not df.empty:
                        sales_dfs.append(df)

                if not sales_dfs:
                    st.error("No data generated for store_001!")
                    st.stop()

                sales_df = pd.concat(sales_dfs, ignore_index=True)
                daily_sales = (
                    sales_df.groupby(["date", "store_id", "product_id", "category"])
                    .agg({"quantity_sold": "sum", "revenue": "sum", "cost": "sum",
                          "profit": "sum", "discount_percent": "mean", "unit_price": "mean"})
                    .reset_index()
                    .rename(columns={"revenue": "sales"})
                )

                if file_paths.get("promotions"):
                    promo_df = pd.read_parquet(file_paths["promotions"][0])
                    promo_summary = (promo_df.groupby(["date", "product_id"])["discount_percent"]
                                     .max().reset_index())
                    promo_summary["has_promotion"] = 1
                    daily_sales = daily_sales.merge(
                        promo_summary[["date", "product_id", "has_promotion"]],
                        on=["date", "product_id"], how="left"
                    )
                    daily_sales["has_promotion"] = daily_sales["has_promotion"].fillna(0)
                else:
                    daily_sales["has_promotion"] = 0

                if file_paths.get("customer_traffic"):
                    traffic_dfs = []
                    for tf in file_paths["customer_traffic"]:
                        tdf = pd.read_parquet(tf)
                        traffic_dfs.append(tdf[tdf['store_id'] == 'store_001'])
                    if traffic_dfs:
                        traffic_df = pd.concat(traffic_dfs, ignore_index=True)
                        traffic_summary = (
                            traffic_df.groupby(["date", "store_id"])
                            .agg({"customer_traffic": "sum", "is_holiday": "max"})
                            .reset_index()
                        )
                        daily_sales = daily_sales.merge(traffic_summary, on=["date", "store_id"], how="left")

                daily_sales['customer_traffic'] = daily_sales['customer_traffic'].fillna(0)
                daily_sales['is_holiday'] = daily_sales['is_holiday'].fillna(0)

                store_daily = (
                    daily_sales.groupby(["date", "store_id"])
                    .agg({"sales": "sum", "quantity_sold": "sum", "profit": "sum",
                          "has_promotion": "mean", "customer_traffic": "first", "is_holiday": "first"})
                    .reset_index()
                )
                store_daily["date"] = pd.to_datetime(store_daily["date"])
                st.session_state.sample_data = store_daily
                st.session_state.forecast_results = None
                st.success(f"✅ Generated {len(store_daily)} trading days for store_001 ({start_date.date()} → {end_date.date()})")
            except Exception as e:
                st.error(f"Error generating data: {e}")
                logger.error(f"Data generation error: {e}", exc_info=True)
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    # ── Data preview ──────────────────────────────────────────────────────────
    if st.session_state.sample_data is not None:
        st.markdown("---")
        st.subheader("📋 Historical Data Preview")

        data = st.session_state.sample_data
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Store", "store_001")
        c2.metric("Trading Days", len(data))
        c3.metric("Date Range", f"{data['date'].min().date()} → {data['date'].max().date()}")
        c4.metric("Avg Daily Sales", f"${data['sales'].mean():.2f}")

        st.dataframe(data.head(20), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("2️⃣ Run Forecast")
        st.info(f"Iterative day-by-day ensemble forecast — each prediction feeds back as lag features for the next day.")

        if st.button("🚀 Run Forecast", type="primary", key="run_forecast_btn"):
            with st.spinner(f"Running {forecast_days}-day ensemble forecast..."):
                results, error = run_forecast(st.session_state.sample_data, forecast_days)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state.forecast_results = results
                    st.success(f"✅ Forecast complete — {len(results)} predictions generated")

        # ── Results ───────────────────────────────────────────────────────────
        if st.session_state.forecast_results is not None:
            st.markdown("---")
            st.subheader("📈 Forecast Results")

            fdf = st.session_state.forecast_results
            hist = st.session_state.sample_data.copy()
            hist['date'] = pd.to_datetime(hist['date'])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Store", "store_001")
            c2.metric("Forecast Days", forecast_days)
            c3.metric("Total Forecast Revenue", f"${fdf['predicted_sales'].sum():,.2f}")
            delta = fdf['predicted_sales'].mean() - hist['sales'].mean()
            c4.metric("Avg Daily Forecast", f"${fdf['predicted_sales'].mean():.2f}",
                      delta=f"{delta:+.2f} vs historical")

            # Chart 1 — Historical
            st.subheader("Historical Sales — store_001")
            fig1 = go.Figure()
            hist_sorted = hist.sort_values('date')
            fig1.add_trace(go.Scatter(
                x=hist_sorted['date'], y=hist_sorted['sales'],
                mode='lines+markers', name='Historical Sales',
                line=dict(color='#1f77b4', width=2), marker=dict(size=4),
                fill='tozeroy', fillcolor='rgba(31,119,180,0.15)'
            ))
            fig1.update_layout(
                title=f"Historical Sales Data ({len(hist)} trading days)",
                xaxis_title="Date", yaxis_title="Daily Sales ($)",
                hovermode='x unified', height=400, showlegend=True,
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Chart 2 — Forecast with CI
            st.subheader(f"{forecast_days}-Day Sales Forecast — store_001")
            fdf_sorted = fdf.copy()
            fdf_sorted['date'] = pd.to_datetime(fdf_sorted['date'])
            fdf_sorted = fdf_sorted.sort_values('date')

            fig2 = go.Figure()
            if 'upper_bound' in fdf_sorted.columns and 'lower_bound' in fdf_sorted.columns:
                fig2.add_trace(go.Scatter(
                    x=fdf_sorted['date'], y=fdf_sorted['upper_bound'],
                    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig2.add_trace(go.Scatter(
                    x=fdf_sorted['date'], y=fdf_sorted['lower_bound'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(255,165,0,0.2)',
                    name='95% Confidence Interval', hoverinfo='skip'
                ))
            fig2.add_trace(go.Scatter(
                x=fdf_sorted['date'], y=fdf_sorted['predicted_sales'],
                mode='lines+markers', name='Predicted Sales',
                line=dict(color='#ff7f0e', width=2.5), marker=dict(size=5)
            ))
            fig2.update_layout(
                title=f"{forecast_days}-Day Forecast with 95% Confidence Interval",
                xaxis_title="Date", yaxis_title="Predicted Sales ($)",
                hovermode='x unified', height=400, showlegend=True,
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
            )
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("📋 View Detailed Forecast Data"):
                st.dataframe(fdf_sorted.sort_values('date'), use_container_width=True, hide_index=True)
                csv = fdf.to_csv(index=False)
                st.download_button(
                    "📥 Download Forecast CSV", data=csv,
                    file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    else:
        st.warning("⚠️ Generate sample data above to proceed.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Insights")
    st.caption("Performance metrics, feature importance, and model comparison from the latest MLflow training run.")

    info = st.session_state.model_info
    metrics = info.get('metrics', {})
    params = info.get('params', {})

    if not metrics:
        st.warning("Load the production model to see insights.")
        st.stop()

    # ── Section 1: Model performance ──────────────────────────────────────────
    st.subheader("🏆 Model Performance (Test Set)")

    model_names = ['xgboost', 'lightgbm', 'ensemble']
    display_names = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'ensemble': 'Ensemble ⭐'}
    metric_keys = ['rmse', 'mae', 'mape', 'r2']
    metric_labels = {'rmse': 'RMSE ($)', 'mae': 'MAE ($)', 'mape': 'MAPE (%)', 'r2': 'R²'}

    # Summary metrics row for ensemble
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ensemble R²", f"{metrics.get('ensemble_r2', 0):.4f}",
              help="Coefficient of determination — closer to 1.0 is better")
    c2.metric("Ensemble RMSE", f"${metrics.get('ensemble_rmse', 0):.2f}",
              help="Root Mean Squared Error")
    c3.metric("Ensemble MAE", f"${metrics.get('ensemble_mae', 0):.2f}",
              help="Mean Absolute Error")
    c4.metric("Ensemble MAPE", f"{metrics.get('ensemble_mape', 0):.2f}%",
              help="Mean Absolute Percentage Error")

    # Comparison table
    st.markdown("#### Model Comparison")
    rows = []
    for m in model_names:
        rows.append({
            'Model': display_names[m],
            'RMSE ($)': round(metrics.get(f'{m}_rmse', 0), 2),
            'MAE ($)': round(metrics.get(f'{m}_mae', 0), 2),
            'MAPE (%)': round(metrics.get(f'{m}_mape', 0), 2),
            'R²': round(metrics.get(f'{m}_r2', 0), 4),
        })
    comp_df = pd.DataFrame(rows).set_index('Model')

    def highlight_best(s):
        if s.name in ['RMSE ($)', 'MAE ($)', 'MAPE (%)']:
            best = s.min()
            return ['background-color: #d4edda; font-weight: bold' if v == best else '' for v in s]
        elif s.name == 'R²':
            best = s.max()
            return ['background-color: #d4edda; font-weight: bold' if v == best else '' for v in s]
        return ['' for _ in s]

    st.dataframe(comp_df.style.apply(highlight_best), use_container_width=True)

    # Visual comparison bar chart
    st.markdown("#### RMSE & MAE Comparison")
    fig_cmp = go.Figure()
    rmse_vals = [metrics.get(f'{m}_rmse', 0) for m in model_names]
    mae_vals = [metrics.get(f'{m}_mae', 0) for m in model_names]
    labels = [display_names[m] for m in model_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig_cmp.add_trace(go.Bar(name='RMSE ($)', x=labels, y=rmse_vals,
                              marker_color=colors, opacity=0.85))
    fig_cmp.add_trace(go.Bar(name='MAE ($)', x=labels, y=mae_vals,
                              marker_color=colors, opacity=0.5))
    fig_cmp.update_layout(barmode='group', height=350,
                           yaxis_title='Error ($)',
                           plot_bgcolor='white', paper_bgcolor='white',
                           xaxis=dict(showgrid=False),
                           yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")

    # ── Section 2: Feature importance ─────────────────────────────────────────
    st.subheader("🔍 Feature Importance (XGBoost)")

    fi_rows = []
    for k, v in params.items():
        if k.startswith('xgb_top_feature_'):
            try:
                idx = int(k.split('_')[-1])
                name, score_str = v.rsplit(' ', 1)
                score = float(score_str.strip('()'))
                fi_rows.append({'rank': idx, 'feature': name, 'importance': score})
            except Exception:
                pass

    if fi_rows:
        fi_df = pd.DataFrame(fi_rows).sort_values('rank').head(20)
        fi_df['importance_pct'] = (fi_df['importance'] * 100).round(2)

        fig_fi = go.Figure(go.Bar(
            x=fi_df['importance_pct'],
            y=fi_df['feature'],
            orientation='h',
            marker=dict(
                color=fi_df['importance_pct'],
                colorscale='Blues',
                showscale=False
            ),
            text=fi_df['importance_pct'].apply(lambda x: f"{x:.2f}%"),
            textposition='outside'
        ))
        fig_fi.update_layout(
            title="Top 20 Features by Importance",
            xaxis_title="Importance (%)",
            yaxis=dict(autorange='reversed'),
            height=600,
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            margin=dict(l=200)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Feature group breakdown
        st.markdown("#### Feature Group Breakdown")
        groups = {
            'Lag features': fi_df[fi_df['feature'].str.contains('lag')]['importance'].sum(),
            'Rolling features': fi_df[fi_df['feature'].str.contains('rolling')]['importance'].sum(),
            'Business features': fi_df[fi_df['feature'].str.contains('quantity|profit|traffic|promotion')]['importance'].sum(),
            'Date features': fi_df[fi_df['feature'].str.contains('month|day|week|year|quarter|holiday|weekend')]['importance'].sum(),
            'Cyclical features': fi_df[fi_df['feature'].str.contains('sin|cos')]['importance'].sum(),
        }
        group_df = pd.DataFrame([
            {'Group': k, 'Total Importance': round(v * 100, 2)}
            for k, v in groups.items() if v > 0
        ]).sort_values('Total Importance', ascending=False)

        fig_grp = px.pie(group_df, names='Group', values='Total Importance',
                          title='Feature Importance by Group',
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_grp.update_traces(textposition='inside', textinfo='percent+label')
        fig_grp.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_grp, use_container_width=True)
    else:
        st.info("Feature importance not available for this run.")

    st.markdown("---")

    # ── Section 3: Training info ───────────────────────────────────────────────
    st.subheader("📦 Training Configuration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", params.get('train_size', 'N/A'))
    c2.metric("Validation Samples", params.get('val_size', 'N/A'))
    c3.metric("Test Samples", params.get('test_size', 'N/A'))
    c4.metric("Total Features", params.get('n_features', len(st.session_state.feature_cols)))

    st.markdown("#### Pipeline Architecture")
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("""
        **Training Pipeline (Apache Airflow)**
        1. `extract_data` — Generate synthetic retail data
        2. `validate_data` — Schema & quality checks
        3. `train_models` — XGBoost + LightGBM w/ Optuna (50 trials)
        4. `evaluate_models` — RMSE, MAE, MAPE, R² on test set
        5. `register_models` — Push to MLflow Model Registry
        6. `promote_champion` — Tag best model as Production
        7. `generate_report` — HTML comparison report
        """)
    with arch_col2:
        st.markdown("""
        **Feature Engineering**
        - 8 date features (year, month, day, DOW, quarter, etc.)
        - 7 lag features (1, 2, 3, 7, 14, 21, 30 days)
        - 25 rolling features (windows 3/7/14/21/30 × mean/std/min/max/median)
        - 6 cyclical features (sin/cos for month, day, DOW)
        - Categorical encoding via OrdinalEncoder

        **Hyperparameter Tuning**
        - Optuna TPE sampler, 50 trials per model
        - Objective: minimize validation RMSE
        - Early stopping: 50 rounds
        """)

    st.markdown("---")

    # ── Section 4: Tech stack ──────────────────────────────────────────────────
    st.subheader("🛠️ Technology Stack")
    stack_cols = st.columns(4)
    stack_cols[0].markdown("**Orchestration**\n- Apache Airflow 3.0\n- Astronomer Runtime\n- DAG: `@weekly` schedule")
    stack_cols[1].markdown("**ML & Tuning**\n- XGBoost 3.x\n- LightGBM 4.x\n- Optuna (TPE sampler)")
    stack_cols[2].markdown("**MLOps**\n- MLflow 2.9 tracking\n- Model Registry\n- MinIO artifact store")
    stack_cols[3].markdown("**Infrastructure**\n- Docker Compose\n- PostgreSQL (metadata)\n- Redis (task broker)")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Retail Sales Forecasting Platform · XGBoost + LightGBM Ensemble · Apache Airflow · MLflow · MinIO · Streamlit")
