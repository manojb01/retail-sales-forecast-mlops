from datetime import datetime, timedelta
from airflow.sdk import dag, task
from airflow.providers.standard.operators.bash import BashOperator
import pandas as pd
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Add include path
sys.path.append("/usr/local/airflow/include")

from ml_models.train_models import ModelTrainer
from utils.mlflow_utils import MLflowManager
from data_validation.validators import DataValidator


default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 7, 22),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "catchup": False,
    "schedule": "@weekly",
}


@dag(
    default_args=default_args,
    description="Train sales forecasting models",
    tags=["ml", "training", "sales"],
)
def sales_forecast_training():
    @task()
    def extract_data_task():
        from utils.data_generator import RealisticSalesDataGenerator

        data_output_dir = "/tmp/sales_data"
        generator = RealisticSalesDataGenerator(
            start_date="2021-01-01", end_date="2021-12-31"
        )
        print("Generating realistic sales data...")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        print(f"Generated {total_files} files:")
        for data_type, paths in file_paths.items():
            print(f"  - {data_type}: {len(paths)} files")
        return {
            "data_output_dir": data_output_dir,
            "file_paths": file_paths,
            "total_files": total_files,
        }

    @task()
    def validate_data_task(extract_result):
        import glob

        file_paths = extract_result["file_paths"]
        total_rows = 0
        issues_found = []
        print(f"Validating {len(file_paths['sales'])} sales files...")
        for i, sales_file in enumerate(file_paths["sales"][:10]):
            df = pd.read_parquet(sales_file)
            if i == 0:
                print(f"Sales data columns: {df.columns.tolist()}")
            if df.empty:
                issues_found.append(f"Empty file: {sales_file}")
                continue
            required_cols = [
                "date",
                "store_id",
                "product_id",
                "quantity_sold",
                "revenue",
            ]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues_found.append(f"Missing columns in {sales_file}: {missing_cols}")
            total_rows += len(df)
            if df["quantity_sold"].min() < 0:
                issues_found.append(f"Negative quantities in {sales_file}")
            if df["revenue"].min() < 0:
                issues_found.append(f"Negative revenue in {sales_file}")
        for data_type in ["promotions", "store_events", "customer_traffic"]:
            if data_type in file_paths and file_paths[data_type]:
                sample_file = file_paths[data_type][0]
                df = pd.read_parquet(sample_file)
                print(f"{data_type} data shape: {df.shape}")
                print(f"{data_type} columns: {df.columns.tolist()}")
        validation_summary = {
            "total_files_validated": len(file_paths["sales"][:10]),
            "total_rows": total_rows,
            "issues_found": len(issues_found),
            "issues": issues_found[:5],
        }
        if issues_found:
            print(f"Validation completed with {len(issues_found)} issues:")
            for issue in issues_found[:5]:
                print(f"  - {issue}")
        else:
            print(f"Validation passed! Total rows: {total_rows}")
        return validation_summary

    @task()
    def train_models_task(extract_result, validation_summary):
        file_paths = extract_result["file_paths"]
        print("Loading sales data from multiple files...")
        sales_dfs = []
        # Load ALL available files for maximum training data
        # Changed from max_files=50 to load all 365 days of data
        max_files = len(file_paths["sales"])
        for i, sales_file in enumerate(file_paths["sales"][:max_files]):
            df = pd.read_parquet(sales_file)
            sales_dfs.append(df)
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1} files...")
        sales_df = pd.concat(sales_dfs, ignore_index=True)
        print(f"Combined sales data shape: {sales_df.shape}")
        daily_sales = (
            sales_df.groupby(["date", "store_id", "product_id", "category"])
            .agg(
                {
                    "quantity_sold": "sum",
                    "revenue": "sum",
                    "cost": "sum",
                    "profit": "sum",
                    "discount_percent": "mean",
                    "unit_price": "mean",
                }
            )
            .reset_index()
        )
        daily_sales = daily_sales.rename(columns={"revenue": "sales"})
        if file_paths.get("promotions"):
            promo_df = pd.read_parquet(file_paths["promotions"][0])
            promo_summary = (
                promo_df.groupby(["date", "product_id"])["discount_percent"]
                .max()
                .reset_index()
            )
            promo_summary["has_promotion"] = 1
            daily_sales = daily_sales.merge(
                promo_summary[["date", "product_id", "has_promotion"]],
                on=["date", "product_id"],
                how="left",
            )
            daily_sales["has_promotion"] = daily_sales["has_promotion"].fillna(0)
        if file_paths.get("customer_traffic"):
            traffic_dfs = []
            for traffic_file in file_paths["customer_traffic"]:
                traffic_dfs.append(pd.read_parquet(traffic_file))
            traffic_df = pd.concat(traffic_dfs, ignore_index=True)
            traffic_summary = (
                traffic_df.groupby(["date", "store_id"])
                .agg({"customer_traffic": "sum", "is_holiday": "max"})
                .reset_index()
            )
            daily_sales = daily_sales.merge(
                traffic_summary, on=["date", "store_id"], how="left"
            )
        print(f"Final training data shape: {daily_sales.shape}")
        print(f"Columns: {daily_sales.columns.tolist()}")
        trainer = ModelTrainer()
        store_daily_sales = (
            daily_sales.groupby(["date", "store_id"])
            .agg(
                {
                    "sales": "sum",
                    "quantity_sold": "sum",
                    "profit": "sum",
                    "has_promotion": "mean",
                    "customer_traffic": "first",
                    "is_holiday": "first",
                }
            )
            .reset_index()
        )
        store_daily_sales["date"] = pd.to_datetime(store_daily_sales["date"])
        train_df, val_df, test_df = trainer.prepare_data(
            store_daily_sales,
            target_col="sales",
            date_col="date",
            group_cols=["store_id"],
            categorical_cols=["store_id"],
        )
        print(
            f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
        )
        
        # DEBUG: Save train/val/test data as CSVs for analysis
        debug_data_dir = "/usr/local/airflow/data"
        os.makedirs(debug_data_dir, exist_ok=True)
        
        train_df.to_csv(f"{debug_data_dir}/train_data.csv", index=False)
        val_df.to_csv(f"{debug_data_dir}/val_data.csv", index=False)
        test_df.to_csv(f"{debug_data_dir}/test_data.csv", index=False)
        
        # Also save raw store_daily_sales before feature engineering
        store_daily_sales.to_csv(f"{debug_data_dir}/raw_store_daily_sales.csv", index=False)
        
        print(f"DEBUG: Saved training data CSVs to {debug_data_dir}/")
        print(f"  - train_data.csv: {len(train_df)} rows")
        print(f"  - val_data.csv: {len(val_df)} rows")
        print(f"  - test_data.csv: {len(test_df)} rows")
        print(f"  - raw_store_daily_sales.csv: {len(store_daily_sales)} rows")
        print(f"Train data sample (first 3 rows):")
        print(train_df.head(3).to_string())
        print(f"Train 'sales' column stats: min={train_df['sales'].min():.2f}, max={train_df['sales'].max():.2f}, mean={train_df['sales'].mean():.2f}")
        
        results = trainer.train_all_models(
            train_df, val_df, test_df, target_col="sales", use_optuna=True
        )
        for model_name, model_results in results.items():
            if "metrics" in model_results:
                print(f"\n{model_name} metrics:")
                for metric, value in model_results["metrics"].items():
                    print(f"  {metric}: {value:.4f}")
        print("\nVisualization charts have been generated and saved to MLflow/MinIO")
        print("Charts include:")
        print("  - Model metrics comparison")
        print("  - Predictions vs actual values")
        print("  - Residuals analysis")
        print("  - Error distribution")
        print("  - Feature importance comparison")
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                "metrics": model_results.get("metrics", {})
            }
        import mlflow

        current_run_id = (
            mlflow.active_run().info.run_id if mlflow.active_run() else None
        )
        return {
            "training_results": serializable_results,
            "mlflow_run_id": current_run_id,
        }

    @task()
    def evaluate_models_task(training_result):
        results = training_result["training_results"]
        mlflow_manager = MLflowManager()
        best_model_name = None
        best_rmse = float("inf")
        for model_name, model_results in results.items():
            if "metrics" in model_results and "rmse" in model_results["metrics"]:
                if model_results["metrics"]["rmse"] < best_rmse:
                    best_rmse = model_results["metrics"]["rmse"]
                    best_model_name = model_name
        logger.info(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")
        best_run = mlflow_manager.get_best_model(metric="rmse", ascending=True)
        return {"best_model": best_model_name, "best_run_id": best_run["run_id"]}

    @task()
    def register_models_task(evaluation_result, training_result):
        # Register ALL models for tracking (Champion/Challenger pattern)
        # But we will only promote the Champion to Production later
        best_run_id = evaluation_result["best_run_id"]
        results = training_result["training_results"]
        
        mlflow_manager = MLflowManager()
        model_versions = {}
        
        logger.info(f"Registering all models from run {best_run_id}")
        
        # Determine all available models from the training results
        available_models = [m for m in results.keys() if m in ["xgboost", "lightgbm", "ensemble"]]
        
        for model_name in available_models:
            version = mlflow_manager.register_model(best_run_id, model_name, model_name)
            model_versions[model_name] = version
            logger.info(f"Registered {model_name} version: {version}")
            
        return model_versions

    @task()
    def transition_to_production_task(model_versions, evaluation_result):
        # Only promote the BEST model (Champion) to Production
        best_model_name = evaluation_result["best_model"]
        mlflow_manager = MLflowManager()
        
        if best_model_name in model_versions:
            version = model_versions[best_model_name]
            mlflow_manager.transition_model_stage(best_model_name, version, "Production")
            logger.info(f"🏆 Champion Model: {best_model_name} v{version}")
            logger.info(f"Transitioned {best_model_name} v{version} to Production")
        else:
            logger.info(f"Warning: Best model {best_model_name} was not found in registered models.")
            
        return f"Champion model {best_model_name} transitioned to production"

    @task()
    def generate_performance_report_task(training_result, validation_summary):
        results = training_result["training_results"]
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_rows": (
                    validation_summary.get("total_rows", 0) if validation_summary else 0
                ),
                "files_validated": (
                    validation_summary.get("total_files_validated", 0)
                    if validation_summary
                    else 0
                ),
                "issues_found": (
                    validation_summary.get("issues_found", 0)
                    if validation_summary
                    else 0
                ),
                "issues": (
                    validation_summary.get("issues", []) if validation_summary else []
                ),
            },
            "model_performance": {},
        }
        if results:
            for model_name, model_results in results.items():
                if "metrics" in model_results:
                    report["model_performance"][model_name] = model_results["metrics"]
        import json

        with open("/tmp/performance_report.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Performance report generated")
        logger.info(f"Models trained: {list(report['model_performance'].keys())}")
        return report

    # Task dependencies using function calls
    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)
    training_result = train_models_task(extract_result, validation_summary)
    evaluation_result = evaluate_models_task(training_result)
    model_versions = register_models_task(evaluation_result, training_result)
    transition = transition_to_production_task(model_versions, evaluation_result)
    report = generate_performance_report_task(training_result, validation_summary)
    cleanup = BashOperator(
        task_id="cleanup",
        bash_command="rm -rf /tmp/sales_data /tmp/performance_report.json || true",
    )
    report >> cleanup


sales_forecast_training_dag = sales_forecast_training()
