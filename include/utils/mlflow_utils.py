import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib
from .service_discovery import get_mlflow_endpoint, get_minio_endpoint

logger = logging.getLogger(__name__)


class MLflowManager:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        mlflow_config = self.config['mlflow']
        # Use service discovery to get tracking URI
        self.tracking_uri = get_mlflow_endpoint()
        
        self.experiment_name = mlflow_config['experiment_name']
        self.registry_name = mlflow_config['registry_name']
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Try to create experiment, with fallback
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set experiment {self.experiment_name}: {e}")
            # Try with localhost if initial connection failed
            if 'mlflow' in self.tracking_uri:
                self.tracking_uri = "http://localhost:5001"
                mlflow.set_tracking_uri(self.tracking_uri)
                os.environ['MLFLOW_TRACKING_URI'] = self.tracking_uri
                logger.info(f"Retrying with localhost: {self.tracking_uri}")
                try:
                    mlflow.set_experiment(self.experiment_name)
                except Exception as e2:
                    logger.error(f"Failed to connect to MLflow: {e2}")
        
        # Configure S3 endpoint for MinIO using service discovery
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = get_minio_endpoint()
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, model_name: str, input_example: Optional[pd.DataFrame] = None,
                  signature: Optional[Any] = None, registered_model_name: Optional[str] = None):
        """
        Log model to MLflow with compatibility for different versions.
        Falls back to saving models as artifacts if MLflow model logging fails.
        """
        try:
            # Save model to a temporary file first
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{model_name}_model.pkl")
                joblib.dump(model, model_path)
                
                # Log as artifact
                mlflow.log_artifact(model_path, artifact_path=f"models/{model_name}")
                logger.info(f"Successfully saved {model_name} model as artifact")
                
                # Also save metadata
                metadata = {
                    "model_type": model_name,
                    "framework": type(model).__module__,
                    "class": type(model).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                metadata_path = os.path.join(tmpdir, f"{model_name}_metadata.yaml")
                with open(metadata_path, 'w') as f:
                    yaml.dump(metadata, f)
                mlflow.log_artifact(metadata_path, artifact_path=f"models/{model_name}")
                
        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {e}")
            # Don't fail the entire run, just log the error
    
    def log_artifacts(self, artifact_path: str):
        mlflow.log_artifacts(artifact_path)
    
    def log_figure(self, figure, artifact_file: str):
        mlflow.log_figure(figure, artifact_file)
    
    def end_run(self, status: str = "FINISHED"):
        # Get run ID before ending
        run = mlflow.active_run()
        run_id = run.info.run_id if run else None
        
        mlflow.end_run(status=status)
        logger.info("Ended MLflow run")
        
        # Sync artifacts to S3 after run ends
        if run_id and status == "FINISHED":
            try:
                from include.utils.mlflow_s3_utils import MLflowS3Manager
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(run_id)
                logger.info(f"Synced artifacts to S3 for run {run_id}")
            except Exception as e:
                logger.warning(f"Failed to sync artifacts to S3: {e}")
    
    def get_best_model(self, metric: str = "rmse", ascending: bool = True) -> Dict[str, Any]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if len(runs) == 0:
            raise ValueError("No runs found in the experiment")
        
        best_run = runs.iloc[0]
        return {
            "run_id": best_run["run_id"],
            "metrics": {col.replace("metrics.", ""): val 
                       for col, val in best_run.items() 
                       if col.startswith("metrics.")},
            "params": {col.replace("params.", ""): val 
                      for col, val in best_run.items() 
                      if col.startswith("params.")}
        }
    
    def load_model(self, model_uri: str):
        """Load model from MLflow or from artifacts"""
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.debug(f"Could not load as pyfunc model: {e}")
            # Try loading from artifacts
            if "runs:/" in model_uri:
                run_id = model_uri.split("/")[1]
                artifact_path = "/".join(model_uri.split("/")[2:])
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=f"{artifact_path}_model.pkl"
                )
                return joblib.load(local_path)
            else:
                raise ValueError(f"Cannot load model from {model_uri}")
    
    def register_model(self, run_id: str, model_name: str, artifact_path: str) -> str:
        """Register model with MLflow Model Registry"""
        full_model_name = f"{self.registry_name}_{model_name}"
        model_uri = f"runs:/{run_id}/models/{artifact_path}"

        try:
            # First, ensure the registered model exists
            try:
                self.client.get_registered_model(full_model_name)
                logger.info(f"Registered model '{full_model_name}' already exists")
            except Exception:
                # Create the registered model if it doesn't exist
                self.client.create_registered_model(full_model_name)
                logger.info(f"Created registered model '{full_model_name}'")

            # Create a new model version
            model_version = self.client.create_model_version(
                name=full_model_name,
                source=model_uri,
                run_id=run_id
            )
            logger.info(f"Registered {model_name} as version {model_version.version}")
            return model_version.version

        except Exception as e:
            logger.warning(f"Model registration failed for {model_name}: {e}")
            logger.warning(f"Using run_id as version fallback")
            return run_id

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """Transition model to a stage (Production, Staging, Archived, None)"""
        full_model_name = f"{self.registry_name}_{model_name}"

        try:
            # Check if version is a run_id (fallback case) or actual version number
            if len(version) > 10:  # run_id is typically 32 chars
                logger.warning(f"Version '{version}' appears to be a run_id, not a version number")
                logger.warning(f"Skipping stage transition for {model_name}")
                return

            # EXPLICITLY archive all existing Production versions first
            if stage == "Production":
                try:
                    # Get all versions currently in Production
                    filter_string = f"name='{full_model_name}'"
                    all_versions = self.client.search_model_versions(filter_string)
                    
                    for v in all_versions:
                        if v.current_stage == "Production" and v.version != version:
                            logger.info(f"Archiving old Production version: {model_name} v{v.version}")
                            self.client.transition_model_version_stage(
                                name=full_model_name,
                                version=v.version,
                                stage="Archived"
                            )
                except Exception as e:
                    logger.warning(f"Could not archive existing Production versions: {e}")

            # 1. Set alias for newer MLflow versions (recommended best practice)
            try:
                alias_name = stage.lower()  # e.g., "production", "staging"
                self.client.set_registered_model_alias(full_model_name, alias_name, version)
                logger.info(f"Set alias '{alias_name}' for {model_name} v{version}")
            except (AttributeError, Exception) as e:
                logger.debug(f"Could not set alias for {model_name}: {e}")

            # 2. Transition to the new stage
            try:
                self.client.transition_model_version_stage(
                    name=full_model_name,
                    version=version,
                    stage=stage,
                    archive_existing_versions=False  # We already archived manually above
                )
                logger.info(f"✅ Transitioned {model_name} v{version} to {stage}")
            except Exception as e:
                logger.warning(f"Stage transition failed: {e}")
                raise e

        except Exception as e:
            logger.warning(f"Model stage transition failed for {model_name}: {e}")

    def get_latest_model_version(self, model_name: str, stage: Optional[str] = None) -> Dict[str, Any]:
        """Get the latest model version, optionally filtered by stage"""
        full_model_name = f"{self.registry_name}_{model_name}"

        try:
            # Search for model versions
            filter_string = f"name='{full_model_name}'"
            versions = self.client.search_model_versions(filter_string)

            if not versions:
                raise ValueError(f"No model versions found for {model_name}")

            # Filter by stage if specified
            if stage:
                versions = [v for v in versions if v.current_stage == stage]
                if not versions:
                    raise ValueError(f"No model versions found for {model_name} in stage {stage}")

            latest_version = max(versions, key=lambda x: int(x.version))
            return {
                "version": latest_version.version,
                "stage": latest_version.current_stage,
                "run_id": latest_version.run_id,
                "source": latest_version.source
            }
        except Exception as e:
            logger.warning(f"Could not get model version: {e}")
            # Fallback to finding the best run
            best_model = self.get_best_model()
            return {
                "version": best_model["run_id"],
                "stage": "None",
                "run_id": best_model["run_id"],
                "source": f"runs:/{best_model['run_id']}/models"
            }