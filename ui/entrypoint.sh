#!/bin/bash

# Install system dependencies for LightGBM
apt-get update && apt-get install -y libgomp1

# Install dependencies - MATCH SCHEDULER VERSIONS EXACTLY
pip install streamlit==1.32.2 pandas==2.1.4 numpy==1.26.3 plotly==5.19.0 mlflow==2.9.2 scikit-learn==1.5.2 xgboost==3.1.3 lightgbm==4.3.0 joblib==1.3.2 boto3==1.34.25 python-dateutil>=2.9.0 dill==0.3.7 holidays==0.89 pyyaml

# Run Streamlit app
streamlit run app.py --server.address 0.0.0.0 --server.port 8501