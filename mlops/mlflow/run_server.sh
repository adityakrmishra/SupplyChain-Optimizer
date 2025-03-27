#!/bin/bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow server \
    --backend-store-uri postgresql://mlflow:password@localhost/mlflowdb \
    --default-artifact-root s3://supplychain-ml-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 4
