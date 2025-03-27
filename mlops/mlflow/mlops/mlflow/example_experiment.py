import mlflow
from mlflow.tracking import MlflowClient

def track_experiment():
    mlflow.set_experiment("supplychain-demand-forecast")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("horizon_days", 30)
        
        # Log metrics
        mlflow.log_metric("mae", 12.4)
        mlflow.log_metric("rmse", 15.7)
        
        # Log artifacts
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact("model.pkl")
        
        # Register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="prophet-model",
            registered_model_name="demand-forecaster"
        )
