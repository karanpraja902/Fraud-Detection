import bentoml
from bentoml.io import JSON
import mlflow


# Load the latest MLflow model (version 4) from local mlruns artifacts
model_name = "fraud-detector"
model_version = 4
mlflow_model_uri = f"runs:/{model_name}/{model_version}"

# MLflow local path for the latest model
mlflow_model_uri = "mlruns/902899545813332380/5509992e7a08492db5bdaecc787a3d52/artifacts/best_model"

# Import MLflow model into BentoML
bentoml_model = bentoml.mlflow.import_model(
    name=model_name,
    model_uri=mlflow_model_uri,
)

# Create a BentoML runner for the imported model
runner = bentoml_model.to_runner()

# Define the BentoML service
svc = bentoml.Service(name="fraud_detector_service", runners=[runner])


@svc.api(input=JSON(), output=JSON())
async def predict(input_json):
    """
    Expects input_json as a dictionary with features for fraud detection.
    Returns fraud probabilities.
    """
    # The model expects a list of records; wrap input in a list if it's a single record
    if isinstance(input_json, dict):
        input_data = [input_json]
    else:
        input_data = input_json

    preds = await runner.predict.async_run(input_data)
    return {"fraud_probabilities": preds}


if __name__ == "__main__":
    # Save the BentoML service to generate build artifacts
    saved_path = svc.save()
    print(f"BentoML service saved to: {saved_path}")
