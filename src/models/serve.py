import bentoml
import mlflow.pyfunc
import pandas as pd

@bentoml.service(name="fraud-detector", resources={"cpu": "1"})
class FraudDetector:
    def __init__(self):
        # Load best model from MLflow Model Registry
        # Adjust this URI if you registered the model with a different name/stage
        model_uri = "models:/fraud_detector/Production"
        self.model = mlflow.pyfunc.load_model(model_uri)

    @bentoml.api
    def predict(self, transaction: dict) -> dict:
        """
        transaction: dict of features
        Example:
        {
          "V1": -1.3598071336738,
          "V2": -0.0727811733098497,
          "V3": 2.53634673796914,
          "Amount": 149.62
        }
        """
        df = pd.DataFrame([transaction])
        prediction = self.model.predict(df)

        # convert numpy/pandas output to python list for JSON response
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()

        return {"prediction": prediction}