import bentoml
import joblib
import pandas as pd


@bentoml.service(name="fraud-detector", resources={"cpu": "1"})
class FraudDetector:
    def __init__(self):
        # Load model directly from the saved pickle file
        model_path = "/app/models/fraud-detector/model.pkl"
        print(f"Loading model from: {model_path}")
        try:
            self.model = joblib.load(model_path)
            print("Model loaded successfully!")
            print(f"Model type: {type(self.model)}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

        # Optimized threshold from best run
        self.threshold = 0.105

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
        proba = self.model.predict(df)

        # Apply threshold to convert probabilities to class predictions
        predictions = (proba >= self.threshold).astype(int)

        # convert numpy/pandas output to python list for JSON response
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        return {"prediction": predictions, "probability": proba.tolist()}
