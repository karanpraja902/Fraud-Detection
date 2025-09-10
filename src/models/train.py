import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

mlflow.set_experiment("Fraud Detection")

def train_model(train_path: str, model_name: str = "RandomForest"):
    """Trains a model and logs it to MLflow."""
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']

    with mlflow.start_run() as run:
        # Define and train model
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        rfc = RandomForestClassifier(**params)
        rfc.fit(X_train, y_train)

        # Log parameters and metrics
        mlflow.log_params(params)
        
        # Note: Evaluate on a validation set in a real scenario
        y_pred_proba = rfc.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, y_pred_proba)
        mlflow.log_metric("train_auc", auc)

        # Log the model with a registered model name
        mlflow.sklearn.log_model(
            sk_model=rfc,
            artifact_path="model",
            registered_model_name="fraud-detector"
        )
        print(f"Model logged with run_id: {run.info.run_id}")