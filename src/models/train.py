import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("data/processed/train.csv")
EXPERIMENT_NAME = "Fraud Detection"
RANDOM_STATE = 42
N_TRIALS = 20

mlflow.set_experiment(EXPERIMENT_NAME)


def load_data():
    """Load processed training data."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    X, y = load_data()

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]
        ),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,  # use all CPU cores
    }

    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        # Predictions
        y_val_proba = clf.predict_proba(X_val)[:, 1]
        y_val_pred = clf.predict(X_val)

        # Metrics
        val_auc = roc_auc_score(y_val, y_val_proba)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)

        # Log trial info
        mlflow.log_params(params)
        mlflow.log_metrics({
            "val_auc": val_auc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
        })

        # Optimize recall (important for fraud detection)
        return val_recall


if __name__ == "__main__":
    with mlflow.start_run(run_name="optuna_study"):
        mlflow.set_tag("task", "fraud_detection")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("optimizer", "optuna")

        # Run Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)

        # Best trial
        best_params = study.best_trial.params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_recall", study.best_value)

        # Retrain best model on full dataset
        X, y = load_data()
        best_model = RandomForestClassifier(
            **best_params, random_state=RANDOM_STATE, n_jobs=-1
        )
        best_model.fit(X, y)

        # Log model artifact
        mlflow.sklearn.log_model(best_model, name="best_model")
        print("Best trial params:", best_params)
        print("Best recall:", study.best_value)