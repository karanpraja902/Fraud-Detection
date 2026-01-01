from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("data/processed/train.csv")
EXPERIMENT_NAME = "Fraud Detection"
RANDOM_STATE = 42
N_TRIALS = 20


def load_data():
    """Load processed training data."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def train_model(train_path: str = str(DATA_PATH)):
    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    def objective(trial):
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.2,
            stratify=y_trainval,
            random_state=RANDOM_STATE,
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
            threshold = trial.suggest_float("threshold", 0.1, 0.5)
            y_val_pred = (y_val_proba >= threshold).astype(int)

            # Metrics
            val_auc = roc_auc_score(y_val, y_val_proba)
            val_precision = precision_score(y_val, y_val_pred)
            val_recall = recall_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)

            # Log trial info
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "val_auc": val_auc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                }
            )

            # Optimize recall (important for fraud detection)
            return val_recall

    with mlflow.start_run(run_name="live-demo"):
        mlflow.set_tag("task", "fraud_detection")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("optimizer", "optuna")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)

        best_params = study.best_trial.params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_recall", study.best_value)

        # Separate model parameters from threshold (threshold is for prediction, not model init)
        model_params = {k: v for k, v in best_params.items() if k != "threshold"}
        optimal_threshold = best_params["threshold"]

        best_model = RandomForestClassifier(
            **model_params, random_state=RANDOM_STATE, n_jobs=-1
        )
        best_model.fit(X_trainval, y_trainval)

        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        y_test_pred = best_model.predict(X_test)

        test_auc = roc_auc_score(y_test, y_test_proba)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        mlflow.log_metrics(
            {
                "test_auc": test_auc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
            }
        )

        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {test_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        precisions, recalls, _ = precision_recall_curve(y_test, y_test_proba)
        plt.figure()
        plt.plot(recalls, precisions, label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig("pr_curve.png")
        mlflow.log_artifact("pr_curve.png")
        plt.close()

        input_example = X_trainval.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            registered_model_name="fraud-detector",
            input_example=input_example,
        )
        print("Best trial params:", best_params)
        print("Best recall:", study.best_value)

        return best_model


if __name__ == "__main__":
    train_model()
