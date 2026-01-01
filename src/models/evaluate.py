"""
Model evaluation module for fraud detection system.
Implements comprehensive model evaluation following Google MLOps whitepaper guidelines.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation class following MLOps best practices.
    Implements continuous evaluation and monitoring capabilities.
    """

    def __init__(
        self,
        model_path: str = "models/latest_model.pkl",
        experiment_name: str = "fraud_detection_eval",
    ):
        """
        Initialize evaluator with model and experiment settings.

        Args:
            model_path: Path to trained model
            experiment_name: MLflow experiment name
        """
        self.model_path = Path(model_path)
        self.experiment_name = experiment_name
        self.model: Optional[BaseEstimator] = None

    def load_model(self):
        """Load the trained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    @property
    def _safe_model(self) -> BaseEstimator:
        """Get model with runtime safety and type narrowing"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model could not be loaded")
        # Type narrowing for mypy + runtime safety
        model = self.model
        if model is None:
            raise ValueError("Model is unexpectedly None after loading")
        return model

    def evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test",
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple metrics and cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            dataset_name: Name of dataset being evaluated
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing all evaluation results and artifacts
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Could not load model")

        # Set MLflow experiment
        mlflow.set_experiment(self.experiment_name)

        results = {
            "dataset": dataset_name,
            "metrics": {},
            "plots": {},
            "cv_results": {},
            "feature_importance": {},
            "bias_analysis": {},
        }

        with mlflow.start_run(run_name=f"evaluation_{dataset_name}"):

            # Basic predictions - use safe model property
            model = self._safe_model
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = model.predict(X)

            # Classification metrics
            results["metrics"] = self._compute_classification_metrics(
                y, y_pred, y_pred_proba
            )

            # Cross-validation for robustness
            results["cv_results"] = self._cross_validation_eval(X, y, cv_folds)

            # Generate evaluation plots
            results["plots"] = self._generate_evaluation_plots(
                X, y, y_pred, y_pred_proba, dataset_name
            )

            # Feature importance analysis
            results["feature_importance"] = self._analyze_feature_importance(X)

            # Bias and fairness analysis
            results["bias_analysis"] = self._bias_and_fairness_analysis(X, y, y_pred)

            # Log all results to MLflow
            self._log_to_mlflow(results, X, y)

            print(f"Evaluation completed for {dataset_name} dataset")
            print(".3f")
            print(".3f")
            print(".3f")

        return results

    def _compute_classification_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = np.mean(y_true == y_pred)
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["f1_score"] = f1_score(y_true, y_pred)

        # Probability-based metrics
        metrics["auc_roc"] = auc(*roc_curve(y_true, y_pred_proba)[:2])
        metrics["auc_pr"] = average_precision_score(y_true, y_pred_proba)

        # Class-specific metrics
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        metrics["precision_fraud"] = report["1"]["precision"]
        metrics["recall_fraud"] = report["1"]["recall"]
        metrics["f1_fraud"] = report["1"]["f1-score"]

        # Additional fraud detection metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        return metrics

    def _cross_validation_eval(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int
    ) -> Dict[str, Any]:
        """Perform cross-validation evaluation for model robustness."""
        cv_results = {}

        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_predictions = cross_val_predict(
                self._safe_model, X, y, cv=cv, method="predict_proba"
            )

            # CV metrics
            cv_auc = []
            cv_precision = []
            cv_recall = []
            cv_f1 = []

            for train_idx, test_idx in cv.split(X, y):
                y_test_cv = y.iloc[test_idx]
                y_pred_cv = cv_predictions[test_idx][:, 1]

                cv_auc.append(auc(*roc_curve(y_test_cv, y_pred_cv)[:2]))
                y_pred_class = (y_pred_cv >= 0.5).astype(int)
                cv_precision.append(precision_score(y_test_cv, y_pred_class))
                cv_recall.append(recall_score(y_test_cv, y_pred_class))
                cv_f1.append(f1_score(y_test_cv, y_pred_class))

            cv_results = {
                "mean_auc": np.mean(cv_auc),
                "std_auc": np.std(cv_auc),
                "mean_precision": np.mean(cv_precision),
                "std_precision": np.std(cv_precision),
                "mean_recall": np.mean(cv_recall),
                "std_recall": np.std(cv_recall),
                "mean_f1": np.mean(cv_f1),
                "std_f1": np.std(cv_f1),
            }

        except Exception as e:
            print(f"Cross-validation failed: {e}")
            cv_results = {"error": str(e)}

        return cv_results

    def _generate_evaluation_plots(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        dataset_name: str,
    ) -> Dict[str, str]:
        """Generate evaluation plots and save them."""
        plots = {}
        output_dir = Path("evaluation_plots")
        output_dir.mkdir(exist_ok=True)

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
        )
        plt.title(f"Confusion Matrix - {dataset_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        cm_path = output_dir / f"confusion_matrix_{dataset_name}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["confusion_matrix"] = str(cm_path)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=".2f")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {dataset_name}")
        plt.legend(loc="lower right")
        roc_path = output_dir / f"roc_curve_{dataset_name}.png"
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["roc_curve"] = str(roc_path)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = average_precision_score(y, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="blue", lw=2, label=".2f")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {dataset_name}")
        plt.legend(loc="lower left")
        pr_path = output_dir / f"precision_recall_{dataset_name}.png"
        plt.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["precision_recall"] = str(pr_path)

        # Prediction Probability Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(
            y_pred_proba[y == 0], alpha=0.7, label="Legitimate", bins=50, density=True
        )
        plt.hist(y_pred_proba[y == 1], alpha=0.7, label="Fraud", bins=50, density=True)
        plt.xlabel("Prediction Probability")
        plt.ylabel("Density")
        plt.title(f"Prediction Probability Distribution - {dataset_name}")
        plt.legend()
        plt.axvline(
            x=0.5, color="red", linestyle="--", alpha=0.7, label="Decision Threshold"
        )
        prob_path = output_dir / f"probability_distribution_{dataset_name}.png"
        plt.savefig(prob_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots["probability_distribution"] = str(prob_path)

        return plots

    def _analyze_feature_importance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance if available."""
        importance_data = {}

        model = self._safe_model
        if hasattr(model, "feature_importances_"):
            # For tree-based models
            importance = model.feature_importances_
            feature_names = X.columns

            # Sort by importance
            indices = np.argsort(importance)[::-1]
            importance_data = {
                "features": feature_names[indices[:10]].tolist(),
                "importance_scores": importance[indices[:10]].tolist(),
            }

            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            plt.barh(range(10), importance[indices[:10]])
            plt.yticks(range(10), feature_names[indices[:10]])
            plt.xlabel("Feature Importance")
            plt.title("Top 10 Feature Importance")
            plt.tight_layout()
            feature_plot_path = Path("evaluation_plots") / "feature_importance.png"
            plt.savefig(feature_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            importance_data["plot_path"] = str(feature_plot_path)

        elif hasattr(model, "coef_"):
            # For linear models
            coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            feature_names = X.columns

            indices = np.argsort(np.abs(coefficients))[::-1]
            importance_data = {
                "features": feature_names[indices[:10]].tolist(),
                "coefficients": coefficients[indices[:10]].tolist(),
            }

        return importance_data

    def _bias_and_fairness_analysis(
        self, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Perform basic bias and fairness analysis."""
        bias_analysis = {}

        # Check for class imbalance in predictions
        pred_fraud_rate = np.mean(y_pred)
        actual_fraud_rate = np.mean(y)

        bias_analysis["prediction_fraud_rate"] = pred_fraud_rate
        bias_analysis["actual_fraud_rate"] = actual_fraud_rate
        bias_analysis["prediction_bias"] = pred_fraud_rate - actual_fraud_rate

        # Calculate disparate impact for basic features
        # This is a simplified analysis - production systems need more comprehensive fairness metrics
        if "Amount" in X.columns:
            # Check if high-value transactions are treated differently
            high_value_threshold = X["Amount"].quantile(0.9)
            high_value_mask = X["Amount"] > high_value_threshold

            if high_value_mask.sum() > 0:
                high_value_fraud_rate = np.mean(y[high_value_mask])
                low_value_fraud_rate = np.mean(y[~high_value_mask])

                bias_analysis["high_value_fraud_rate"] = high_value_fraud_rate
                bias_analysis["low_value_fraud_rate"] = low_value_fraud_rate

        return bias_analysis

    def _log_to_mlflow(self, results: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """Log evaluation results to MLflow."""
        # Log metrics
        for metric_name, metric_value in results["metrics"].items():
            mlflow.log_metric(metric_name, metric_value)

        # Log CV results
        for cv_metric, cv_value in results["cv_results"].items():
            if isinstance(cv_value, (int, float)):
                mlflow.log_metric(f"cv_{cv_metric}", cv_value)

        # Log plots
        for plot_name, plot_path in results["plots"].items():
            mlflow.log_artifact(plot_path, "evaluation_plots")

        # Log feature importance data
        if results["feature_importance"]:
            with open("feature_importance.json", "w") as f:
                json.dump(results["feature_importance"], f, indent=2)
            mlflow.log_artifact("feature_importance.json", "feature_analysis")

        # Log bias analysis
        with open("bias_analysis.json", "w") as f:
            json.dump(results["bias_analysis"], f, indent=2)
        mlflow.log_artifact("bias_analysis.json", "fairness_analysis")

        # Log dataset information
        mlflow.log_param("dataset_shape", f"{X.shape[0]}x{X.shape[1]}")
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_param("target_distribution", dict(y.value_counts()))

        # Log model info
        if hasattr(self.model, "__class__"):
            mlflow.log_param("model_type", self.model.__class__.__name__)


def evaluate_and_monitor(
    model_path: str,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "evaluation",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Convenience function to run full evaluation pipeline.
    Follows the continuous monitoring pattern from Google MLOps whitepaper.
    """
    evaluator = ModelEvaluator(model_path)
    results = evaluator.evaluate_model(X, y, dataset_name)

    # Apply threshold for final predictions
    results["threshold_used"] = threshold
    optimal_threshold = find_optimal_threshold(X, y, threshold)
    results["optimal_threshold"] = optimal_threshold

    return results


def find_optimal_threshold(
    X: pd.DataFrame, y: pd.Series, current_threshold: float = 0.5
) -> float:
    """
    Find optimal threshold based on F1 score maximization.
    This could be enhanced with cost-sensitive approaches.
    """
    from sklearn.model_selection import train_test_split

    # Quick validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Load model and get probabilities
    evaluator = ModelEvaluator()
    evaluator.load_model()
    model = evaluator._safe_model
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = current_threshold

    for thresh in thresholds:
        y_pred = (y_val_proba >= thresh).astype(int)
        current_f1 = f1_score(y_val, y_pred)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(thresh)

    return float(best_threshold)


if __name__ == "__main__":
    # Example usage for standalone evaluation
    import sys

    sys.path.append("src")

    from models.train import load_data

    # Load test data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Run evaluation
    results = evaluate_and_monitor(
        "models/latest_model.pkl", X_test, y_test, "production_test"
    )

    print("Evaluation completed. Check MLflow UI for detailed results.")
    print(".4f")
