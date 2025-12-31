"""
Continuous Training Pipeline for Fraud Detection System.
Implements the training operationalization and continuous training processes
from Google MLOps whitepaper with data validation, training, evaluation, and registration.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Add src to path to import our modules
sys.path.append("src")
from data.preprocess import (
    handle_imbalance,
    load_data,
    normalize_features,
    run_preprocessing,
)
from models.evaluate import ModelEvaluator, evaluate_and_monitor
from models.train import load_data as load_processed_data
from models.train import train_model

# Configuration
DATA_CONFIG = {
    "raw_data_path": "data/raw/creditcard.csv",
    "processed_data_path": "data/processed",
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
}

VALIDATION_CONFIG = {
    "min_samples": 1000,
    "max_class_imbalance_ratio": 0.1,  # Alert if fraud rate changes by more than 10%
    "schema_drift_threshold": 0.05,  # Alert if feature distributions change significantly
    "data_drift_threshold": 0.1,
}

TRAINING_CONFIG = {
    "experiment_name": "fraud_detection_continuous_training",
    "model_name": "fraud_detector",
    "min_auc_score": 0.85,
    "min_recall_fraud": 0.75,
    "registry_model_name": "fraud-detector",
}


@task(name="data_ingestion")
def data_ingestion_task(raw_data_path: str):
    """
    Ingest and validate raw data source.
    Checks for file existence, basic structure, and data quality.
    """
    print("🔄 Ingesting data...")

    if not Path(raw_data_path).exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    # Load basic data info without processing
    df = pd.read_csv(raw_data_path)

    # Basic validation
    if df.empty:
        raise ValueError("Dataset is empty")

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found")

    required_cols = ["Time", "Amount"]  # Critical columns for fraud detection
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ Data ingested: {len(df)} records, {len(df.columns)} columns")

    return {
        "raw_data_path": raw_data_path,
        "num_samples": len(df),
        "num_features": len(df.columns),
        "fraud_rate": df["Class"].mean(),
        "ingestion_timestamp": datetime.now().isoformat(),
    }


@task(name="data_validation")
def data_validation_task(
    data_info: dict, previous_training_stats_path: str = "training_stats.json"
):
    """
    Validate data quality and check for anomalies/drifts.
    Compares current data against baseline statistics from previous training.
    """
    print("🔍 Validating data quality and checking for drifts...")

    raw_data_path = data_info["raw_data_path"]
    df = pd.read_csv(raw_data_path)

    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "drift_detected": False,
        "quality_metrics": {},
    }

    # Basic quality checks
    validation_results["quality_metrics"].update(
        {
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "negative_amounts": (df["Amount"] < 0).sum(),
            "invalid_time_values": (df["Time"] < 0).sum(),
        }
    )

    # Alert on critical issues
    if validation_results["quality_metrics"]["missing_values"] > 0:
        validation_results["errors"].append("Missing values detected in dataset")
        validation_results["is_valid"] = False

    # Check for class imbalance issues
    fraud_rate = df["Class"].mean()
    if fraud_rate < 0.001 or fraud_rate > 0.1:
        validation_results["warnings"].append(f"Unusual fraud rate: {fraud_rate:.4f}")

    # Check distribution drift against previous training data
    if Path(previous_training_stats_path).exists():
        try:
            with open(previous_training_stats_path, "r") as f:
                previous_stats = json.load(f)

            baseline_fraud_rate = previous_stats.get(
                "fraud_rate", data_info["fraud_rate"]
            )

            # Calculate drift
            fraud_rate_drift = (
                abs(fraud_rate - baseline_fraud_rate) / baseline_fraud_rate
            )
            if fraud_rate_drift > VALIDATION_CONFIG["max_class_imbalance_ratio"]:
                validation_results["drift_detected"] = True
                validation_results["warnings"].append(".2%")
                validation_results["quality_metrics"][
                    "fraud_rate_drift"
                ] = fraud_rate_drift

        except Exception as e:
            print(f"Warning: Could not load previous training stats: {e}")

    # Schema validation
    expected_features = len(df.columns) - 1  # Excluding target
    previous_stats = None
    if Path(previous_training_stats_path).exists():
        try:
            with open(previous_training_stats_path, "r") as f:
                previous_stats = json.load(f)
        except Exception:
            pass

    if previous_stats:
        baseline_features = previous_stats.get("num_features", expected_features)
        if expected_features != baseline_features:
            validation_results["drift_detected"] = True
            validation_results["errors"].append("Number of features changed")
            validation_results["is_valid"] = False

    print(
        f"✅ Data validation completed. Valid: {validation_results['is_valid']}, Warnings: {len(validation_results['warnings'])}, Errors: {len(validation_results['errors'])}"
    )

    return validation_results


@task(name="data_preprocessing")
def data_preprocessing_task(validation_results: dict, data_info: dict):
    """
    Preprocess data for training: normalization, feature engineering, imbalance handling.
    Only runs if data validation passed.
    """
    print("🔧 Preprocessing data...")

    if not validation_results["is_valid"]:
        raise ValueError("Data validation failed - cannot proceed with preprocessing")

    raw_data_path = data_info["raw_data_path"]
    processed_path = DATA_CONFIG["processed_data_path"]

    # Run preprocessing
    run_preprocessing(raw_data_path, processed_path)

    # Load processed data and compute additional stats
    X_train = pd.read_csv(f"{processed_path}/train.csv")
    X_test = pd.read_csv(f"{processed_path}/test.csv")

    preprocessing_stats = {
        "processed_train_samples": len(X_train),
        "processed_test_samples": len(X_test),
        "final_features": list(X_train.columns),
        "feature_stats": {
            "mean_transaction": X_train["Amount"].mean(),
            "fraud_transaction_mean": X_train.groupby("Class")["Amount"]
            .mean()
            .to_dict(),
        },
    }

    print(f"✅ Data preprocessing completed: {len(X_train)} training samples")

    return {
        "train_data_path": f"{processed_path}/train.csv",
        "test_data_path": f"{processed_path}/test.csv",
        "preprocessing_stats": preprocessing_stats,
    }


@task(name="model_training")
def model_training_task(preprocessing_results: dict):
    """
    Train model with hyperparameter optimization and evaluation.
    Includes Optuna-based hyperparameter tuning and comprehensive evaluation.
    """
    print("🚀 Training model...")

    train_data_path = preprocessing_results["train_data_path"]

    # Train model (this internally handles MLflow tracking)
    trained_model = train_model(train_data_path)

    # Load test data for initial evaluation
    test_data_path = preprocessing_results["test_data_path"]
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    # Perform comprehensive evaluation
    evaluator = ModelEvaluator(
        "models/latest_model.pkl", "fraud_detection_continuous_training"
    )
    eval_results = evaluator.evaluate_model(X_test, y_test, "continuous_training_test")

    training_results = {
        "model_path": "models/latest_model.pkl",
        "evaluation_results": eval_results,
        "training_timestamp": datetime.now().isoformat(),
        "performance_metrics": eval_results["metrics"],
    }

    # Check if model meets production criteria
    auc_score = eval_results["metrics"].get("auc_roc", 0)
    recall_fraud = eval_results["metrics"].get("recall_fraud", 0)

    meets_criteria = (
        auc_score >= TRAINING_CONFIG["min_auc_score"]
        and recall_fraud >= TRAINING_CONFIG["min_recall_fraud"]
    )

    training_results["meets_production_criteria"] = meets_criteria

    print(".3f")
    print(".3f")
    print(f"Meets production criteria: {meets_criteria}")

    return training_results


@task(name="model_validation")
def model_validation_task(training_results: dict):
    """
    Additional model validation against business requirements and historical performance.
    Includes threshold optimization and production-readiness checks.
    """
    print("✅ Validating model performance...")

    evaluation_results = training_results["evaluation_results"]

    validation_results = {
        "model_ready_for_production": training_results["meets_production_criteria"],
        "validation_checks": {},
        "recommendations": [],
    }

    # Business logic validation
    metrics = evaluation_results["metrics"]

    # Fraud detection specific validations
    if metrics.get("recall_fraud", 0) < 0.8:  # Must catch at least 80% of fraud
        validation_results["validation_checks"]["fraud_recall"] = "FAIL"
        validation_results["recommendations"].append(
            "Improve fraud recall - consider different model or resampling"
        )
    else:
        validation_results["validation_checks"]["fraud_recall"] = "PASS"

    # Check for overfitting using CV metrics
    cv_metrics = evaluation_results.get("cv_results", {})
    test_auc = metrics.get("auc_roc", 0)
    cv_mean_auc = cv_metrics.get("mean_auc", test_auc)

    overfitting_ratio = abs(test_auc - cv_mean_auc) / cv_mean_auc
    if overfitting_ratio > 0.1:  # More than 10% difference
        validation_results["validation_checks"]["overfitting"] = "WARNING"
        validation_results["recommendations"].append(".2%")
    else:
        validation_results["validation_checks"]["overfitting"] = "PASS"

    # Threshold analysis
    optimal_threshold = find_optimal_threshold_for_business(
        training_results["evaluation_results"]
    )
    validation_results["optimal_threshold"] = optimal_threshold

    # Final decision - can override automated criteria based on business rules
    validation_results["model_ready_for_production"] = (
        validation_results["model_ready_for_production"]
        and validation_results["validation_checks"].get("fraud_recall") == "PASS"
    )

    print(
        f"Model ready for production: {validation_results['model_ready_for_production']}"
    )
    if validation_results["recommendations"]:
        print("Recommendations:", validation_results["recommendations"])

    return validation_results


@task(name="model_registration")
def model_registration_task(training_results: dict, validation_results: dict):
    """
    Register model in MLflow Model Registry if it passes validation.
    Includes model metadata and governance information.
    """
    print("📝 Registering model...")

    if not validation_results["model_ready_for_production"]:
        print("⚠️ Model does not meet production criteria - skipping registration")
        return {
            "registered": False,
            "reason": "Model validation failed",
            "registry_info": None,
        }

    # Load model and register
    model = joblib.load(training_results["model_path"])

    # Create new version in registry
    evaluation_results = training_results["evaluation_results"]
    metrics = evaluation_results["metrics"]

    with mlflow.start_run():
        # Log model with signature
        import pandas as pd
        from mlflow.models import infer_signature

        # Get sample for signature
        sample_df = pd.read_csv(f"{DATA_CONFIG['processed_data_path']}/test.csv")
        sample_input = sample_df.drop("Class", axis=1).head(5)
        sample_output = model.predict(sample_input)

        signature = infer_signature(sample_input, sample_output)

        # Register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=TRAINING_CONFIG["registry_model_name"],
            signature=signature,
            input_example=sample_input.head(1),
        )

        # Add model metadata
        mlflow.set_tags(
            {
                "model_type": "fraud_detection",
                "training_type": "continuous_training",
                "data_source": "credit_card_transactions",
                "algorithm": model.__class__.__name__,
                "stage": "Staging",  # Can be promoted to Production via governance
            }
        )

        mlflow.log_params(
            {
                "training_timestamp": training_results["training_timestamp"],
                "optimal_threshold": validation_results["optimal_threshold"],
                "validation_status": "PASSED",
                "meets_production_criteria": validation_results[
                    "model_ready_for_production"
                ],
            }
        )

        # Log performance metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"performance_{metric_name}", metric_value)

    registration_info = {
        "model_name": TRAINING_CONFIG["registry_model_name"],
        "stage": "Staging",
        "version": "latest",  # Would get actual version from registry
        "registration_timestamp": datetime.now().isoformat(),
        "performance_snapshot": metrics,
    }

    print(f"✅ Model registered in MLflow: {TRAINING_CONFIG['registry_model_name']}")

    return {
        "registered": True,
        "reason": "Model validation passed",
        "registry_info": registration_info,
    }


@task(name="update_training_stats")
def update_training_stats_task(data_info: dict, preprocessing_results: dict):
    """
    Update training statistics for future drift detection.
    Saves baseline statistics for comparison in future runs.
    """
    print("💾 Updating baseline training statistics...")

    stats_path = "training_stats.json"

    # Combine information from current run
    current_stats = {
        **data_info,
        **preprocessing_results["preprocessing_stats"],
        "last_training_run": datetime.now().isoformat(),
        "mlflow_experiment": TRAINING_CONFIG["experiment_name"],
    }

    # Load existing stats if available
    if Path(stats_path).exists():
        try:
            with open(stats_path, "r") as f:
                existing_stats = json.load(f)
            # Merge with history (keep last few runs)
            existing_stats["training_history"] = existing_stats.get(
                "training_history", []
            )
            existing_stats["training_history"].append(
                {
                    "run_date": existing_stats.get("last_training_run"),
                    "fraud_rate": existing_stats.get("fraud_rate"),
                    "num_samples": existing_stats.get("num_samples"),
                }
            )
            # Keep only last 5 runs
            existing_stats["training_history"] = existing_stats["training_history"][-5:]
            current_stats["training_history"] = existing_stats["training_history"]
        except Exception as e:
            print(f"Warning: Could not load existing stats: {e}")

    # Save updated stats
    with open(stats_path, "w") as f:
        json.dump(current_stats, f, indent=2, default=str)

    print(f"✅ Training statistics saved to {stats_path}")

    return current_stats


# Main flow following Google MLOps whitepaper pattern
@flow(
    name="Continuous Training Pipeline",
    description="Automated ML training pipeline with validation and registration",
)
def continuous_training_pipeline(
    raw_data_path: str = DATA_CONFIG["raw_data_path"], trigger_type: str = "manual"
):
    """
    End-to-end continuous training pipeline implementing:
    1. Data ingestion and validation
    2. Data preprocessing
    3. Model training with evaluation
    4. Model validation against business requirements
    5. Model registration if production-ready
    6. Baseline statistics update

    Args:
        raw_data_path: Path to raw data file
        trigger_type: Type of trigger (manual, scheduled, event-driven)
    """
    print("🚀 Starting Continuous Training Pipeline...")
    print(f"Trigger type: {trigger_type}")
    print(f"Data source: {raw_data_path}")

    # Phase 1: Data ingestion and validation
    data_info = data_ingestion_task(raw_data_path)
    validation_results = data_validation_task(data_info)
    preprocessing_results = data_preprocessing_task(validation_results, data_info)

    # Phase 2: Model training and validation
    training_results = model_training_task(preprocessing_results)
    validation_results = model_validation_task(training_results)

    # Phase 3: Model registration and artifacts
    registration_results = model_registration_task(training_results, validation_results)
    training_stats = update_training_stats_task(data_info, preprocessing_results)

    # Summary
    pipeline_results = {
        "pipeline_completion": datetime.now().isoformat(),
        "trigger_type": trigger_type,
        "steps_completed": [
            "data_ingestion",
            "data_validation",
            "data_preprocessing",
            "model_training",
            "model_validation",
            "model_registration",
            "stats_update",
        ],
        "model_registered": registration_results["registered"],
        "data_drift_detected": validation_results.get("drift_detected", False),
        "recommendations": validation_results.get("recommendations", []),
    }

    print("🎉 Continuous Training Pipeline completed!")
    print(f"Model registered: {registration_results['registered']}")

    return pipeline_results


def find_optimal_threshold_for_business(evaluation_results: dict) -> float:
    """
    Find optimal threshold based on business requirements (prioritize fraud recall).
    This is a simplified version - production systems would have cost-benefit analysis.
    """
    # For fraud detection, we often want to maximize recall while maintaining reasonable precision
    # Use F1 score for balanced optimization, but could be customized based on business needs
    cv_results = evaluation_results.get("cv_results", {})

    # Default to 0.5 if no CV results
    if not cv_results or "mean_f1" not in cv_results:
        return 0.5

    # Could implement more sophisticated threshold optimization here
    # For now, return default as CV results don't give us prediction thresholds
    return 0.5


# Schedule and trigger configurations
if __name__ == "__main__":
    import sys

    trigger_type = sys.argv[1] if len(sys.argv) > 1 else "manual"

    print("🔄 Running Continuous Training Pipeline...")
    print(f"Configuration: {DATA_CONFIG}")

    results = continuous_training_pipeline(trigger_type=trigger_type)

    print("Pipeline completed successfully!")
    if results["recommendations"]:
        print("\n📋 Recommendations:")
        for rec in results["recommendations"]:
            print(f"  - {rec}")
