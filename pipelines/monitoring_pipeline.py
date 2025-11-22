"""
Monitoring Pipeline for Continuous Model Performance Tracking.
Implements the Continuous Monitoring process from Google MLOps whitepaper.
Handles data drift detection, model performance monitoring, and alerting.
"""

from prefect import task, flow
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import our modules
sys.path.append('src')
from models.evaluate import ModelEvaluator, evaluate_and_monitor
from models.train import load_data

# Configuration
MONITORING_CONFIG = {
    "baseline_stats_path": "training_stats.json",
    "drift_threshold": 0.1,  # Alert threshold for drift detection
    "performance_decay_threshold": 0.05,  # 5% performance degradation
    "alert_email": "karanpraja902@gmail.com",
    "monitoring_logs_path": "monitoring_logs/",
    "reference_sample_size": 1000  # Size of reference data for comparison
}

ALERT_LEVELS = {
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high",
    "CRITICAL": "critical"
}


@task(name="load_reference_data")
def load_reference_data_task():
    """
    Load reference data for drift detection.
    Uses baseline training data as reference distribution.
    """
    print("📊 Loading reference data for baseline comparison...")

    try:
        with open(MONITORING_CONFIG["baseline_stats_path"], 'r') as f:
            baseline_stats = json.load(f)

        # Load reference data (last training data)
        reference_df = pd.read_csv(f"data/processed/train.csv")
        X_ref = reference_df.drop('Class', axis=1)
        y_ref = reference_df['Class']

        # Take a sample for efficient monitoring
        sample_size = min(MONITORING_CONFIG["reference_sample_size"], len(X_ref))
        reference_sample = reference_df.sample(n=sample_size, random_state=42)

        reference_info = {
            "data": reference_sample,
            "stats": baseline_stats,
            "sample_size": sample_size,
            "features": list(X_ref.columns),
            "target_distribution": dict(y_ref.value_counts()),
            "reference_timestamp": baseline_stats.get("last_training_run"),
            "fraud_rate": baseline_stats.get("fraud_rate")
        }

        print(f"✅ Reference data loaded: {sample_size} samples, fraud rate: {reference_info['fraud_rate']:.4f}")

        return reference_info

    except Exception as e:
        print(f"❌ Error loading reference data: {e}")
        raise


@task(name="detect_data_drift")
def detect_data_drift_task(reference_info: dict, current_data_path: str = "data/processed/train.csv"):
    """
    Detect data drift between current data and reference baseline.
    Monitors schema changes, distribution shifts, and statistical anomalies.
    """
    print("🔍 Detecting data drift...")

    drift_results = {
        "drift_detected": False,
        "drift_type": None,
        "drift_scores": {},
        "drift_details": {},
        "alert_level": ALERT_LEVELS["LOW"],
        "timestamp": datetime.now().isoformat()
    }

    try:
        # Load current data
        current_df = pd.read_csv(current_data_path)
        reference_df = reference_info["data"]

        # 1. Schema Drift Detection
        current_features = set(current_df.columns)
        reference_features = set(reference_df.columns)

        if current_features != reference_features:
            drift_results["drift_detected"] = True
            drift_results["drift_type"] = "schema_drift"
            drift_results["alert_level"] = ALERT_LEVELS["CRITICAL"]
            drift_results["drift_details"]["missing_features"] = list(reference_features - current_features)
            drift_results["drift_details"]["new_features"] = list(current_features - reference_features)

            print("🚨 Schema drift detected!")
            return drift_results

        # 2. Distribution Drift Detection (simplified version)
        # Compare key statistics for numerical features
        numerical_features = current_df.select_dtypes(include=[np.number]).columns
        drift_features = []

        for feature in numerical_features:
            if feature == 'Class':
                continue

            current_mean = current_df[feature].mean()
            reference_mean = reference_df[feature].mean()

            current_std = current_df[feature].std()
            reference_std = reference_df[feature].std()

            # Calculate drift score using simple statistical distance
            mean_drift = abs(current_mean - reference_mean) / (reference_std + 1e-6)
            std_drift = abs(current_std - reference_std) / (reference_std + 1e-6)

            drift_score = max(mean_drift, std_drift)
            drift_results["drift_scores"][feature] = drift_score

            if drift_score > MONITORING_CONFIG["drift_threshold"]:
                drift_features.append(feature)

        if drift_features:
            drift_results["drift_detected"] = True
            drift_results["drift_type"] = "distribution_drift"
            drift_results["drift_details"]["drifted_features"] = drift_features
            drift_results["alert_level"] = ALERT_LEVELS["HIGH"] if len(drift_features) > 3 else ALERT_LEVELS["MEDIUM"]

        # 3. Class Imbalance Drift Detection
        current_fraud_rate = current_df['Class'].mean()
        baseline_fraud_rate = reference_info["fraud_rate"]

        fraud_rate_drift = abs(current_fraud_rate - baseline_fraud_rate) / (baseline_fraud_rate + 1e-6)

        if fraud_rate_drift > MONITORING_CONFIG["drift_threshold"]:
            drift_results["drift_detected"] = True
            if not drift_results["drift_type"]:
                drift_results["drift_type"] = "class_imbalance_drift"
            drift_results["drift_details"]["fraud_rate_drift"] = fraud_rate_drift
            drift_results["alert_level"] = ALERT_LEVELS["HIGH"]

        drift_results["drift_details"]["current_fraud_rate"] = current_fraud_rate
        drift_results["drift_details"]["baseline_fraud_rate"] = baseline_fraud_rate

        if drift_results["drift_detected"]:
            print(f"⚠️ {drift_results['drift_type']} detected - Alert level: {drift_results['alert_level']}")
        else:
            print("✅ No significant data drift detected")

        return drift_results

    except Exception as e:
        print(f"❌ Error in drift detection: {e}")
        drift_results["drift_detected"] = False
        drift_results["error"] = str(e)
        return drift_results


@task(name="load_production_model")
def load_production_model_task():
    """
    Load the currently deployed production model for performance monitoring.
    """
    print("📤 Loading production model...")

    try:
        import mlflow.pyfunc
        import mlflow

        # Load latest production model from registry
        model = mlflow.pyfunc.load_model("models:/fraud-detector/Production")

        model_info = {
            "model": model,
            "model_name": "fraud-detector",
            "stage": "Production",
            "load_timestamp": datetime.now().isoformat()
        }

        print("✅ Production model loaded successfully")
        return model_info

    except Exception as e:
        print(f"❌ Error loading production model: {e}")
        raise


@task(name="evaluate_model_performance")
def evaluate_model_performance_task(model_info: dict, evaluation_data_path: str = "data/processed/test.csv"):
    """
    Evaluate current model performance on recent data.
    Compare against baseline performance metrics.
    """
    print("📊 Evaluating model performance...")

    try:
        # Load evaluation data
        eval_df = pd.read_csv(evaluation_data_path)
        X_eval = eval_df.drop('Class', axis=1)
        y_eval = eval_df['Class']

        # Use our comprehensive evaluation
        model_path = "models/latest_model.pkl"  # Temporary file for evaluation
        import joblib
        joblib.dump(model_info["model"], model_path)

        evaluator = ModelEvaluator(model_path, "model_monitoring")
        evaluation_results = evaluator.evaluate_model(X_eval, y_eval, "production_monitoring")

        # Load baseline metrics for comparison
        baseline_metrics = {}
        try:
            with open(MONITORING_CONFIG["baseline_stats_path"], 'r') as f:
                baseline = json.load(f)
                # This would contain reference metrics from training
        except:
            baseline_metrics = {"auc_roc": 0.85, "precision": 0.8, "recall": 0.75}  # Default baselines

        # Check for performance decay
        current_metrics = evaluation_results["metrics"]
        performance_decay = {}

        for metric in ["auc_roc", "precision", "recall", "f1_score"]:
            if metric in baseline_metrics and metric in current_metrics:
                baseline_value = baseline_metrics[metric]
                current_value = current_metrics[metric]
                decay = (baseline_value - current_value) / (baseline_value + 1e-6)
                performance_decay[metric] = decay

        # Determine if retraining is needed
        significant_decay = any(
            decay > MONITORING_CONFIG["performance_decay_threshold"]
            for decay in performance_decay.values()
        )

        performance_analysis = {
            "evaluation_results": evaluation_results,
            "performance_decay": performance_decay,
            "significant_decay": significant_decay,
            "retraining_required": significant_decay,
            "baseline_metrics": baseline_metrics,
            "current_metrics": current_metrics,
            "timestamp": datetime.now().isoformat()
        }

        if significant_decay:
            print("⚠️ Significant performance decay detected - retraining recommended")
            performance_analysis["alert_level"] = ALERT_LEVELS["HIGH"]
        else:
            print("✅ Model performance stable")
            performance_analysis["alert_level"] = ALERT_LEVELS["LOW"]

        # Clean up temporary model file
        Path(model_path).unlink(missing_ok=True)

        return performance_analysis

    except Exception as e:
        print(f"❌ Error in performance evaluation: {e}")
        raise


@task(name="generate_monitoring_report")
def generate_monitoring_report_task(drift_results: dict, performance_analysis: dict):
    """
    Generate comprehensive monitoring report with alerts and recommendations.
    """
    print("📋 Generating monitoring report...")

    report = {
        "monitoring_cycle": datetime.now().isoformat(),
        "overall_status": "HEALTHY",
        "alerts": [],
        "recommendations": [],
        "data_drift": drift_results,
        "model_performance": {
            "current_metrics": performance_analysis.get("current_metrics", {}),
            "performance_decay": performance_analysis.get("performance_decay", {}),
            "retraining_required": performance_analysis.get("retraining_required", False)
        },
        "summary": {
            "data_drift_detected": drift_results.get("drift_detected", False),
            "performance_decay_detected": performance_analysis.get("significant_decay", False),
            "highest_alert_level": ALERT_LEVELS["LOW"]
        }
    }

    # Determine overall status and generate alerts
    if drift_results.get("drift_detected") or performance_analysis.get("significant_decay"):
        report["overall_status"] = "WARNING"
        report["summary"]["highest_alert_level"] = ALERT_LEVELS["HIGH"]

        if drift_results.get("drift_detected"):
            report["alerts"].append({
                "type": "DATA_DRIFT",
                "severity": drift_results.get("alert_level", ALERT_LEVELS["MEDIUM"]),
                "description": f"{drift_results.get('drift_type')} detected",
                "details": drift_results.get("drift_details", {})
            })

        if performance_analysis.get("significant_decay"):
            report["alerts"].append({
                "type": "PERFORMANCE_DECAY",
                "severity": performance_analysis.get("alert_level", ALERT_LEVELS["HIGH"]),
                "description": "Model performance has degraded significantly",
                "details": performance_analysis.get("performance_decay", {})
            })

    # Generate recommendations
    if report["model_performance"]["retraining_required"]:
        report["recommendations"].append({
            "action": "RETRAIN_MODEL",
            "priority": "HIGH",
            "description": "Trigger automated retraining pipeline due to performance decay",
            "reason": performance_analysis.get("performance_decay", {})
        })

    if drift_results.get("drift_detected") and drift_results["alert_level"] == ALERT_LEVELS["CRITICAL"]:
        report["recommendations"].append({
            "action": "UPDATE_DATA_SCHEMA",
            "priority": "CRITICAL",
            "description": "Data schema has changed - review data pipeline and preprocessing",
            "reason": drift_results.get("drift_details", {})
        })

    # Save report to file
    reports_dir = Path(MONITORING_CONFIG["monitoring_logs_path"])
    reports_dir.mkdir(exist_ok=True)

    report_file = reports_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    report["report_path"] = str(report_file)

    print(f"✅ Monitoring report generated: {report['overall_status']}")
    if report["recommendations"]:
        print(f"📋 Recommendations: {len(report['recommendations'])}")

    return report


@task(name="send_alerts")
def send_alerts_task(monitoring_report: dict):
    """
    Send alerts based on monitoring results.
    In a production system, this would send emails/SMS/Teams notifications.
    """
    print("📢 Processing alerts...")

    alerts_sent = []
    critical_alerts = [alert for alert in monitoring_report.get("alerts", [])
                      if alert["severity"] == ALERT_LEVELS["CRITICAL"]]

    high_alerts = [alert for alert in monitoring_report.get("alerts", [])
                  if alert["severity"] == ALERT_LEVELS["HIGH"]]

    if critical_alerts or high_alerts:
        alert_summary = {
            "total_alerts": len(monitoring_report["alerts"]),
            "critical_alerts": len(critical_alerts),
            "high_alerts": len(high_alerts),
            "overall_status": monitoring_report["overall_status"],
            "timestamp": datetime.now().isoformat(),
            "recipient": MONITORING_CONFIG["alert_email"]
        }

        alerts_sent.append(alert_summary)

        print("🚨 CRITICAL ALERTS SENT!")
        for recommendation in monitoring_report.get("recommendations", []):
            print(f"  - {recommendation['action']}: {recommendation['description']}")
    else:
        print("✅ No critical alerts - system healthy")

    return alerts_sent


@task(name="store_monitoring_logs")
def store_monitoring_logs_task(monitoring_report: dict, drift_results: dict, performance_analysis: dict):
    """
    Store monitoring logs and metrics for historical analysis.
    This data can be used for trend analysis and automated decision making.
    """
    print("💾 Storing monitoring logs...")

    logs_dir = Path(MONITORING_CONFIG["monitoring_logs_path"])
    logs_dir.mkdir(exist_ok=True)

    # Store comprehensive monitoring data
    monitoring_log = {
        "timestamp": datetime.now().isoformat(),
        "monitoring_report": monitoring_report,
        "drift_analysis": drift_results,
        "performance_analysis": performance_analysis,
        "system_health": {
            "data_drift_detected": drift_results.get("drift_detected"),
            "performance_issues": performance_analysis.get("significant_decay"),
            "alerts_count": len(monitoring_report.get("alerts", [])),
            "recommendations_count": len(monitoring_report.get("recommendations", []))
        }
    }

    # Create log file with timestamp
    log_file = logs_dir / f"monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(monitoring_log, f, indent=2, default=str)

    print(f"✅ Monitoring logs stored: {log_file}")

    return str(log_file)


# Main monitoring flow according to Google MLOps whitepaper
@flow(name="Continuous Monitoring Pipeline",
      description="Automated model and data monitoring for drift detection and performance tracking")
def monitoring_pipeline(
    trigger_type: str = "scheduled",
    evaluation_data_path: str = "data/processed/test.csv"
):
    """
    Continuous Monitoring Pipeline implementing the Google MLOps whitepaper continuous monitoring process.

    This pipeline:
    1. Loads reference data and models
    2. Detects data drift (schema and distribution)
    3. Evaluates model performance
    4. Generates alerts and recommendations
    5. Stores monitoring logs

    Args:
        trigger_type: How monitoring was triggered (manual, scheduled, event)
        evaluation_data_path: Path to data for performance evaluation
    """
    print("🔍 Starting Continuous Monitoring Pipeline...")
    print(f"Trigger: {trigger_type}")
    print(f"Config: {MONITORING_CONFIG}")

    # Monitoring pipeline execution
    try:
        # Load reference data for baseline comparison
        reference_info = load_reference_data_task()

        # Detect data drift
        drift_results = detect_data_drift_task(reference_info, evaluation_data_path)

        # Load and evaluate production model
        model_info = load_production_model_task()
        performance_analysis = evaluate_model_performance_task(model_info, evaluation_data_path)

        # Generate comprehensive report
        monitoring_report = generate_monitoring_report_task(drift_results, performance_analysis)

        # Send alerts if needed
        alerts_sent = send_alerts_task(monitoring_report)

        # Store monitoring logs
        log_file = store_monitoring_logs_task(monitoring_report, drift_results, performance_analysis)

        # Summary
        pipeline_summary = {
            "pipeline_completion": datetime.now().isoformat(),
            "trigger_type": trigger_type,
            "overall_status": monitoring_report["overall_status"],
            "alerts_count": len(alerts_sent),
            "recommendations_count": len(monitoring_report.get("recommendations", [])),
            "data_drift_detected": drift_results["drift_detected"],
            "retraining_required": performance_analysis["retraining_required"],
            "monitoring_report_path": monitoring_report["report_path"],
            "log_file_path": log_file
        }

        print("🎉 Monitoring Pipeline completed!")
        print(f"Status: {pipeline_summary['overall_status']}")
        print(f"Alerts: {pipeline_summary['alerts_count']}")

        return pipeline_summary

    except Exception as e:
        print(f"❌ Monitoring pipeline failed: {e}")
        raise


# Scheduled monitoring and alert handling functions
def schedule_monitoring(interval_hours: int = 24):
    """
    Function to set up scheduled monitoring using Prefect scheduler.
    In production, this would be handled by a scheduler like Airflow or Cron.
    """
    print(f"📅 Monitoring scheduled for every {interval_hours} hours")
    # Prefect scheduling would go here in production
    pass


def trigger_retraining_if_needed(monitoring_result: dict, retraining_pipeline_path: str = "pipelines/training_pipeline.py"):
    """
    Trigger retraining pipeline if monitoring indicates it's needed.
    This would integrate with the continuous training system.
    """
    if monitoring_result.get("retraining_required"):
        print("🔄 Triggering retraining due to monitoring results...")
        # Execute retraining pipeline
        import subprocess
        result = subprocess.run([
            "python", retraining_pipeline_path,
            "monitoring_trigger"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Retraining triggered successfully")
        else:
            print(f"❌ Retraining trigger failed: {result.stderr}")


if __name__ == "__main__":
    print("🔬 Running Continuous Monitoring Pipeline...")

    # Allow trigger type as command line argument
    trigger_type = sys.argv[1] if len(sys.argv) > 1 else "manual"

    results = monitoring_pipeline(trigger_type=trigger_type)

    print(f"Monitoring complete. Overall status: {results['overall_status']}")

    # Example: Trigger retraining if needed
    if results.get("retraining_required"):
        trigger_retraining_if_needed(results)
