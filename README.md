# Fraud-Detection

## Project Overview & Architecture

The goal is to build an end-to-end system that not only classifies credit card transactions but also operationalizes the entire machine learning lifecycle. This includes automated data validation, experiment tracking, model training, deployment, and continuous monitoring for performance degradation or data drift.

## The architecture can be visualized as a continuous loop:

- Data Ingestion & Versioning: Raw data is pulled and versioned with DVC.

- Orchestrated Pipeline (Prefect): A Prefect flow automates preprocessing, training, and evaluation.

- Experiment Tracking (MLflow): All training runs, parameters, metrics, and models are logged. The best model is registered.

- Deployment (BentoML): The registered model is packaged into a containerized API service.

- Serving: The BentoML service serves real-time prediction requests.

- Monitoring (Evidently & Prometheus): A monitoring pipeline continuously checks for data drift. Metrics are exposed for visualization.

- CI/CD (GitHub Actions): Git pushes trigger automated testing, training, and deployment.

- Retraining Trigger: If monitoring detects significant drift, the training pipeline is automatically re-triggered.
