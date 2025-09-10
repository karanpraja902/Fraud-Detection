# 💳 Fraud-Detection

This project demonstrates a full MLOps lifecycle for a credit card fraud detection system. It covers data versioning, pipeline orchestration, experiment tracking, model serving, and automated monitoring.

## 🏛️ Architecture
[Insert architecture diagram here]

## 🛠️ Tech Stack
- **Data Versioning**: DVC
- **Experiment Tracking**: MLflow
- **Pipeline Orchestration**: Prefect
- **Model Serving**: BentoML
- **Monitoring**: Evidently AI, Prometheus, Grafana
- **CI/CD**: GitHub Actions

## 🚀 Getting Started

### Prerequisites
- Python 3.10.14
- Docker

### Setup
1. Clone the repo: `git clone ...`
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Pull data: `dvc pull`

### Running the System
- **Train the model**: `python pipelines/training_pipeline.py`
- **Serve the model**: `bentoml serve fraud_detector:latest`
- **Run monitoring**: `python pipelines/monitoring_pipeline.py`