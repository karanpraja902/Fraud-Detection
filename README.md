# 💳 Fraud-Detection

This project demonstrates a complete MLOps lifecycle for a credit card fraud detection system, implementing Google MLOps whitepaper best practices. It covers the entire machine learning pipeline from data ingestion to production deployment.

## 🏛️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Data Pipeline  │───▶│  Model Training │
│   (Credit Card  │    │  (Preprocessing │    │  (MLflow)       │
│    Transactions)│    │   + Validation) │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   CI/CD         │    │   Model Serving  │
│   (Drift + Perf)│    │   (GitHub       │    │   (Flask REST    │
│                 │    │    Actions)     │    │    API)          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

- **Data Versioning**: DVC (Git-based data versioning)
- **Experiment Tracking**: MLflow (model versioning, metrics, artifacts)
- **Pipeline Orchestration**: Prefect (workflow automation)
- **Model Serving**: Flask REST API (real-time predictions)
- **Monitoring**: Custom drift detection + performance monitoring
- **CI/CD**: GitHub Actions (automated testing & deployment)
- **Infrastructure**: Docker + Kubernetes (containerization & orchestration)
- **Development**: Python 3.10, scikit-learn, pandas

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Setup
```bash
# Clone repository
git clone https://github.com/karanpraja902/Fraud-Detection.git
cd Fraud-Detection

# Install dependencies
pip install -r requirements.txt

# Verify data exists
ls data/raw/ data/processed/
```

### 🎯 Serve Model (Simplest Approach)
```bash
# One-command solution: retrain + serve
python retrain_and_serve.py

# Web interface available at http://localhost:3000
```

### 🌐 Web Interface
The web interface provides a user-friendly way to test fraud detection:

1. **Simple Form**: Only requires Transaction Time and Amount
2. **Smart Backend**: Automatically uses representative sample data for advanced features
3. **Real-time Results**: Instant analysis with confidence scores
4. **Visual Feedback**: Clear indicators for normal vs fraudulent transactions

### 🧪 Test the API
```bash
# Health check
curl http://localhost:3000/health

# JSON API prediction (full features)
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 125.5, "Amount": 49.99, "...": "all features"}'

# Web form prediction (simplified)
curl -X POST http://localhost:3000/predict/form \
  -d "Time=125.50&Amount=49.99"
```

## 🔄 MLOps Pipeline Steps

This project implements the complete MLOps lifecycle following Google MLOps whitepaper principles:

### 1. 📊 **Data Management**
- **Data Ingestion**: Raw credit card transaction data (284K+ samples)
- **Data Versioning**: DVC tracks data changes and enables reproducibility
- **Data Validation**: Automated quality checks and anomaly detection
- **Preprocessing**: Feature scaling, normalization, and imbalance handling

```bash
# Check data status
dvc status

# Pull latest data (if using remote storage)
dvc pull
```

### 2. 🏗️ **Model Development**
- **Experiment Tracking**: MLflow logs all experiments, parameters, and metrics
- **Hyperparameter Tuning**: Automated optimization using Optuna
- **Model Training**: Multiple algorithms with cross-validation
- **Model Evaluation**: Comprehensive metrics (AUC, precision, recall, F1)

```bash
# Run training pipeline
python pipelines/training_pipeline.py

# View experiments in MLflow UI
mlflow ui
```

### 3. ✅ **Model Validation & Testing**
- **Cross-Validation**: Robust performance estimation
- **Business Metrics**: Fraud detection specific KPIs
- **Model Registry**: Version control and staging (Development → Staging → Production)
- **Automated Testing**: Unit tests and integration tests

```bash
# Run tests
make test

# Check code quality
make lint
```

### 4. 🚀 **Model Deployment**
- **Containerization**: Docker images for consistent deployment
- **Orchestration**: Kubernetes manifests for production scaling
- **CI/CD**: GitHub Actions automate testing and deployment
- **Blue-Green Deployment**: Zero-downtime releases with rollback capability

```bash
# Build and deploy
make build
make deploy

# Or use Docker Compose for local testing
docker-compose -f infra/docker/docker-compose.yml up -d
```

### 5. 👁️ **Monitoring & Observability**
- **Data Drift Detection**: Monitors feature distribution changes
- **Model Performance**: Tracks accuracy decay over time
- **Automated Alerts**: Notifications for critical issues
- **Retraining Triggers**: Automatic model updates when needed

```bash
# Run monitoring pipeline
python pipelines/monitoring_pipeline.py

# View monitoring logs
ls monitoring_logs/
```

### 6. 🔒 **Production Serving**
- **REST API**: Flask-based HTTP endpoints for real-time predictions
- **Health Checks**: Automated service monitoring
- **Load Balancing**: Multiple replicas for high availability
- **Security**: Non-root containers, minimal attack surface

```bash
# Start production server
python retrain_and_serve.py

# API Endpoints:
# GET  /health     - Service health check
# GET  /           - API information
# POST /predict    - Real-time fraud prediction
```

## 📁 Project Structure

```
Fraud-Detection/
├── 📊 data/                    # Data management
│   ├── raw/                    # Raw transaction data
│   └── processed/              # Preprocessed features
├── 🤖 models/                  # Trained models & artifacts
├── 🔧 src/                     # Source code
│   ├── data/                   # Data processing scripts
│   └── models/                 # ML model code
├── 🔄 pipelines/               # MLOps pipelines
│   ├── training_pipeline.py    # Automated training
│   ├── deployment_pipeline.py  # Deployment automation
│   └── monitoring_pipeline.py  # Performance monitoring
├── 🐳 infra/                   # Infrastructure as code
│   ├── docker/                 # Container definitions
│   └── k8s/                    # Kubernetes manifests
├── 🧪 tests/                   # Test suites
├── 📈 mlruns/                  # MLflow experiment logs
├── 🔄 mlartifacts/             # MLflow model artifacts
└── 📋 requirements.txt         # Python dependencies
```

## 🎯 Key Features

- **🔄 Continuous Training**: Automated model retraining on new data
- **📊 Experiment Tracking**: Full lineage from data to predictions
- **🔍 Data Validation**: Automated drift detection and quality checks
- **🚀 One-Click Deployment**: Docker + Kubernetes for production
- **📈 Performance Monitoring**: Real-time model health tracking
- **🔒 Production Ready**: Security hardened, scalable architecture

## 🛠️ Available Commands

```bash
# Development
make install          # Install dependencies
make test            # Run test suite
make lint            # Code quality checks
make serve           # Start local model server

# Deployment
make build           # Build Docker image
make deploy          # Deploy to Kubernetes
docker-compose up    # Start local stack

# Pipelines
make train           # Run training pipeline
make monitor         # Run monitoring pipeline

# Cleanup
make clean           # Remove artifacts
```

## 📊 Model Performance

Current model achieves:
- **AUC-ROC**: 0.967 (excellent fraud detection)
- **Precision**: High precision on fraud predictions
- **Recall**: Strong recall for catching fraudulent transactions
- **F1-Score**: Balanced performance metric

## 🔮 Future Enhancements

- [ ] **A/B Testing**: Compare model versions in production
- [ ] **Feature Store**: Centralized feature management
- [ ] **Model Explainability**: SHAP/LIME integration
- [ ] **Multi-Model Serving**: Ensemble predictions
- [ ] **Advanced Monitoring**: Prometheus + Grafana dashboards

---

**Built with ❤️ following MLOps best practices**
```
