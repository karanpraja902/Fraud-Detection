# 💳 Fraud-Detection

End-to-end fraud detection system implementing production MLOps practices. Features automated pipelines, model versioning, and production deployment.

## 🏗️ Architecture

```
Data → Preprocessing → Training → Validation → Deployment → Monitoring
```

**Design Decisions:**
- **Modular Pipeline**: Separated concerns for maintainability and testing
- **MLflow Integration**: Centralized experiment tracking and model registry
- **Docker + Kubernetes**: Containerized deployment for consistency across environments
- **Flask REST API**: Lightweight, production-ready serving layer
- **Automated Retraining**: Continuous model improvement based on performance metrics

## 🛠️ Tech Stack

- **ML Framework**: scikit-learn, pandas, numpy
- **MLOps Tools**: MLflow (tracking), DVC (data versioning), BentoML (serving)
- **Infrastructure**: Docker, Kubernetes, Flask
- **Development**: Python 3.10, Jupyter notebooks

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Setup
```bash
git clone https://github.com/karanpraja902/Fraud-Detection.git
cd Fraud-Detection
pip install -r requirements.txt
```

### 🎯 Serve Model
```bash
# One-command solution: retrain + serve
python retrain_and_serve.py
# Web interface at http://localhost:3000
```

### 📊 Run EDA Analysis
```bash
# Execute all EDA notebooks
./run_eda_simple.sh

# Or run individual notebooks
jupyter notebook notebooks/
```

### 🧪 Test API
```bash
curl http://localhost:3000/health
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 125.5, "Amount": 49.99}'
```

## 🔄 MLOps Pipeline

1. **Data Management**: DVC versioning, automated preprocessing
2. **Model Development**: MLflow tracking, cross-validation, hyperparameter tuning
3. **Validation & Testing**: Automated testing, performance metrics
4. **Deployment**: Docker + Kubernetes, CI/CD automation
5. **Monitoring**: Drift detection, performance tracking, automated retraining
6. **Production Serving**: REST API with health checks and load balancing

## 📊 EDA Notebooks

**Design Decision:** Implemented comprehensive exploratory data analysis as the foundation of the MLOps pipeline, addressing the critical gap in data science workflows.

### 📈 01_exploration.ipynb
- **Purpose**: Complete data profiling and statistical analysis
- **Key Insights**: 577:1 class imbalance, PCA feature characteristics, correlation analysis
- **Design**: Automated execution with comprehensive visualizations

### 🎯 02_baseline_model.ipynb
- **Purpose**: Establish performance baselines for fraud detection
- **Models**: Logistic Regression, Random Forest with class balancing
- **Metrics**: AUC-ROC, Precision-Recall curves, feature importance analysis
- **Design**: Cross-validation and statistical significance testing

### 🔬 03_experiments.ipynb
- **Purpose**: Advanced experimentation framework for model improvement
- **Techniques**: SMOTE sampling, XGBoost/LightGBM, hyperparameter optimization
- **Design**: Modular approach for easy extension and comparison

## 📁 Project Structure

```
Fraud-Detection/
├── 📊 data/                    # Data management
│   ├── raw/                    # Raw transaction data
│   └── processed/              # Preprocessed features
├── 📓 notebooks/               # EDA and experimentation
│   ├── 01_exploration.ipynb    # Data profiling & analysis
│   ├── 02_baseline_model.ipynb # Baseline model development
│   └── 03_experiments.ipynb    # Advanced techniques
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

---

**Built with ❤️ following MLOps best practices**
```
