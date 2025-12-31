"""
Configuration file for the fraud detection project.
Centralizes all path and configuration management.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Notebook directories
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# File paths
RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# MLflow configuration
EXPERIMENT_NAME = "fraud_detection_notebooks"

# Plot styling
PLOT_STYLE = "seaborn-v0_8"
