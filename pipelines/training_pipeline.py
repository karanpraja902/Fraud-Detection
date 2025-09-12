from prefect import task, flow
import sys
# Add src to path to import our modules
sys.path.append('src')
from data.preprocess import run_preprocessing
from models.train import train_model


@task
def preprocess_data_task():
    print("Preprocessing data...")
    run_preprocessing("data/raw/creditcard.csv", "data/processed")
    return "data/processed/train.csv"


@task
def train_model_task(train_path: str):
    print("Training model...")
    train_model(train_path)


@flow(name="Fraud Detection Training Pipeline")
def training_pipeline():
    train_data_path = preprocess_data_task()
    train_model_task(train_data_path)

if __name__ == "__main__":
    training_pipeline()
