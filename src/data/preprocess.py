import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(path)


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the 'Amount' and 'Time' features."""
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["Time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    return df


def handle_imbalance(X: pd.DataFrame, y: pd.Series):
    """Handles class imbalance using undersampling and SMOTE."""
    # Note: A real-world approach might be more complex
    undersample = RandomUnderSampler(sampling_strategy=0.1)
    X_under, y_under = undersample.fit_resample(X, y)

    smote = SMOTE(sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X_under, y_under)

    return X_resampled, y_resampled


def run_preprocessing(input_path: str, output_path: str):
    """Main function to run all preprocessing steps."""
    df = load_data(input_path)
    df = normalize_features(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_processed, y_train_processed = handle_imbalance(X_train, y_train)

    # Save processed data
    pd.concat([X_train_processed, y_train_processed], axis=1).to_csv(
        f"{output_path}/train.csv", index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(f"{output_path}/test.csv", index=False)


if __name__ == "__main__":
    input_path = "data/raw/creditcard.csv"  # path to my raw dataset
    output_path = "data/processed"  # folder where train/test will be saved

    run_preprocessing(input_path, output_path)
