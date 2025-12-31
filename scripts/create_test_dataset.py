#!/usr/bin/env python
"""Create synthetic credit card dataset for CI/CD testing

This script generates a realistic synthetic fraud dataset that matches
the structure and characteristics of the original credit card fraud dataset.
Used for CI/CD testing when Kaggle credentials are not available.
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_synthetic_dataset(output_path='data/raw/creditcard.csv', n_samples=10000):
    """Create a realistic synthetic fraud dataset

    Parameters:
        output_path (str): Path to save the CSV file
        n_samples (int): Number of transactions to generate

    Returns:
        pd.DataFrame: The generated dataset
    """
    np.random.seed(42)  # For reproducible results

    n_pca_features = 28

    # Create PCA features (V1-V28) - normally distributed with some correlation
    # In real data, these are PCA-transformed features, so they're approximately standard normal
    pca_features = np.random.randn(n_samples, n_pca_features)

    # Add some realistic correlations between features (fraud patterns)
    # Fraud transactions often have different patterns in PCA space
    fraud_mask = np.random.binomial(1, 0.0017, n_samples).astype(bool)

    # Make fraud transactions have slightly different distribution in V1-V5
    for i in range(5):
        pca_features[fraud_mask, i] += np.random.normal(0, 0.5, fraud_mask.sum())

    # Time feature - represents seconds elapsed between transactions
    # Real dataset spans ~48 hours (172,792 seconds)
    time = np.sort(np.random.uniform(0, 172792, n_samples))

    # Amount feature - exponential distribution with realistic bounds
    # Most transactions are small, few are large (matches real fraud data)
    amount = np.random.exponential(scale=88.35, size=n_samples)
    amount = np.clip(amount, 0, 25691.16)  # Match real dataset bounds

    # Fraud transactions tend to have higher amounts on average
    amount[fraud_mask] *= np.random.uniform(1.5, 3.0, fraud_mask.sum())

    # Create target with realistic imbalance (0.172% fraud rate in original dataset)
    fraud_rate = 0.00172
    classes = np.random.binomial(1, fraud_rate, n_samples)

    # Create DataFrame with proper column names
    df = pd.DataFrame(
        pca_features,
        columns=[f'V{i}' for i in range(1, n_pca_features + 1)]
    )

    df['Time'] = time
    df['Amount'] = amount
    df['Class'] = classes

    # Ensure proper directory structure exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV (no index column, matches original format)
    df.to_csv(output_path, index=False)

    # Print summary statistics
    print("✅ Created synthetic credit card fraud dataset")
    print(f"   � Output: {output_path}")
    print(f"   �📊 Samples: {n_samples:,}")
    print(f"   🎯 Fraud rate: {df['Class'].mean():.3%} ({df['Class'].sum()} fraud transactions)")
    print(f"   🔢 Features: {len(df.columns) - 1} (V1-V28 + Time + Amount)")
    print(f"   ⏱️  Time range: {df['Time'].min():.0f}s - {df['Time'].max():.0f}s ({df['Time'].max()/3600:.1f} hours)")
    print(f"   💰 Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
    print(f"   📈 Amount mean: ${df['Amount'].mean():.2f}")

    return df

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create synthetic credit card fraud dataset')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples to generate (default: 10000)')
    parser.add_argument('--output', type=str, default='data/raw/creditcard.csv',
                       help='Output file path (default: data/raw/creditcard.csv)')

    args = parser.parse_args()

    # Create dataset with specified parameters
    df = create_synthetic_dataset(n_samples=args.samples, output_path=args.output)

    # Quick validation
    assert len(df) == args.samples, f"Incorrect number of samples: expected {args.samples}, got {len(df)}"
    assert 'Class' in df.columns, "Missing Class column"
    assert df['Class'].nunique() == 2, "Should have binary classification"
    assert df['Class'].sum() > 0, "Should have at least some fraud cases"

    print(f"\n✅ Dataset validation passed for {args.samples} samples!")
