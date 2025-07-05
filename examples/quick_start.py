#!/usr/bin/env python3
"""
Quick start example for multi-horizon glucose prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf

from glucose_prediction.data.preprocessing import load_and_preprocess_data
from glucose_prediction.data.sequence_generator import create_sequences
from glucose_prediction.models.multi_horizon_model import train_and_evaluate_multi_horizon_model
from glucose_prediction.evaluation.clarke_error_grid import evaluate_glucose_predictions_with_clarke
from glucose_prediction.visualization.plotting import plot_training_history
from glucose_prediction.utils.config import ExperimentConfig

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


def main():
    """Quick start example."""
    # Configuration
    config = ExperimentConfig()
    
    # Update this path to your data file
    data_path = "../data/raw/sample_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable with your actual data file location")
        return
    
    print("=== Multi-Horizon Glucose Prediction Quick Start ===")
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df, df_scaled, scaler_x, scaler_y = load_and_preprocess_data(data_path)
    print(f"   Loaded {len(df)} records")
    
    # Step 2: Create sequences
    print("\n2. Creating sequences...")
    X, y_60, y_120 = create_sequences(
        df_scaled, 
        config.model.sequence_length,
        config.model.prediction_horizon_60,
        config.model.prediction_horizon_120
    )
    print(f"   Created {len(X)} sequences")
    
    # Step 3: Train model (with reduced epochs for quick demo)
    print("\n3. Training model...")
    model, history, eval_data, results = train_and_evaluate_multi_horizon_model(
        X, y_60, y_120, 
        config.model.sequence_length, 
        X.shape[2], 
        scaler_y,
        epochs=10,  # Reduced for quick demo
        batch_size=32,
        validation_split=0.2
    )
    
    # Step 4: Evaluate results
    print("\n4. Evaluating results...")
    clarke_results = evaluate_glucose_predictions_with_clarke(eval_data, scaler_y)
    
    # Step 5: Display summary
    print("\n5. Results Summary:")
    print("=" * 50)
    print(f"60-minute predictions:")
    print(f"  - R² Score: {results['60min']['metrics']['r2']:.4f}")
    print(f"  - Clinical Acceptability: {clarke_results['60min']['clinical_acceptability']:.1f}%")
    print(f"120-minute predictions:")
    print(f"  - R² Score: {results['120min']['metrics']['r2']:.4f}")
    print(f"  - Clinical Acceptability: {clarke_results['120min']['clinical_acceptability']:.1f}%")
    
    print("\nQuick start completed! Check the generated plots for detailed analysis.")


if __name__ == "__main__":
    main()