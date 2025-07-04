#!/usr/bin/env python3
"""
Main script for multi-horizon glucose prediction.
"""

import argparse
import yaml
from pathlib import Path

from src.data.preprocessing import load_and_preprocess_data
from src.data.sequence_generator import create_sequences
from src.models.multi_horizon_model import train_and_evaluate_multi_horizon_model
from src.evaluation.clarke_error_grid import evaluate_glucose_predictions_with_clarke
from src.visualization.plots import plot_training_history, plot_original_scale_predictions
from src.utils.config import ModelConfig, DataConfig

def main():
    parser = argparse.ArgumentParser(description="Multi-horizon glucose prediction")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, required=True, 
                       help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="results", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = ModelConfig(**config['model'])
    data_config = DataConfig(**config['data'])
    
    # Your main execution logic here
    print("Starting multi-horizon glucose prediction...")
    
    # Load and preprocess data
    df, df_scaled, scaler_x, scaler_y = load_and_preprocess_data(args.data)
    
    # Create sequences
    X, y_60, y_120 = create_sequences(
        df_scaled, 
        model_config.sequence_length,
        model_config.prediction_horizon_60,
        model_config.prediction_horizon_120
    )
    
    # Train model
    model, history, eval_data, results = train_and_evaluate_multi_horizon_model(
        X, y_60, y_120, model_config.sequence_length, X.shape[2], scaler_y
    )
    
    # Evaluate with Clarke Error Grid
    clarke_results = evaluate_glucose_predictions_with_clarke(eval_data, scaler_y)
    
    # Generate plots
    plot_training_history(history)
    plot_original_scale_predictions(eval_data, scaler_y)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()