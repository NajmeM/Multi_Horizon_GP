#!/usr/bin/env python3
"""
Main script for multi-horizon glucose prediction.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from glucose_prediction.data.preprocessing import load_and_preprocess_data, validate_data_quality
from glucose_prediction.data.sequence_generator import create_sequences
from glucose_prediction.models.multi_horizon_model import train_and_evaluate_multi_horizon_model
from glucose_prediction.evaluation.clarke_error_grid import evaluate_glucose_predictions_with_clarke
from glucose_prediction.visualization.plotting import (
    plot_training_history, plot_original_scale_predictions, 
    plot_glucose_predictions_comparison
)
from glucose_prediction.utils.config import ExperimentConfig, setup_logging

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Multi-horizon glucose prediction using deep learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, required=True, 
                       help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="results", 
                       help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--save-model", type=str, default=None,
                       help="Path to save the trained model")
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, "glucose_prediction.log")
    setup_logging(args.log_level, log_file)
    
    logger.info("Starting multi-horizon glucose prediction")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            config = ExperimentConfig.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = ExperimentConfig()
            logger.info("Using default configuration")
        
        # Validate input file
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"Data file not found: {args.data}")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df, df_scaled, scaler_x, scaler_y = load_and_preprocess_data(
            args.data, 
            config.model.sequence_length,
            config.data.features
        )
        
        # Validate data quality
        validation_results = validate_data_quality(df)
        logger.info(f"Data validation completed: {validation_results['total_records']} records")
        
        if validation_results['anomalies']['negative_glucose'] > 0:
            logger.warning(f"Found {validation_results['anomalies']['negative_glucose']} negative glucose values")
        
        # Create sequences
        logger.info("Creating sequences...")
        X, y_60, y_120 = create_sequences(
            df_scaled, 
            config.model.sequence_length,
            config.model.prediction_horizon_60,
            config.model.prediction_horizon_120
        )
        
        logger.info(f"Data shapes: X: {X.shape}, y_60min: {y_60.shape}, y_120min: {y_120.shape}")
        
        # Prepare model save path
        model_save_path = args.save_model or os.path.join(args.output, "best_model.h5")
        
        # Train and evaluate multi-horizon model
        logger.info("Training multi-horizon model...")
        model, history, eval_data, results = train_and_evaluate_multi_horizon_model(
            X, y_60, y_120, 
            config.model.sequence_length, 
            X.shape[2], 
            scaler_y,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            validation_split=config.data.validation_split,
            patience=config.model.patience,
            model_save_path=model_save_path,
            lstm_units_60=config.model.lstm_units_60,
            lstm_units_120=config.model.lstm_units_120,
            cnn_filters=config.model.cnn_filters,
            dropout_rate=config.model.dropout_rate,
            l2_reg=config.model.l2_reg
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        if config.evaluation.save_plots:
            plot_dir = os.path.join(args.output, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_training_history(
                history, 
                os.path.join(plot_dir, f"training_history.{config.evaluation.plot_format}")
            )
            plot_original_scale_predictions(
                eval_data, 
                scaler_y,
                os.path.join(plot_dir, f"predictions_original_scale.{config.evaluation.plot_format}")
            )
        else:
            plot_training_history(history)
            plot_original_scale_predictions(eval_data, scaler_y)
        
        # Perform Clarke Error Grid Analysis
        if config.evaluation.clarke_analysis:
            logger.info("Performing Clarke Error Grid Analysis...")
            clarke_results = evaluate_glucose_predictions_with_clarke(eval_data, scaler_y)
            
            # Print clinical significance summary
            print("\nClinical Significance Summary:")
            print("============================")
            print(f"60-minute predictions clinical acceptability: {clarke_results['60min']['clinical_acceptability']:.1f}%")
            print(f"120-minute predictions clinical acceptability: {clarke_results['120min']['clinical_acceptability']:.1f}%")
            
            # Compare model performance
            print("\nComparison between 60-minute and 120-minute predictions:")
            print("--------------------------------------------------------")
            print(f"Statistical accuracy - 60-min R²: {results['60min']['metrics']['r2']:.4f}, 120-min R²: {results['120min']['metrics']['r2']:.4f}")
            print(f"Clinical accuracy - 60-min Zone A: {clarke_results['60min']['percentages'][0]:.1f}%, 120-min Zone A: {clarke_results['120min']['percentages'][0]:.1f}%")
            
            # Draw conclusion
            if clarke_results['60min']['percentages'][0] > clarke_results['120min']['percentages'][0]:
                print("\nConclusion: The 60-minute prediction model shows better clinical accuracy.")
            else:
                print("\nConclusion: The 120-minute prediction model shows better clinical accuracy.")
        
        # Save results
        if config.evaluation.generate_reports:
            results_file = os.path.join(args.output, "results.json")
            import json
            
            # Prepare results for JSON serialization
            json_results = {
                'model_config': config.model.__dict__,
                'data_config': config.data.__dict__,
                'validation_results': validation_results,
                'model_metrics': {
                    '60min': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                             for k, v in results['60min']['metrics'].items()},
                    '120min': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in results['120min']['metrics'].items()}
                }
            }
            
            if config.evaluation.clarke_analysis:
                json_results['clarke_results'] = clarke_results
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()