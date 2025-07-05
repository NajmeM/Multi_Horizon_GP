#!/usr/bin/env python3
"""
Batch processing example for multiple patient files.
"""

import sys
import os
import glob
import json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
import pandas as pd

from glucose_prediction.data.preprocessing import load_and_preprocess_data, create_patient_output_folder
from glucose_prediction.data.sequence_generator import create_sequences
from glucose_prediction.models.multi_horizon_model import train_and_evaluate_multi_horizon_model
from glucose_prediction.evaluation.clarke_error_grid import evaluate_glucose_predictions_with_clarke
from glucose_prediction.utils.config import ExperimentConfig

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


def process_patient_file(file_path: str, config: ExperimentConfig, output_dir: str) -> dict:
    """
    Process a single patient file.
    
    Parameters:
    -----------
    file_path : str
        Path to patient data file
    config : ExperimentConfig
        Configuration object
    output_dir : str
        Base output directory
        
    Returns:
    --------
    dict
        Processing results
    """
    try:
        print(f"\nProcessing {os.path.basename(file_path)}...")
        
        # Create patient-specific output folder
        patient_output, patient_id, timestamp = create_patient_output_folder(
            file_path, output_dir
        )
        
        # Load and preprocess data
        df, df_scaled, scaler_x, scaler_y = load_and_preprocess_data(file_path)
        
        # Create sequences
        X, y_60, y_120 = create_sequences(
            df_scaled, 
            config.model.sequence_length,
            config.model.prediction_horizon_60,
            config.model.prediction_horizon_120
        )
        
        if len(X) < 50:  # Minimum data requirement
            print(f"   Insufficient data for {patient_id} (only {len(X)} sequences)")
            return {
                'patient_id': patient_id,
                'status': 'insufficient_data',
                'sequences': len(X)
            }
        
        # Train model
        model_save_path = os.path.join(patient_output, 'models', f'{patient_id}_model.h5')
        model, history, eval_data, results = train_and_evaluate_multi_horizon_model(
            X, y_60, y_120, 
            config.model.sequence_length, 
            X.shape[2], 
            scaler_y,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            validation_split=config.data.validation_split,
            patience=config.model.patience,
            model_save_path=model_save_path
        )
        
        # Evaluate with Clarke Error Grid
        clarke_results = evaluate_glucose_predictions_with_clarke(eval_data, scaler_y)
        
        # Save results
        results_data = {
            'patient_id': patient_id,
            'timestamp': timestamp,
            'data_summary': {
                'total_records': len(df),
                'sequences_created': len(X),
                'features_used': config.data.features
            },
            'model_metrics': {
                '60min': {k: float(v) for k, v in results['60min']['metrics'].items()},
                '120min': {k: float(v) for k, v in results['120min']['metrics'].items()}
            },
            'clarke_results': clarke_results,
            'config': config.model.__dict__
        }
        
        results_file = os.path.join(patient_output, 'results', f'{patient_id}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"   ✓ Completed {patient_id}")
        print(f"     60-min R²: {results['60min']['metrics']['r2']:.4f}")
        print(f"     120-min R²: {results['120min']['metrics']['r2']:.4f}")
        print(f"     Clinical acceptability: {clarke_results['60min']['clinical_acceptability']:.1f}% / {clarke_results['120min']['clinical_acceptability']:.1f}%")
        
        return {
            'patient_id': patient_id,
            'status': 'success',
            'results': results_data,
            'output_path': patient_output
        }
        
    except Exception as e:
        print(f"   ✗ Error processing {os.path.basename(file_path)}: {str(e)}")
        return {
            'patient_id': os.path.splitext(os.path.basename(file_path))[0],
            'status': 'error',
            'error': str(e)
        }


def generate_summary_report(batch_results: list, output_dir: str) -> None:
    """Generate a summary report for batch processing."""
    
    successful_results = [r for r in batch_results if r['status'] == 'success']
    
    if not successful_results:
        print("No successful results to summarize.")
        return
    
    # Aggregate statistics
    summary = {
        'total_patients': len(batch_results),
        'successful_patients': len(successful_results),
        'failed_patients': len([r for r in batch_results if r['status'] == 'error']),
        'insufficient_data': len([r for r in batch_results if r['status'] == 'insufficient_data']),
        'aggregate_metrics': {
            '60min': {
                'r2_mean': np.mean([r['results']['model_metrics']['60min']['r2'] for r in successful_results]),
                'r2_std': np.std([r['results']['model_metrics']['60min']['r2'] for r in successful_results]),
                'clinical_acceptability_mean': np.mean([r['results']['clarke_results']['60min']['clinical_acceptability'] for r in successful_results])
            },
            '120min': {
                'r2_mean': np.mean([r['results']['model_metrics']['120min']['r2'] for r in successful_results]),
                'r2_std': np.std([r['results']['model_metrics']['120min']['r2'] for r in successful_results]),
                'clinical_acceptability_mean': np.mean([r['results']['clarke_results']['120min']['clinical_acceptability'] for r in successful_results])
            }
        },
        'individual_results': [
            {
                'patient_id': r['patient_id'],
                '60min_r2': r['results']['model_metrics']['60min']['r2'] if r['status'] == 'success' else None,
                '120min_r2': r['results']['model_metrics']['120min']['r2'] if r['status'] == 'success' else None,
                '60min_clinical': r['results']['clarke_results']['60min']['clinical_acceptability'] if r['status'] == 'success' else None,
                '120min_clinical': r['results']['clarke_results']['120min']['clinical_acceptability'] if r['status'] == 'success' else None,
                'status': r['status']
            }
            for r in batch_results
        ]
    }
    
    # Save summary
    summary_file = os.path.join(output_dir, 'batch_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create summary DataFrame and CSV
    df_summary = pd.DataFrame(summary['individual_results'])
    csv_file = os.path.join(output_dir, 'batch_results.csv')
    df_summary.to_csv(csv_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total patients processed: {summary['total_patients']}")
    print(f"Successful: {summary['successful_patients']}")
    print(f"Failed: {summary['failed_patients']}")
    print(f"Insufficient data: {summary['insufficient_data']}")
    
    if successful_results:
        print(f"\nAggregate Performance (n={len(successful_results)}):")
        print(f"60-min predictions:")
        print(f"  - R² Score: {summary['aggregate_metrics']['60min']['r2_mean']:.4f} ± {summary['aggregate_metrics']['60min']['r2_std']:.4f}")
        print(f"  - Clinical Acceptability: {summary['aggregate_metrics']['60min']['clinical_acceptability_mean']:.1f}%")
        print(f"120-min predictions:")
        print(f"  - R² Score: {summary['aggregate_metrics']['120min']['r2_mean']:.4f} ± {summary['aggregate_metrics']['120min']['r2_std']:.4f}")
        print(f"  - Clinical Acceptability: {summary['aggregate_metrics']['120min']['clinical_acceptability_mean']:.1f}%")
    
    print(f"\nDetailed results saved to:")
    print(f"  - JSON: {summary_file}")
    print(f"  - CSV: {csv_file}")


def main():
    """Main batch processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process multiple patient files")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing patient CSV files")
    parser.add_argument("--output-dir", type=str, default="batch_results",
                       help="Output directory for results")
    parser.add_argument("--pattern", type=str, default="*.csv",
                       help="File pattern to match")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
    
    # Find all patient files
    pattern = os.path.join(args.input_dir, args.pattern)
    patient_files = glob.glob(pattern)
    
    if not patient_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(patient_files)} patient files to process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file
    batch_results = []
    for i, file_path in enumerate(patient_files, 1):
        print(f"\n[{i}/{len(patient_files)}] Processing {os.path.basename(file_path)}")
        result = process_patient_file(file_path, config, args.output_dir)
        batch_results.append(result)
    
    # Generate summary report
    generate_summary_report(batch_results, args.output_dir)
    
    print(f"\nBatch processing completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()