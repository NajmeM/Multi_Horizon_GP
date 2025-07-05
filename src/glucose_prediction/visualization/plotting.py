"""
Plotting and visualization utilities for glucose prediction.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def plot_training_history(history, save_path: Optional[str] = None) -> None:
    """
    Plot training and validation loss for multi-horizon predictions.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 60-minute prediction history
    ax1.plot(history.history['output_60min_loss'], label='Train Loss (60min)', linewidth=2)
    ax1.plot(history.history['val_output_60min_loss'], label='Val Loss (60min)', linewidth=2)
    ax1.set_title('60-Minute Prediction Training History', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 120-minute prediction history
    ax2.plot(history.history['output_120min_loss'], label='Train Loss (120min)', linewidth=2)
    ax2.plot(history.history['val_output_120min_loss'], label='Val Loss (120min)', linewidth=2)
    ax2.set_title('120-Minute Prediction Training History', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_scaled_predictions(eval_data: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot predictions vs actual values in normalized scale.
    
    Parameters:
    -----------
    eval_data : dict
        Dictionary containing evaluation data
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 60-minute predictions
    ax1.plot(eval_data['y_test_60'], label='Actual Glucose (60min)', linewidth=2, alpha=0.8)
    ax1.plot(eval_data['y_pred_60'], label='Predicted Glucose (60min)', linewidth=2, alpha=0.8)
    ax1.set_title('60-Minute Glucose Prediction (Normalized Scale)', fontsize=14)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Normalized Glucose Level', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 120-minute predictions
    ax2.plot(eval_data['y_test_120'], label='Actual Glucose (120min)', linewidth=2, alpha=0.8)
    ax2.plot(eval_data['y_pred_120'], label='Predicted Glucose (120min)', linewidth=2, alpha=0.8)
    ax2.set_title('120-Minute Glucose Prediction (Normalized Scale)', fontsize=14)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Normalized Glucose Level', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Scaled predictions plot saved to {save_path}")
    
    plt.show()


def plot_original_scale_predictions(eval_data: Dict[str, Any], scaler_y, save_path: Optional[str] = None) -> None:
    """
    Plot predictions vs actual values in original glucose scale (mg/dL).
    
    Parameters:
    -----------
    eval_data : dict
        Dictionary containing evaluation data
    scaler_y : sklearn scaler
        Scaler for converting back to original scale
    save_path : str, optional
        Path to save the plot
    """
    from ..models.multi_horizon_model import convert_predictions_to_original_scale
    
    # Convert to original scale
    y_test_60_orig = convert_predictions_to_original_scale(eval_data['y_test_60'], scaler_y)
    y_pred_60_orig = convert_predictions_to_original_scale(eval_data['y_pred_60'], scaler_y)
    y_test_120_orig = convert_predictions_to_original_scale(eval_data['y_test_120'], scaler_y)
    y_pred_120_orig = convert_predictions_to_original_scale(eval_data['y_pred_120'], scaler_y)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 60-minute predictions
    ax1.plot(y_test_60_orig, 'g-', label='Actual Glucose (60min)', linewidth=2, alpha=0.8)
    ax1.plot(y_pred_60_orig, 'r--', label='Predicted Glucose (60min)', linewidth=2, alpha=0.8)
    ax1.set_title('60-Minute Glucose Prediction (Original Scale)', fontsize=14)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Glucose Level (mg/dL)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 120-minute predictions
    ax2.plot(y_test_120_orig, 'g-', label='Actual Glucose (120min)', linewidth=2, alpha=0.8)
    ax2.plot(y_pred_120_orig, 'r--', label='Predicted Glucose (120min)', linewidth=2, alpha=0.8)
    ax2.set_title('120-Minute Glucose Prediction (Original Scale)', fontsize=14)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Glucose Level (mg/dL)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Original scale predictions plot saved to {save_path}")
    
    plt.show()


def create_hourly_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe to hourly resolution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with time column
        
    Returns:
    --------
    pd.DataFrame
        Hourly resampled dataframe
    """
    df_hourly = df.copy()
    df_hourly = df_hourly.set_index('time')
    # Resample to hourly data (mean values)
    hourly_mean = df_hourly.resample('1H').mean()
    # Reset index to make time a column again
    hourly_mean = hourly_mean.reset_index()
    return hourly_mean


def plot_glucose_predictions_comparison(
    df: pd.DataFrame, 
    eval_data: Dict[str, Any], 
    scaler_y,
    X_test_indices: np.ndarray,
    sequence_length: int, 
    prediction_horizon_60: int, 
    prediction_horizon_120: int,
    days_to_show: int = 1,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Plot glucose predictions comparison with ground truth over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with timestamps
    eval_data : dict
        Evaluation data dictionary
    scaler_y : sklearn scaler
        Glucose scaler
    X_test_indices : np.ndarray
        Test set indices
    sequence_length : int
        Input sequence length
    prediction_horizon_60 : int
        60-minute prediction horizon
    prediction_horizon_120 : int
        120-minute prediction horizon
    days_to_show : int
        Number of days to show in plot
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    pd.DataFrame
        Hourly results dataframe
    """
    from ..models.multi_horizon_model import convert_predictions_to_original_scale
    
    # Convert predictions back to original scale
    y_test_60_orig = convert_predictions_to_original_scale(eval_data['y_test_60'], scaler_y)
    y_pred_60_orig = convert_predictions_to_original_scale(eval_data['y_pred_60'], scaler_y)
    y_test_120_orig = convert_predictions_to_original_scale(eval_data['y_test_120'], scaler_y)
    y_pred_120_orig = convert_predictions_to_original_scale(eval_data['y_pred_120'], scaler_y)

    # Collect corresponding timestamps and prediction values
    test_times, actuals, preds_60, preds_120 = [], [], [], []
    for i, idx in enumerate(X_test_indices):
        adjusted_idx = idx + sequence_length + prediction_horizon_60
        if adjusted_idx < len(df):
            test_times.append(df['time'].iloc[adjusted_idx])
            actuals.append(y_test_60_orig[i][0])
            preds_60.append(y_pred_60_orig[i][0])
            preds_120.append(y_pred_120_orig[i][0])

    # Create results DataFrame
    results_df = pd.DataFrame({
        'time': test_times,
        'actual': actuals,
        'pred_60min': preds_60,
        'pred_120min': preds_120
    })

    # Resample to hourly data
    hourly_results = create_hourly_df(results_df)
    start_date = hourly_results['time'].min().normalize()
    end_date = start_date + timedelta(days=days_to_show)
    filtered_data = hourly_results[
        (hourly_results['time'] >= start_date) & 
        (hourly_results['time'] < end_date)
    ]

    # Plot comparison on a single day
    comparison_day = start_date + timedelta(days=min(2, days_to_show-1))
    next_day = comparison_day + timedelta(days=1)
    daily_data = filtered_data[
        (filtered_data['time'] >= comparison_day) & 
        (filtered_data['time'] < next_day)
    ]

    if len(daily_data) == 0:
        logger.warning("No data available for the selected day")
        return hourly_results

    plt.figure(figsize=(16, 8))
    plt.plot(daily_data['time'], daily_data['actual'], 'k-', 
             linewidth=3, label='Ground Truth', alpha=0.8)
    plt.plot(daily_data['time'], daily_data['pred_60min'], 'r--', 
             linewidth=2.5, label='60 min Prediction', alpha=0.8)
    plt.plot(daily_data['time'], daily_data['pred_120min'], 'g-.', 
             linewidth=2.5, label='120 min Prediction', alpha=0.8)
    
    plt.title(f'Glucose Prediction vs Ground Truth ({comparison_day.strftime("%Y-%m-%d")})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Glucose Level (mg/dL)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(40, 350)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    else:
        plt.savefig(f'glucose_comparison_{comparison_day.strftime("%Y-%m-%d")}.png', 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

    return hourly_results