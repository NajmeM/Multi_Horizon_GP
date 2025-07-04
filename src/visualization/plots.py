import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

def plot_training_history(history, output_folder: Optional[str] = None, patient_id: str = ""):
    #Plot training and validation loss for multi-horizon predictions.
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['output_60min_loss'], label='Train Loss (60min)')
    plt.plot(history.history['val_output_60min_loss'], label='Val Loss (60min)')
    plt.title(f'60-Minute Prediction Training History - Patient: {patient_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['output_120min_loss'], label='Train Loss (120min)')
    plt.plot(history.history['val_output_120min_loss'], label='Val Loss (120min)')
    plt.title(f'120-Minute Prediction Training History - Patient: {patient_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'plots', f'training_history_{patient_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_scaled_predictions(eval_data: dict):
    """Plot predictions vs actual values in normalized scale."""
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(eval_data['y_test_60'], label='Actual Glucose (60min)')
    plt.plot(eval_data['y_pred_60'], label='Predicted Glucose (60min)')
    plt.title('60-Minute Glucose Prediction (Normalized Scale)')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Glucose Level')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(eval_data['y_test_120'], label='Actual Glucose (120min)')
    plt.plot(eval_data['y_pred_120'], label='Predicted Glucose (120min)')
    plt.title('120-Minute Glucose Prediction (Normalized Scale)')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Glucose Level')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('multi_horizon_scaled_predictions.png')
    plt.show()

def plot_original_scale_predictions(eval_data: dict, scaler_y):
    #Plot predictions vs actual values in original glucose scale (mg/dL).
    # Convert to original scale
    y_test_60_orig = convert_predictions_to_original_scale(eval_data['y_test_60'], scaler_y)
    y_pred_60_orig = convert_predictions_to_original_scale(eval_data['y_pred_60'], scaler_y)
    y_test_120_orig = convert_predictions_to_original_scale(eval_data['y_test_120'], scaler_y)
    y_pred_120_orig = convert_predictions_to_original_scale(eval_data['y_pred_120'], scaler_y)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_test_60_orig, 'g-', label='Actual Glucose (60min)')
    plt.plot(y_pred_60_orig, 'r--', label='Predicted Glucose (60min)')
    plt.title('60-Minute Glucose Prediction (Original Scale)')
    plt.xlabel('Sample Index')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(y_test_120_orig, 'g-', label='Actual Glucose (120min)')
    plt.plot(y_pred_120_orig, 'r--', label='Predicted Glucose (120min)')
    plt.title('120-Minute Glucose Prediction (Original Scale)')
    plt.xlabel('Sample Index')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('multi_horizon_original_scale_predictions.png')
    plt.show()