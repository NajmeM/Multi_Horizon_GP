import pandas as pd
import numpy as np
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def load_and_preprocess_data(file_path: str, sequence_length: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """Load and preprocess glucose data from CSV."""
    df = pd.read_csv(file_path, sep=';')
    
    # Convert the time column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Create a combined insulin feature
    df['insulin'] = df['basal_rate'] + df['bolus_volume_delivered']
    
    # Select the features we'll use
    features = ['insulin', 'calories', 'steps', 'carb_input', 'glucose']
    
    # Scale the data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Fit the scaler on the entire dataset for features
    df_scaled = pd.DataFrame(scaler_x.fit_transform(df[features]), columns=features)
    
    # For y (glucose), we'll keep it separate
    glucose_values = df['glucose'].values.reshape(-1, 1)
    scaler_y.fit(glucose_values)
    
    return df, df_scaled, scaler_x, scaler_y


def create_patient_output_folder(file_path: str, base_output_dir: str = "patient_results") -> Tuple[str, str, str]:
    """Create a unique output folder for each patient based on filename and timestamp."""
    # Extract patient ID from filename
    filename = os.path.basename(file_path)
    patient_id = os.path.splitext(filename)[0]  # Remove .csv extension
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create patient-specific folder structure
    patient_folder = os.path.join(base_output_dir, patient_id)
    run_folder = os.path.join(patient_folder, f"run_{timestamp}")
    
    # Create directories if they don't exist
    os.makedirs(run_folder, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    subdirs = ['models', 'plots', 'results', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(run_folder, subdir), exist_ok=True)
    
    return run_folder, patient_id, timestamp