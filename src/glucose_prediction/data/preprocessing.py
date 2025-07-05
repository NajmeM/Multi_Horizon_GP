"""
Data preprocessing module for glucose prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_and_preprocess_data(
    file_path: str, 
    sequence_length: int = 12,
    features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """
    Load and preprocess the glucose data from CSV.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing glucose data
    sequence_length : int, default=12
        Length of input sequences
    features : List[str], optional
        List of features to use. If None, uses default features.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler]
        Original dataframe, scaled dataframe, feature scaler, glucose scaler
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path, sep=';')
        logger.info(f"Loaded {len(df)} records")
        
        # Convert the time column to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Create a combined insulin feature
        df['insulin'] = df['basal_rate'] + df['bolus_volume_delivered']
        
        # Select the features we'll use
        if features is None:
            features = ['insulin', 'calories', 'steps', 'carb_input', 'glucose']
        
        # Validate that all features exist in the dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")
        
        # Scale the data
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Fit the scaler on the entire dataset for features
        df_scaled = pd.DataFrame(scaler_x.fit_transform(df[features]), columns=features)
        
        # For y (glucose), we'll keep it separate
        glucose_values = df['glucose'].values.reshape(-1, 1)
        scaler_y.fit(glucose_values)
        
        logger.info("Data preprocessing completed successfully")
        return df, df_scaled, scaler_x, scaler_y
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate the quality of the glucose data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to validate
        
    Returns:
    --------
    dict
        Dictionary containing validation results
    """
    validation_results = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'glucose_range': {
            'min': df['glucose'].min(),
            'max': df['glucose'].max(),
            'mean': df['glucose'].mean(),
            'std': df['glucose'].std()
        },
        'time_range': {
            'start': df['time'].min(),
            'end': df['time'].max(),
            'duration_days': (df['time'].max() - df['time'].min()).days
        }
    }
    
    # Check for anomalies
    validation_results['anomalies'] = {
        'negative_glucose': (df['glucose'] < 0).sum(),
        'extreme_high_glucose': (df['glucose'] > 600).sum(),
        'extreme_low_glucose': (df['glucose'] < 20).sum()
    }
    
    return validation_results


def create_patient_output_folder(file_path: str, base_output_dir: str = "patient_results") -> Tuple[str, str, str]:
    """
    Create a unique output folder for each patient based on filename and timestamp.
    
    Parameters:
    -----------
    file_path : str
        Path to the patient data file
    base_output_dir : str
        Base directory for outputs
        
    Returns:
    --------
    Tuple[str, str, str]
        Output folder path, patient ID, timestamp
    """
    import os
    from datetime import datetime
    
    # Extract patient ID from filename
    filename = os.path.basename(file_path)
    patient_id = os.path.splitext(filename)[0]
    
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