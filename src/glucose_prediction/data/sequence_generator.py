"""
Sequence generation module for time series data.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def create_sequences(
    df_scaled: pd.DataFrame, 
    sequence_length: int = 12, 
    prediction_horizon_60: int = 12, 
    prediction_horizon_120: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Parameters:
    -----------
    df_scaled : pd.DataFrame
        Scaled dataframe containing features
    sequence_length : int, default=12
        Length of input sequences (12 = 1 hour with 5-min intervals)
    prediction_horizon_60 : int, default=12
        Prediction horizon for 60-minute predictions
    prediction_horizon_120 : int, default=24
        Prediction horizon for 120-minute predictions
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Input sequences (X), 60-min targets (y_60), 120-min targets (y_120)
    """
    logger.info(f"Creating sequences with length {sequence_length}")
    
    feature_columns = df_scaled.columns
    
    # Create sequences and targets
    X = []
    y_60 = []
    y_120 = []
    
    max_horizon = max(prediction_horizon_60, prediction_horizon_120)
    total_sequences = len(df_scaled) - sequence_length - max_horizon
    
    if total_sequences <= 0:
        raise ValueError(f"Dataset too small. Need at least {sequence_length + max_horizon} records, got {len(df_scaled)}")
    
    for i in range(total_sequences):
        # Input sequence
        X.append(df_scaled.iloc[i:i+sequence_length].values)
        
        # Target: Glucose level 60 minutes ahead
        target_idx_60 = i + sequence_length + prediction_horizon_60
        if target_idx_60 < len(df_scaled):
            y_60.append(df_scaled.iloc[target_idx_60]['glucose'])
        
        # Target: Glucose level 120 minutes ahead
        target_idx_120 = i + sequence_length + prediction_horizon_120
        if target_idx_120 < len(df_scaled):
            y_120.append(df_scaled.iloc[target_idx_120]['glucose'])
    
    X = np.array(X)
    y_60 = np.array(y_60)
    y_120 = np.array(y_120)
    
    logger.info(f"Created {len(X)} sequences with shapes: X={X.shape}, y_60={y_60.shape}, y_120={y_120.shape}")
    
    return X, y_60, y_120


def create_sequences_with_metadata(
    df_scaled: pd.DataFrame,
    df_original: pd.DataFrame,
    sequence_length: int = 12,
    prediction_horizon_60: int = 12,
    prediction_horizon_120: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create sequences with metadata for tracking timestamps and indices.
    
    Parameters:
    -----------
    df_scaled : pd.DataFrame
        Scaled dataframe
    df_original : pd.DataFrame
        Original dataframe with timestamps
    sequence_length : int
        Length of input sequences
    prediction_horizon_60 : int
        60-minute prediction horizon
    prediction_horizon_120 : int
        120-minute prediction horizon
        
    Returns:
    --------
    Tuple containing sequences and metadata dataframe
    """
    X, y_60, y_120 = create_sequences(df_scaled, sequence_length, prediction_horizon_60, prediction_horizon_120)
    
    # Create metadata
    metadata = []
    max_horizon = max(prediction_horizon_60, prediction_horizon_120)
    
    for i in range(len(df_scaled) - sequence_length - max_horizon):
        metadata.append({
            'sequence_start_idx': i,
            'sequence_end_idx': i + sequence_length - 1,
            'target_60_idx': i + sequence_length + prediction_horizon_60,
            'target_120_idx': i + sequence_length + prediction_horizon_120,
            'start_time': df_original.iloc[i]['time'],
            'end_time': df_original.iloc[i + sequence_length - 1]['time'],
            'target_60_time': df_original.iloc[i + sequence_length + prediction_horizon_60]['time'] if i + sequence_length + prediction_horizon_60 < len(df_original) else None,
            'target_120_time': df_original.iloc[i + sequence_length + prediction_horizon_120]['time'] if i + sequence_length + prediction_horizon_120 < len(df_original) else None
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    return X, y_60, y_120, metadata_df