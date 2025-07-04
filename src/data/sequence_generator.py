import numpy as np
import pandas as pd

from typing import Tuple

def create_sequences(df_scaled: pd.DataFrame, 
                    sequence_length: int = 12,
                    prediction_horizon_60: int = 12,
                    prediction_horizon_120: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for time series prediction."""
    feature_columns = df_scaled.columns
    
    # Create sequences and targets
    X = []
    y_60 = []
    y_120 = []
    
    for i in range(len(df_scaled) - max(sequence_length, prediction_horizon_120)):
        # Input sequence
        X.append(df_scaled.iloc[i:i+sequence_length].values)
        
        # Target: Glucose level 60 minutes 
        y_60.append(df_scaled.iloc[i+prediction_horizon_60]['glucose'])
        
        # Target: Glucose level 120 minutes 
        y_120.append(df_scaled.iloc[i+prediction_horizon_120]['glucose'])
    
    return np.array(X), np.array(y_60), np.array(y_120)