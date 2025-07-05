"""
Multi-horizon glucose prediction model implementation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Conv1D, MaxPooling1D, Flatten, Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def build_multi_horizon_model(
    sequence_length: int, 
    n_features: int,
    lstm_units_60: int = 64,
    lstm_units_120: int = 64,
    cnn_filters: int = 64,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001
) -> Model:
    """
    Build a model that predicts both 60min and 120min horizons with regularization.
    
    Parameters:
    -----------
    sequence_length : int
        Length of input sequences
    n_features : int
        Number of input features
    lstm_units_60 : int
        Number of LSTM units for 60-min branch
    lstm_units_120 : int
        Number of LSTM units for 120-min branch
    cnn_filters : int
        Number of CNN filters
    dropout_rate : float
        Dropout rate for regularization
    l2_reg : float
        L2 regularization strength
        
    Returns:
    --------
    Model
        Compiled Keras model
    """
    logger.info(f"Building multi-horizon model with {n_features} features and sequence length {sequence_length}")
    
    # Shared input layer
    input_layer = Input(shape=(sequence_length, n_features))
    
    # LSTM branch for 60-minute predictions with regularization
    lstm_60 = LSTM(
        units=lstm_units_60, 
        return_sequences=True, 
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(input_layer)
    lstm_60 = BatchNormalization()(lstm_60)
    lstm_60 = Dropout(dropout_rate)(lstm_60)
    
    lstm_60 = LSTM(
        units=lstm_units_60//2, 
        kernel_regularizer=l2(l2_reg), 
        recurrent_regularizer=l2(l2_reg)
    )(lstm_60)
    lstm_60 = BatchNormalization()(lstm_60)
    lstm_60 = Dropout(dropout_rate)(lstm_60)
    
    dense_60 = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(lstm_60)
    output_60 = Dense(1, name='output_60min')(dense_60)
    
    # CNN part for feature extraction
    conv1 = Conv1D(
        filters=cnn_filters, 
        kernel_size=3, 
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    )(input_layer)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv1D(
        filters=cnn_filters*2, 
        kernel_size=3, 
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    pool = MaxPooling1D(pool_size=2)(conv2)
    
    # LSTM branch for temporal patterns
    lstm = LSTM(
        lstm_units_120, 
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(input_layer)
    lstm = BatchNormalization()(lstm)
    
    lstm = LSTM(
        lstm_units_120//2,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(lstm)
    lstm = BatchNormalization()(lstm)
    
    # CNN-LSTM connection for 120-min predictions
    cnn_lstm = LSTM(
        32, 
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(pool)
    cnn_lstm = BatchNormalization()(cnn_lstm)
    
    # Flatten the CNN output
    flat = Flatten()(pool)
    
    # Combine branches (CNN, LSTM, and CNN-LSTM)
    concat = Concatenate()([flat, lstm, cnn_lstm])
    dense_120 = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(concat)
    dense_120 = BatchNormalization()(dense_120)
    dense_120 = Dropout(dropout_rate)(dense_120)
    
    dense_120 = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(dense_120)
    dense_120 = BatchNormalization()(dense_120)
    output_120 = Dense(1, name='output_120min')(dense_120)
    
    # Create the model with multiple outputs
    model = Model(inputs=input_layer, outputs=[output_60, output_120])
    
    # Compile with appropriate loss and metrics
    model.compile(
        optimizer='adam',
        loss={'output_60min': 'mse', 'output_120min': 'mse'},
        metrics={'output_60min': 'mae', 'output_120min': 'mae'}
    )
    
    logger.info("Model built successfully")
    return model


def train_and_evaluate_multi_horizon_model(
    X: np.ndarray, 
    y_60: np.ndarray, 
    y_120: np.ndarray, 
    sequence_length: int, 
    n_features: int, 
    scaler_y,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    patience: int = 10,
    model_save_path: Optional[str] = None,
    **model_kwargs
) -> Tuple[Model, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Train and evaluate a multi-horizon glucose prediction model.
    
    Parameters:
    -----------
    X : np.ndarray
        Input sequences
    y_60 : np.ndarray
        60-minute targets
    y_120 : np.ndarray
        120-minute targets
    sequence_length : int
        Length of sequences
    n_features : int
        Number of features
    scaler_y : sklearn scaler
        Scaler for glucose values
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Training batch size
    validation_split : float
        Fraction of data to use for validation
    patience : int
        Early stopping patience
    model_save_path : str, optional
        Path to save the best model
    **model_kwargs
        Additional arguments for model building
        
    Returns:
    --------
    Tuple containing trained model, history, evaluation data, and results
    """
    logger.info("Starting multi-horizon model training")
    
    # Split the data
    X_train, X_test, y_train_60, y_test_60, y_train_120, y_test_120 = train_test_split(
        X, y_60, y_120, test_size=validation_split, random_state=42, shuffle=False
    )
    
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Build the model
    model = build_multi_horizon_model(sequence_length, n_features, **model_kwargs)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            verbose=1, 
            restore_best_weights=True
        )
    ]
    
    if model_save_path:
        callbacks.append(
            ModelCheckpoint(
                model_save_path, 
                monitor='val_loss', 
                save_best_only=True,
                verbose=1
            )
        )
    
    # Prepare training targets
    y_train_dict = {
        'output_60min': y_train_60,
        'output_120min': y_train_120
    }
    
    # Prepare validation targets
    y_val_dict = {
        'output_60min': y_test_60,
        'output_120min': y_test_120
    }
    
    # Train the model
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train_dict,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_val_dict),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Extract predictions
    y_pred_60 = y_pred[0]  # First output is 60min prediction
    y_pred_120 = y_pred[1]  # Second output is 120min prediction
    
    # Calculate metrics for 60min predictions
    mse_60 = mean_squared_error(y_test_60, y_pred_60)
    mae_60 = mean_absolute_error(y_test_60, y_pred_60)
    r2_60 = r2_score(y_test_60, y_pred_60)
    
    # Calculate metrics for 120min predictions
    mse_120 = mean_squared_error(y_test_120, y_pred_120)
    mae_120 = mean_absolute_error(y_test_120, y_pred_120)
    r2_120 = r2_score(y_test_120, y_pred_120)
    
    # Print metrics
    logger.info("Model evaluation completed")
    print("\\nMulti-Horizon Model Metrics:")
    print("60-minute prediction:")
    print(f"Mean Squared Error: {mse_60:.4f}")
    print(f"Mean Absolute Error: {mae_60:.4f}")
    print(f"R² Score: {r2_60:.4f}")
    
    print("\\n120-minute prediction:")
    print(f"Mean Squared Error: {mse_120:.4f}")
    print(f"Mean Absolute Error: {mae_120:.4f}")
    print(f"R² Score: {r2_120:.4f}")
    
    # Prepare evaluation data dictionary
    eval_data = {
        'X_test': X_test,
        'y_test_60': y_test_60,
        'y_test_120': y_test_120,
        'y_pred_60': y_pred_60,
        'y_pred_120': y_pred_120
    }
    
    results = {
        '60min': {
            'metrics': {'mse': mse_60, 'mae': mae_60, 'r2': r2_60},
            'predictions': y_pred_60,
        },
        '120min': {
            'metrics': {'mse': mse_120, 'mae': mae_120, 'r2': r2_120},
            'predictions': y_pred_120,
        }
    }
    
    return model, history, eval_data, results


def convert_predictions_to_original_scale(y_pred: np.ndarray, scaler_y) -> np.ndarray:
    """
    Convert scaled predictions back to the original glucose scale.
    
    Parameters:
    -----------
    y_pred : np.ndarray
        Scaled predictions
    scaler_y : sklearn scaler
        Fitted scaler for glucose values
        
    Returns:
    --------
    np.ndarray
        Predictions in original scale
    """
    # Reshape if needed
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Inverse transform
    y_pred_original = scaler_y.inverse_transform(y_pred)
    
    return y_pred_original