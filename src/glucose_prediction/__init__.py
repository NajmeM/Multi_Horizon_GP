"""
Multi-Horizon Glucose Prediction Package

A deep learning framework for predicting glucose levels at multiple time horizons
using LSTM and CNN-LSTM hybrid models.
"""

__version__ = "0.1.1"
__author__ = "Najmeh Mohajeri"
__email__ = "nmohajeri@gmail.com"

from .data.preprocessing import load_and_preprocess_data
from .data.sequence_generator import create_sequences
from .models.multi_horizon_model import build_multi_horizon_model, train_and_evaluate_multi_horizon_model
from .evaluation.clarke_error_grid import clarke_error_grid, evaluate_glucose_predictions_with_clarke
from .visualization.plotting import plot_training_history, plot_scaled_predictions, plot_original_scale_predictions
from .utils.config import ModelConfig, DataConfig

__all__ = [
    "load_and_preprocess_data",
    "create_sequences",
    "build_multi_horizon_model",
    "train_and_evaluate_multi_horizon_model",
    "clarke_error_grid",
    "evaluate_glucose_predictions_with_clarke",
    "plot_training_history",
    "plot_scaled_predictions",
    "plot_original_scale_predictions",
    "ModelConfig",
    "DataConfig",
]