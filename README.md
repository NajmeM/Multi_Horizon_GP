# Multi-Horizon Glucose Prediction

A comprehensive deep learning framework for predicting glucose levels at multiple time horizons (60 and 120 minutes) using LSTM and CNN-LSTM hybrid models. This project is designed for diabetes management and research applications.

## 🚀 Features

- **Multi-horizon prediction**: Simultaneous prediction of glucose levels at 60 and 120 minutes
- **Hybrid architecture**: Combines LSTM and CNN layers for temporal and spatial feature extraction
- **Clinical evaluation**: Includes Clarke Error Grid analysis for clinical significance assessment
- **Production-ready**: Modular design with comprehensive testing and documentation
- **Batch processing**: Support for processing multiple patient datasets
- **Visualization tools**: Rich plotting capabilities for analysis and presentation

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.10+
- scikit-learn 1.1+
- pandas 1.5+
- matplotlib 3.5+
- See `requirements.txt` for complete list

## 🛠️ Installation

### Option 1: Quick Install
```bash
git clone https://github.com/yourusername/multi-horizon-glucose-prediction.git
cd multi-horizon-glucose-prediction
pip install -r requirements.txt
pip install -e .

### Option 2: Development Install
```bash 
git clone https://github.com/yourusername/multi-horizon-glucose-prediction.git
cd multi-horizon-glucose-prediction
pip install -r requirements.txt
pip install -e ".[dev,notebook]"

### Option 3: Conda Environment
git clone https://github.com/yourusername/multi-horizon-glucose-prediction.git
cd multi-horizon-glucose-prediction
conda env create -f environment.yml
conda activate glucose-prediction
pip install -e .

🚀 Quick Start
### Command Line Usage
# Basic usage
python main.py --data path/to/your/glucose_data.csv

# With custom configuration
python main.py --data path/to/your/glucose_data.csv --config config/custom_config.yaml --output results/

# Batch processing
python examples/batch_processing.py --input-dir data/patients/ --output-dir batch_results/


### Python API Usage
from glucose_prediction import (
    load_and_preprocess_data, create_sequences, 
    train_and_evaluate_multi_horizon_model,
    evaluate_glucose_predictions_with_clarke
)

# Load and preprocess data
df, df_scaled, scaler_x, scaler_y = load_and_preprocess_data("data.csv")

# Create sequences
X, y_60, y_120 = create_sequences(df_scaled)

# Train model
model, history, eval_data, results = train_and_evaluate_multi_horizon_model(
    X, y_60, y_120, sequence_length=12, n_features=5, scaler_y=scaler_y
)

# Clinical evaluation
clarke_results = evaluate_glucose_predictions_with_clarke(eval_data, scaler_y)