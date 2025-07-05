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
