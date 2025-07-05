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


<<<<<<< HEAD
### Option 1: Quick Install
```bash
git clone https://github.com/yourusername/multi-horizon-glucose-prediction.git
cd multi-horizon-glucose-prediction
pip install -r requirements.txt
pip install -e .
=======
## 📁 Project Structure
multi-horizon-glucose-prediction/
├── src/glucose_prediction/     # Main package
│   ├── data/                   # Data processing modules
│   ├── models/                 # Model implementations
│   ├── evaluation/             # Evaluation metrics
│   ├── visualization/          # Plotting utilities
│   └── utils/                  # Configuration and helpers
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── examples/                   # Example scripts
├── config/                     # Configuration files
├── docs/                       # Documentation
└── main.py                     # Main CLI script

### 📚 Documentation

Installation Guide
Usage Guide
Methodology

### 🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Make your changes
Add tests for new functionality
Run the test suite (pytest)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### 📬 Contact

Author: Najmeh Mohajeri
Email: nmohajeri@gmail.com
GitHub: @NajmehM
>>>>>>> 853f524 (add doc files)
