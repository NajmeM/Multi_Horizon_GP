### 🚀 Quick Start
Command Line Usage
```bash
# Basic usage
python main.py --data path/to/your/glucose_data.csv

# With custom configuration
python main.py --data path/to/your/glucose_data.csv --config config/custom_config.yaml --output results/

# Batch processing
python examples/batch_processing.py --input-dir data/patients/ --output-dir batch_results/
```
### Python API Usage
```python
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
```
### 📊 Data Format
Your CSV file should contain the following columns (semicolon-separated):
```csv
time;glucose;basal_rate;bolus_volume_delivered;calories;steps;carb_input
2023-01-01 00:00:00;120.5;1.2;0.0;0;0;0
2023-01-01 00:05:00;118.3;1.2;0.0;0;0;0
...
```
Required columns:

time: Timestamp in datetime format
glucose: Glucose level (mg/dL)
basal_rate: Basal insulin rate
bolus_volume_delivered: Bolus insulin volume
calories: Caloric intake
steps: Step count
carb_input: Carbohydrate intake

### 🔧 Configuration
Configuration is managed through YAML files. Example configuration:
```yaml
model:
  sequence_length: 12        # 1 hour of 5-minute intervals
  prediction_horizon_60: 12  # 60 minutes ahead
  prediction_horizon_120: 24 # 120 minutes ahead
  lstm_units_60: 64
  lstm_units_120: 64
  dropout_rate: 0.3
  epochs: 100
  batch_size: 32

data:
  features:
    - insulin
    - calories
    - steps
    - carb_input
    - glucose
  test_size: 0.2
  random_state: 42

evaluation:
  save_plots: true
  clarke_analysis: true
  generate_reports: true
```