📈 Model Architecture
The model uses a hybrid architecture combining:

LSTM Branch (60-min): Specialized for short-term predictions
CNN-LSTM Branch (120-min): Combines convolutional and recurrent layers for long-term patterns
Multi-output: Simultaneous prediction of both horizons
Regularization: L2 regularization, dropout, and batch normalization

🎯 Evaluation Metrics
Statistical Metrics

R² Score: Coefficient of determination
MSE: Mean Squared Error
MAE: Mean Absolute Error

Clinical Metrics (Clarke Error Grid)

Zone A: Clinically accurate (target: >75%)
Zone B: Benign errors (acceptable)
Zones C, D, E: Clinical errors (should be minimized)
Clinical Acceptability: Percentage in Zones A+B

