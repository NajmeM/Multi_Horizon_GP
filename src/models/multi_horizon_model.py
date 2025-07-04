from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, Concatenate
from tensorflow.keras.regularizers import l2

def build_multi_horizon_model(sequence_length: int, n_features: int):
    """Build a model that predicts both 60min and 120min horizons."""
        # Shared input layer
    input_layer = Input(shape=(sequence_length, n_features))
    
    # LSTM branch for 60-minute predictions with regularization
    lstm_60 = LSTM(units=64, return_sequences=True, 
                   kernel_regularizer=l2(0.001),  # L2 regularization
                   recurrent_regularizer=l2(0.001))(input_layer)
    lstm_60 = BatchNormalization()(lstm_60)  # Add batch normalization
    lstm_60 = Dropout(0.3)(lstm_60)  # Increased dropout rate
    lstm_60 = LSTM(units=32, 
                   kernel_regularizer=l2(0.001), 
                   recurrent_regularizer=l2(0.001))(lstm_60)
    lstm_60 = BatchNormalization()(lstm_60)
    lstm_60 = Dropout(0.3)(lstm_60)  
    dense_60 = Dense(64, activation='relu', 
                     kernel_regularizer=l2(0.001))(lstm_60)  # L2 regularization
    output_60 = Dense(1, name='output_60min')(dense_60)
    
    
    # CNN part
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu',
                   kernel_regularizer=l2(0.001))(input_layer)  # L2 regularization
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu',
                   kernel_regularizer=l2(0.001))(conv1)
    conv2 = BatchNormalization()(conv2)
    pool = MaxPooling1D(pool_size=2)(conv2)
    
    lstm = LSTM(64, return_sequences=True,
               kernel_regularizer=l2(0.001),
               recurrent_regularizer=l2(0.001))(input_layer)
    lstm = BatchNormalization()(lstm)
    lstm = LSTM(32,
               kernel_regularizer=l2(0.001),
               recurrent_regularizer=l2(0.001))(lstm)
    lstm = BatchNormalization()(lstm)
    
    # CNN-LSTM connection 
    cnn_lstm = LSTM(32, 
                   kernel_regularizer=l2(0.001),
                   recurrent_regularizer=l2(0.001))(pool)
    cnn_lstm = BatchNormalization()(cnn_lstm)
    
    # Flatten the CNN output
    flat = Flatten()(pool)
    
    # Combine branches (CNN, LSTM, and CNN-LSTM)
    concat = Concatenate()([flat, lstm, cnn_lstm])
    dense_120 = Dense(64, activation='relu',
                      kernel_regularizer=l2(0.001))(concat)  # L2 regularization
    dense_120 = BatchNormalization()(dense_120)
    dense_120 = Dropout(0.3)(dense_120)  # Increased dropout rate
    
    dense_120 = Dense(32, activation='relu',
                      kernel_regularizer=l2(0.001))(dense_120)
    dense_120 = BatchNormalization()(dense_120)
    output_120 = Dense(1, name='output_120min')(dense_120)
    
    # Create the model with multiple outputs
    model = Model(inputs=input_layer, outputs=[output_60, output_120])
    
    # Compile with different loss weights and added regularization in the optimizer
    model.compile(
        optimizer='adam',
        loss={'output_60min': 'mse', 'output_120min': 'mse'},
        metrics={'output_60min': 'mae', 'output_120min': 'mae'}
    )
    
    return model

def train_and_evaluate_multi_horizon_model(X, y_60, y_120, sequence_length: int, 
                                         n_features: int, scaler_y):
    """Train and evaluate a multi-horizon glucose prediction model."""
    X_train, X_test, y_train_60, y_test_60, y_train_120, y_test_120 = train_test_split(
        X, y_60, y_120, test_size=0.2, random_state=42)
    
    # Build the model
    model = build_multi_horizon_model(sequence_length, n_features)
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('glucose_prediction_multi_horizon.h5', monitor='val_loss', save_best_only=True)
    
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
    history = model.fit(
        X_train, y_train_dict,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_val_dict),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
     # Add this debug print to understand the shapes
    print(f"Shapes - X_test: {X_test.shape}, y_test_60: {y_test_60.shape}")
    print(f"Shapes - y_pred: {len(y_pred)}, y_pred[0]: {y_pred[0].shape}, y_pred[1]: {y_pred[1].shape}")
    
    # Then adjust the access based on the actual shape
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
    print("\nMulti-Horizon Model Metrics:")
    print("60-minute prediction:")
    print(f"Mean Squared Error: {mse_60:.4f}")
    print(f"Mean Absolute Error: {mae_60:.4f}")
    print(f"R² Score: {r2_60:.4f}")
    
    print("\n120-minute prediction:")
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

    # Return model, history, evaluation data, and results
    return model, history, eval_data, results
