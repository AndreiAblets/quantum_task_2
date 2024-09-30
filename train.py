

"""
train.py

Script for training a regression model on the provided dataset.

Usage:
    python train.py

Requirements:
    - train.csv: The training dataset.
    - Python 3.6 or higher.
    - Install dependencies from requirements.txt.
"""

import pandas as pd
import numpy as np
import argparse
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def main():
    # Load the dataset
    data = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")

    # Drop redundant feature
    data.drop(columns=['8'], inplace=True)
    print("Feature 8 dropped from the dataset.")

    # Handle missing values
    data.dropna(inplace=True)
    print("Missing values handled.")

    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled.")

    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'.")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print("Data split into training and validation sets.")

    # Build the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    print("Model compiled.")

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    print("Model training completed.")

    # Evaluate the model
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse:.4f}")

    # Save the model
    model.save('regression_model.h5')
    print("Model saved as 'regression_model.h5'.")

if __name__ == '__main__':
    main()
