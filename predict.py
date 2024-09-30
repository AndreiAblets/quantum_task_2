"""
predict.py

Script for making predictions on the test dataset using the trained model.

Usage:
    python predict.py

Requirements:
    - hidden_test.csv: The test dataset.
    - regression_model.h5: The trained model.
    - scaler.pkl: The scaler used during training.
    - Python 3.6 or higher.
    - Install dependencies from requirements.txt.
"""

import pandas as pd
import numpy as np
import argparse
import joblib

from tensorflow.keras.models import load_model

def main():
    # Load the test dataset
    test_data = pd.read_csv('hidden_test.csv')
    print("Test dataset loaded successfully.")

    # Drop redundant feature
    test_data.drop(columns=['8'], inplace=True)
    print("Feature 8 dropped from the test dataset.")

    # Load the scaler
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded.")

    # Scale features
    X_test = scaler.transform(test_data)
    print("Test features scaled.")

    # Load the trained model
    model = load_model('regression_model.h5')
    print("Trained model loaded.")

    # Make predictions
    predictions = model.predict(X_test)
    print("Predictions made.")

    # Save predictions to a CSV file
    output = pd.DataFrame({'Prediction': predictions.flatten()})
    output.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'.")

if __name__ == '__main__':
    main()
