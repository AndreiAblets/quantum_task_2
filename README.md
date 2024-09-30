# Regression on Tabular Data

## Project Overview

This project involves building a regression model to predict a target variable based on 53 anonymized features. The main goal is to create a model that minimizes the RMSE on the provided test dataset.

## Repository Contents

- `EDA.ipynb`: Jupyter notebook containing exploratory data analysis.
- `train.py`: Script for training the regression model.
- `predict.py`: Script for generating predictions on the test dataset.
- `predictions.csv`: File containing the prediction results.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

## Instructions

### Setup

#### Prerequisites

- Python 3.6 or higher.
- Virtual environment (recommended).

### Installation and usage

1. **Clone the repository**
 git clone https://github.com/yourusername/your-repo-name.git
 cd your-repo-name
   
2. **Install the required packages**
 pip install -r requirements.txt
   
3. **Training the Model**
Place train.csv in the project directory.
Run the training script:
python train.py

This will:

Load and preprocess the data.
Train the neural network model.
Save the trained model as regression_model.h5.
Save the scaler as scaler.pkl.

4. **Making Predictions**
Place hidden_test.csv in the project directory.
Run the prediction script:
python predict.py

This will:

Load and preprocess the test data.
Load the trained model and scaler.
Generate predictions.
Save the predictions to predictions.csv.

5. **Files**
train.py: Script for training the model.
predict.py: Script for making predictions.
regression_model.h5: Trained Keras model (generated after running train.py).
scaler.pkl: Scaler object used for feature scaling (generated after running train.py).
predictions.csv: CSV file containing the predictions (generated after running predict.py).

6. **Notes**
Ensure that train.csv and hidden_test.csv are placed in the project directory before running the scripts.
The scripts are designed to be executed from the terminal.
The model architecture and hyperparameters can be adjusted in train.py for experimentation.

7. **Given more time, the following enhancements could be made**

Extensive Hyperparameter Tuning:
Utilize Grid Search, Random Search, or Bayesian Optimization for hyperparameter optimization.
Explore different network architectures, learning rates, and activation functions.

Model Comparison:
Experiment with other regression algorithms like Random Forest, XGBoost, and LightGBM.
Implement ensemble methods to combine predictions from multiple models.

Cross-Validation:
Implement k-fold cross-validation for a more robust evaluation.
Use techniques like Stratified K-Fold if the target variable distribution requires it.

Model Interpretability:
Utilize SHAP values or LIME to understand feature importance and model decisions.

Automated Machine Learning (AutoML):
Leverage AutoML tools to automate model selection and hyperparameter tuning.
