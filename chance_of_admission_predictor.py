import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------
# 1. DATA LOADING AND PREPARATION
# ----------------------------------------------------------------------

print("--- 1. Data Loading and Preparation ---")

try:
    # Load the data from the separate CSV file
    df = pd.read_csv('admission_data.csv')
except FileNotFoundError:
    print("Error: 'admission_data.csv' not found. Please ensure the dataset file is present and correctly named.")
    # Create an empty DataFrame as a fallback
    df = pd.DataFrame() 
    
if df.empty:
    print("Exiting script due to data loading failure.")
else:
    # Cleans column names by stripping leading/trailing whitespace
    df.columns = df.columns.str.strip()

    # Drops the 'Serial No.' column as it's an irrelevant identifier
    df = df.drop(columns=['Serial No.'])

    print("\nSample Data Head:")
    print(df.head())
    print("-" * 50)

    # ----------------------------------------------------------------------
    # 2. FEATURE ENGINEERING AND SPLITTING
    # ----------------------------------------------------------------------

    # Define Features (X) and Target (y)
    # The target is 'Chance of Admit'
    y = df['Chance of Admit']
    X = df.drop('Chance of Admit', axis=1)

    print("--- 2. Feature and Target Split ---")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print("-" * 50)

    # Splits the data into training and testing sets
    # We use test_size=0.2 (20% for testing) and random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("--- 3. Train/Test Split ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("-" * 50)

    # ----------------------------------------------------------------------
    # 4. MODEL TRAINING (Linear Regression)
    # ----------------------------------------------------------------------

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Trains the model using the training data
    print("--- 4. Training Linear Regression Model ---")
    model.fit(X_train, y_train)
    print("Model training complete.")
    print("-" * 50)

    # Displays model coefficients and intercept
    print("Model Coefficients (Feature Weights):")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print("-" * 50)


    # ----------------------------------------------------------------------
    # 5. MODEL EVALUATION
    # ----------------------------------------------------------------------

    # Makes predictions on the test set
    y_pred = model.predict(X_test)

    print("--- 5. Model Evaluation ---")

    # Calculates evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R-squared Score (R2): {r2:.4f} (Closer to 1 is better)")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} (Measure of typical error)")
    print("-" * 50)


    # ----------------------------------------------------------------------
    # 6. MAKING NEW PREDICTIONS
    # ----------------------------------------------------------------------

    # Defines a new applicant's profile for prediction (single row of data)
    new_applicant_data = pd.DataFrame({
        'GRE Score': [325],
        'TOEFL Score': [115],
        'University Rating': [4],
        'SOP': [4.0],
        'LOR': [4.5],
        'CGPA': [9.00],
        'Research': [1]
    })

    # Ensures the new data columns match the training data columns exactly
    new_applicant_data = new_applicant_data[X.columns]

    # Predicts the chance of admission
    predicted_chance = model.predict(new_applicant_data)[0]

    print("--- 6. New Prediction Example ---")
    print(f"New Applicant Profile:\n{new_applicant_data.to_string(index=False)}")

    # Rounds the prediction and ensure it's between 0 and 1
    final_chance = max(0, min(1, predicted_chance))

    print(f"\nPredicted Chance of Admission: {final_chance:.4f} (or {final_chance*100:.2f}%)")
    print("-" * 50)
