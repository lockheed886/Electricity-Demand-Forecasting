import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Set up logging to both console and file
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the folder where this script is saved
log_file = os.path.join(script_dir, "regression_modeling.log")  # Log file in the script's folder
logging.basicConfig(
    level=logging.INFO,  # Show all important messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format with time, level, and message
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]  # Save to file and show on screen
)
logger = logging.getLogger(__name__)

# Load the processed data from Task 4
input_file = os.path.join(script_dir, "outlier_processed_data.csv")  # Path to the processed file
try:
    df = pd.read_csv(input_file)  # Read the CSV into a DataFrame
    logger.info(f"Successfully loaded processed data from: {input_file}")
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')  # Ensure date is in datetime format
except Exception as e:
    logger.error(f"Error loading processed data: {e}")
    raise

# Function to build and evaluate regression model
def build_and_evaluate_regression(df):
    # --- Feature Selection ---
    logger.info("Selecting relevant features for regression.")
    # Assuming 'temperature_2m' is a proxy for electricity demand as the target
    target_column = 'temperature_2m'  # Target variable (to predict)
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the dataset.")
        return None

    # Select predictors: time-based features and other relevant columns
    feature_columns = ['hour', 'day', 'month', 'year', 'is_weekend']  # Chosen predictors
    for col in feature_columns:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' not found, skipping.")
            feature_columns.remove(col)
    if not feature_columns:
        logger.error("No valid features selected for regression.")
        return None

    X = df[feature_columns]  # Predictor variables
    y = df[target_column]   # Target variable

    # --- Data Splitting ---
    logger.info("Splitting data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    # --- Model Development ---
    logger.info("Building linear regression model.")
    model = LinearRegression()  # Create a linear regression model
    model.fit(X_train, y_train)  # Train the model
    logger.info("Model training completed.")

    # --- Model Evaluation ---
    logger.info("Evaluating the model.")
    y_pred = model.predict(X_test)  # Predict on test set
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # R-squared score
    evaluation_metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }
    logger.info(f"Evaluation Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
    print("Evaluation Metrics:", evaluation_metrics)

    # Save metrics to a CSV file
    metrics_file = os.path.join(script_dir, "regression_metrics.csv")
    pd.DataFrame([evaluation_metrics]).to_csv(metrics_file, index=False)
    logger.info(f"Evaluation metrics saved to: {metrics_file}")

    # --- Plot Actual vs. Predicted Values ---
    logger.info("Plotting actual vs. predicted values.")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)  # Scatter plot of actual vs. predicted
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Ideal line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Electricity Demand')
    actual_vs_pred_plot = os.path.join(script_dir, "actual_vs_predicted.png")
    plt.savefig(actual_vs_pred_plot)  # Save the plot
    plt.close()
    logger.info(f"Actual vs. Predicted plot saved to: {actual_vs_pred_plot}")

    # --- Residual Analysis ---
    logger.info("Performing residual analysis.")
    residuals = y_test - y_pred  # Calculate residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)  # Histogram of residuals
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    residuals_plot = os.path.join(script_dir, "residuals_distribution.png")
    plt.savefig(residuals_plot)  # Save the plot
    plt.close()
    logger.info(f"Residuals distribution plot saved to: {residuals_plot}")

    # Comment on performance
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    logger.info(f"Residual Analysis: Mean={residual_mean:.2f}, Std={residual_std:.2f}")
    if abs(residual_mean) < 0.1 and r2 > 0.5:
        logger.info("Model performs well: residuals are centered near zero, and R2 is reasonable.")
    else:
        logger.warning("Model may need improvement: residuals are not centered or R2 is low.")

    # Save the trained model
    model_file = os.path.join(script_dir, "regression_model.pkl")
    import joblib
    joblib.dump(model, model_file)  # Save the model
    logger.info(f"Trained model saved to: {model_file}")

    return model, evaluation_metrics

# Run the regression modeling
if __name__ == "__main__":
    if 'df' in locals() and not df.empty:
        model, evaluation_metrics = build_and_evaluate_regression(df)
        if model is not None:
            print("Regression modeling completed successfully.")
            print("Evaluation Metrics:", evaluation_metrics)
            logger.info("Regression modeling completed successfully.")
        else:
            print("Regression modeling failed.")
    else:
        logger.error("No valid data to perform regression modeling.")
        print("Regression modeling failed due to no valid data.")