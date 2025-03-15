import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging to both console and file
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the folder where this script is saved
log_file = os.path.join(script_dir, "outlier_handling.log")  # Log file in the script's folder
logging.basicConfig(
    level=logging.INFO,  # Show all important messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format with time, level, and message
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]  # Save to file and show on screen
)
logger = logging.getLogger(__name__)

# Load the preprocessed data from Task 3
input_file = os.path.join(script_dir, "preprocessed_data.csv")  # Input file in the script's folder
try:
    df = pd.read_csv(input_file)  # Read the CSV into a DataFrame
    logger.info(f"Successfully loaded preprocessed data from: {input_file}")
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')  # Ensure date is in datetime format
except Exception as e:
    logger.error(f"Error loading preprocessed data: {e}")
    raise

# Function to detect and handle outliers
def detect_and_handle_outliers(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns  # Get all number columns
    original_df = df.copy()  # Keep a copy of the original data for comparison

    # Validate ranges for specific columns
    if 'hour' in df.columns:
        df['hour'] = df['hour'].clip(0, 23)  # Limit hour to 0-23
        logger.info("Validated 'hour' to range 0-23.")
    if 'day' in df.columns:
        df['day'] = df['day'].clip(1, 31)  # Limit day to 1-31
        logger.info("Validated 'day' to range 1-31.")
    if 'month' in df.columns:
        df['month'] = df['month'].clip(1, 12)  # Limit month to 1-12
        logger.info("Validated 'month' to range 1-12.")
    if 'year' in df.columns:
        df['year'] = df['year'].clip(2020, 2025)  # Limit year to 2020-2025 (adjust as needed)
        logger.info("Validated 'year' to range 2020-2025.")

    # --- IQR-based Detection ---
    logger.info("Performing IQR-based outlier detection.")
    iqr_outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR  # Lower limit for outliers
        upper_bound = Q3 + 1.5 * IQR  # Upper limit for outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        iqr_outliers[col] = outliers.tolist()  # Store outliers
        if not outliers.empty:
            logger.warning(f"IQR Outliers in {col}: {outliers.tolist()}")
        else:
            logger.info(f"No IQR outliers detected in {col}.")

    # --- Z-score Method ---
    logger.info("Performing Z-score outlier detection.")
    zscore_outliers = {}
    for col in numerical_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())  # Calculate Z-scores
        outliers = df[z_scores > 3][col]  # Flag Z-scores > 3
        zscore_outliers[col] = outliers.tolist()  # Store outliers
        if not outliers.empty:
            logger.warning(f"Z-score Outliers in {col}: {outliers.tolist()}")
        else:
            logger.info(f"No Z-score outliers detected in {col}.")

    # --- Evaluate Impact of Outliers ---
    logger.info("Evaluating impact of outliers.")
    impact_summary = {}
    for col in numerical_cols:
        original_mean = original_df[col].mean()
        original_std = original_df[col].std()
        filtered_df = original_df[(original_df[col] >= (Q1 - 1.5 * IQR)) & (original_df[col] <= (Q3 + 1.5 * IQR))]
        filtered_mean = filtered_df[col].mean()
        filtered_std = filtered_df[col].std()
        impact_summary[col] = {
            'original_mean': original_mean,
            'original_std': original_std,
            'filtered_mean': filtered_mean,
            'filtered_std': filtered_std,
            'mean_change': abs(original_mean - filtered_mean),
            'std_change': abs(original_std - filtered_std)
        }
        logger.info(f"{col} Impact: Mean change={impact_summary[col]['mean_change']:.2f}, "
                    f"Std change={impact_summary[col]['std_change']:.2f}")

    # --- Handling Strategy: Cap Outliers ---
    logger.info("Applying capping strategy for outliers.")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)  # Cap outliers to bounds
        logger.info(f"Capped outliers in {col} between {lower_bound:.2f} and {upper_bound:.2f}")

    # --- Before-and-After Visualizations ---
    logger.info("Creating before-and-after visualizations.")
    for col in ['temperature_2m', 'hour', 'day']:  # Example key columns for visualization
        if col in df.columns:
            plt.figure(figsize=(12, 5))
            # Before
            plt.subplot(1, 2, 1)
            sns.boxplot(y=original_df[col].dropna())
            plt.title(f'Before: Boxplot of {col}')
            # After
            plt.subplot(1, 2, 2)
            sns.boxplot(y=df[col].dropna())
            plt.title(f'After: Boxplot of {col}')
            plt.tight_layout()
            viz_plot = os.path.join(script_dir, f"outlier_viz_{col}.png")  # Save in script's folder
            plt.savefig(viz_plot)  # Save the plot
            plt.close()
            logger.info(f"Outlier visualization for {col} saved to: {viz_plot}")

    return df, iqr_outliers, zscore_outliers, impact_summary

# Run the outlier detection and handling
if __name__ == "__main__":
    if 'df' in locals() and not df.empty:
        processed_df, iqr_outliers, zscore_outliers, impact_summary = detect_and_handle_outliers(df)
        if processed_df is not None:
            print("Outlier detection and handling completed successfully.")
            print("IQR Outliers:", iqr_outliers)
            print("Z-score Outliers:", zscore_outliers)
            print("Impact Summary:", impact_summary)
            # Save the modified dataset in the script's folder
            output_file = os.path.join(script_dir, "outlier_processed_data.csv")
            processed_df.to_csv(output_file, index=False)
            logger.info(f"Modified dataset saved to: {output_file}")
            print(f"Modified dataset saved to: {output_file}")
        else:
            print("Outlier detection and handling failed.")
    else:
        logger.error("No valid data to perform outlier detection and handling.")
        print("Outlier detection and handling failed due to no valid data.")