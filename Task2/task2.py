import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Set up logging to show messages on the screen and save them to a file
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the folder where this script is saved
log_file = os.path.join(script_dir, "preprocessing.log")  # Create a log file name in the same folder
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to show all important messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of the log messages (time, level, message)
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]  # Save logs to file and show on screen
)
logger = logging.getLogger(__name__)  # Create a logger to use throughout the script

# Load the combined data file created in Task 1
input_file = os.path.join(script_dir, "combined_data.csv")  # Path to the input file in the script's folder
try:
    df = pd.read_csv(input_file)  # Read the CSV file into a DataFrame
    logger.info(f"Successfully loaded combined data from: {input_file}")  # Log success message
except Exception as e:
    logger.error(f"Error loading combined data: {e}")  # Log error if loading fails
    raise  # Stop the script if there's an error

# Function to clean and prepare the data
def preprocess_data(df):
    # a) Check for missing values and handle them
    missing_percent = df.isnull().mean() * 100  # Calculate the percentage of missing values in each column
    logger.info("Percentage of missing values per column:\n%s", missing_percent)  # Log the percentages
    print("Percentage of missing values per column:\n", missing_percent)  # Show percentages on screen

    # Decide how to handle missing data
    # We assume data is missing randomly (MCAR) since we don’t know more about it yet
    if missing_percent.any():  # If there are any missing values
        # Remove columns that are mostly empty (more than 90% missing)
        high_missing_cols = missing_percent[missing_percent > 90].index  # Find columns with >90% missing
        if not high_missing_cols.empty:  # If there are such columns
            logger.warning(f"Dropping columns with >90% missingness: {high_missing_cols.tolist()}")  # Warn about dropping
            df.drop(columns=high_missing_cols, inplace=True)  # Remove those columns
            missing_percent = df.isnull().mean() * 100  # Update the missing percentages

        # Fill in missing values for important columns (date and temperature)
        important_cols = ['date', 'temperature_2m']  # List of key columns we want to keep
        for column in important_cols:
            if column in df.columns:  # Check if the column exists
                if missing_percent[column] > 0:  # If there are missing values
                    if column == 'date':  # Special handling for date
                        df[column] = pd.to_datetime(df[column], errors='coerce')  # Convert to datetime, ignore errors
                        df[column].interpolate(method='linear', inplace=True, limit_direction='both')  # Fill gaps with a straight line
                    else:  # For temperature_2m
                        df[column].interpolate(method='linear', inplace=True, limit_direction='both')  # Fill gaps with a straight line
                    logger.info(f"Imputed missing values in {column} using linear interpolation")  # Log the action
    else:
        logger.info("No missing values detected.")  # Log if no missing values found

    # Check if all data was lost after handling missing values
    if df.empty:  # If the DataFrame has no rows left
        logger.error("DataFrame is empty after missing data handling. No data to process.")  # Log the error
        return None  # Stop processing and return nothing

    # b) Change data types to the right format
    if 'date' in df.columns:  # Check if there’s a date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert date to a proper date-time format
        # Create new columns from the date
        df['hour'] = df['date'].dt.hour  # Extract the hour
        df['day'] = df['date'].dt.day  # Extract the day
        df['month'] = df['date'].dt.month  # Extract the month
        df['year'] = df['date'].dt.year  # Extract the year
        # Add a season column based on the month
        df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                        'Spring' if x in [3, 4, 5] else
                                        'Summer' if x in [6, 7, 8] else
                                        'Fall')  # Assign season
        logger.info("Converted 'date' to datetime and extracted temporal features.")  # Log the changes
    else:
        logger.warning("No 'date' column found for temporal feature extraction.")  # Warn if no date column

    # Make sure numbers are treated as numbers
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns  # Find all number columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numbers, ignore errors

    # Change text columns to a special category type for better use
    categorical_cols = df.select_dtypes(include=['object']).columns  # Find all text columns
    for col in categorical_cols:
        if col not in ['date']:  # Skip date since it’s already a date
            df[col] = pd.Categorical(df[col])  # Convert to category type
        logger.info(f"Converted {col} to categorical type.")  # Log the conversion

    # c) Fix duplicates and odd data
    initial_rows = len(df)  # Count the rows before checking
    df.drop_duplicates(inplace=True)  # Remove any duplicate rows
    if len(df) < initial_rows:  # If rows were removed
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows.")  # Log how many were removed
    else:
        logger.info("No duplicate rows found.")  # Log if no duplicates

    # Find and fix outliers (unusual numbers) in number columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)  # Find the 25th percentile
        Q3 = df[col].quantile(0.75)  # Find the 75th percentile
        IQR = Q3 - Q1  # Calculate the range between them
        lower_bound = Q1 - 1.5 * IQR  # Set the lower limit for outliers
        upper_bound = Q3 + 1.5 * IQR  # Set the upper limit for outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]  # Find outliers
        if not outliers.empty:  # If there are outliers
            logger.warning(f"Outliers detected in {col}: {outliers.tolist()}")  # Warn about them
            df[col] = df[col].clip(lower_bound, upper_bound)  # Limit outliers to the bounds
            logger.info(f"Capped outliers in {col} using IQR method.")  # Log the fix
        else:
            logger.info(f"No outliers detected in {col}.")  # Log if no outliers

    # d) Create new features from existing data
    if 'date' in df.columns:  # Check if date is available
        df['day_of_week'] = df['date'].dt.day_name()  # Get the day of the week (e.g., Monday)
        df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)  # Mark weekends as 1, weekdays as 0
        logger.info("Derived 'day_of_week' and 'is_weekend' features.")  # Log the new features

    # Adjust number columns to a standard scale (for better analysis later)
    for col in numerical_cols:
        if col not in ['hour', 'day', 'month', 'year']:  # Skip time-related numbers
            df[col] = (df[col] - df[col].mean()) / df[col].std()  # Normalize using z-score
            logger.info(f"Normalized {col} using z-score standardization.")  # Log the normalization

    return df  # Return the processed DataFrame

# Run the preprocessing
if __name__ == "__main__":
    if 'df' in locals():  # Check if data was loaded
        processed_df = preprocess_data(df)  # Process the data
        if processed_df is not None:  # If processing was successful
            print("Data preprocessing completed successfully.")  # Show success message
            print(processed_df.head())  # Show the first few rows to check
            # Save the processed data to a new CSV file in the script's folder
            output_file = os.path.join(script_dir, "preprocessed_data.csv")
            processed_df.to_csv(output_file, index=False)  # Save without row numbers
            logger.info(f"Preprocessed data saved to: {output_file}")  # Log the save
            print(f"Preprocessed data saved to: {output_file}")  # Show the save location
        else:
            print("Data preprocessing failed due to empty DataFrame.")  # Show if it failed
    else:
        print("No data loaded to preprocess.")  # Show if no data was loaded