import os
import glob
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the data directory (updated path)
data_dir = "C:/Users/abdul/Music/DS_Assignment_2/raw"

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load and merge files
def load_and_merge_data(directory):
    # Load and merge weather data (CSV files)
    weather_files = glob.glob(os.path.join(directory, "weather_raw_data", "*.csv"))
    weather_dfs = []
    
    for file in weather_files:
        try:
            df = pd.read_csv(file)
            logger.info(f"Successfully loaded weather CSV file: {file}")
            weather_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading weather CSV file {file}: {e}")
    
    # Merge all weather data into a single DataFrame
    if weather_dfs:
        weather_df = pd.concat(weather_dfs, ignore_index=True)
        weather_df.columns = weather_df.columns.str.lower().str.replace(' ', '_')
        logger.info(f"Weather data merged. Records: {weather_df.shape[0]}, Features: {weather_df.shape[1]}")
    else:
        logger.warning("No weather data files loaded.")
        weather_df = pd.DataFrame()  # Empty DataFrame if no weather files

    # Load and merge electricity data (JSON files)
    electricity_files = glob.glob(os.path.join(directory, "electricity_raw_data", "*.json"))
    electricity_dfs = []
    
    for file in electricity_files:
        try:
            df = pd.read_json(file)
            # Assuming the JSON has a 'data' key with a list of dictionaries
            if 'data' in df.columns:
                df = pd.json_normalize(df['data'])
            logger.info(f"Successfully loaded electricity JSON file: {file}")
            electricity_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading electricity JSON file {file}: {e}")
    
    # Merge all electricity data into a single DataFrame
    if electricity_dfs:
        electricity_df = pd.concat(electricity_dfs, ignore_index=True)
        electricity_df.columns = electricity_df.columns.str.lower().str.replace(' ', '_')
        logger.info(f"Electricity data merged. Records: {electricity_df.shape[0]}, Features: {electricity_df.shape[1]}")
    else:
        logger.warning("No electricity data files loaded.")
        electricity_df = pd.DataFrame()  # Empty DataFrame if no electricity files

    # Combine weather and electricity data
    combined_df = pd.concat([weather_df, electricity_df], ignore_index=True) if not weather_df.empty and not electricity_df.empty else pd.concat([weather_df, electricity_df], ignore_index=True)
    
    if not combined_df.empty:
        # Check for immediate anomalies (e.g., missing values)
        missing_values = combined_df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
        
        # Log the final number of records and features
        num_records = combined_df.shape[0]
        num_features = combined_df.shape[1]
        logger.info(f"Final combined data. Total records: {num_records}, Total features: {num_features}")
        
        return combined_df
    else:
        logger.error("No data was loaded successfully.")
        return None

# Execute the data loading and merging
if __name__ == "__main__":
    data = load_and_merge_data(data_dir)
    if data is not None:
        print("Data loading and merging completed successfully.")
        print(data.head())  # Display the first few rows to verify
        
        # Save the combined data to a CSV file in the script's directory
        output_file = os.path.join(script_dir, "combined_data.csv")
        data.to_csv(output_file, index=False)
        logger.info(f"Combined data saved to: {output_file}")
        print(f"Combined data saved to: {output_file}")
    else:
        print("Data loading and merging failed.")