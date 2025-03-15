import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis

# Set up logging to both console and file
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the folder where this script is saved
log_file = os.path.join(script_dir, "eda.log")  # Create a log file name in the same folder
logging.basicConfig(
    level=logging.INFO,  # Show all important messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format with time, level, and message
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]  # Save to file and show on screen
)
logger = logging.getLogger(__name__)

# Load the preprocessed data from Task 2
input_file = os.path.join(script_dir, "preprocessed_data.csv")  # Path to the preprocessed file
try:
    df = pd.read_csv(input_file)  # Read the CSV into a DataFrame
    logger.info(f"Successfully loaded preprocessed data from: {input_file}")
except Exception as e:
    logger.error(f"Error loading preprocessed data: {e}")
    raise

# Function to perform Exploratory Data Analysis
def perform_eda(df):
    # Ensure 'date' is in datetime format for time series
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601')  # Use ISO8601 format to handle timezone and precision
            logger.info("Successfully converted 'date' to datetime with ISO8601 format.")
        except Exception as e:
            logger.error(f"Error converting 'date' to datetime: {e}")
            return
    else:
        logger.warning("No 'date' column found for time series analysis.")
        return

    # --- Statistical Summary ---
    logger.info("Computing statistical summary for numerical features.")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns  # Get all number columns
    stats_summary = pd.DataFrame()  # Create a table for stats
    for col in numerical_cols:
        stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': skew(df[col].dropna()),  # Measure of asymmetry
            'kurtosis': kurtosis(df[col].dropna())  # Measure of peakedness
        }
        stats_summary[col] = pd.Series(stats)
    stats_summary = stats_summary.T  # Transpose for better readability
    logger.info("Statistical summary computed.")
    stats_output_file = os.path.join(script_dir, "stats_summary.csv")
    stats_summary.to_csv(stats_output_file)  # Save to CSV
    logger.info(f"Statistical summary saved to: {stats_output_file}")
    print("Statistical Summary:\n", stats_summary)

    # --- Time Series Analysis ---
    logger.info("Performing time series analysis for electricity demand.")
    if 'temperature_2m' in df.columns:  # Assuming temperature_2m represents electricity demand for this example
        plt.figure(figsize=(12, 6))  # Set plot size
        plt.plot(df['date'], df['temperature_2m'], label='Temperature (Electricity Demand Proxy)')  # Plot over time
        plt.title('Electricity Demand Over Time')  # Add title
        plt.xlabel('Date')  # Label x-axis
        plt.ylabel('Temperature (Demand)')  # Label y-axis
        plt.legend()  # Show legend
        plt.grid(True)  # Add grid for readability
        # Annotate trends (example: assume a peak in the middle)
        plt.annotate('Possible Peak', xy=(df['date'].iloc[len(df)//2], df['temperature_2m'].iloc[len(df)//2]),
                     xytext=(df['date'].iloc[len(df)//2], df['temperature_2m'].iloc[len(df)//2] + 0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.tight_layout()  # Adjust layout
        time_series_plot = os.path.join(script_dir, "time_series_plot.png")
        plt.savefig(time_series_plot)  # Save the plot
        plt.close()  # Close the figure to free memory
        logger.info(f"Time series plot saved to: {time_series_plot}")
    else:
        logger.warning("No 'temperature_2m' column found for time series analysis.")

    # --- Univariate Analysis ---
    logger.info("Performing univariate analysis for key numerical features.")
    for col in ['temperature_2m', 'hour', 'day', 'month', 'year']:  # Key features to analyze
        if col in df.columns:
            plt.figure(figsize=(12, 4))
            # Histogram
            plt.subplot(1, 3, 1)
            sns.histplot(df[col].dropna(), kde=True)  # Histogram with density curve
            plt.title(f'Histogram of {col}')
            # Boxplot
            plt.subplot(1, 3, 2)
            sns.boxplot(y=df[col].dropna())  # Boxplot to show spread
            plt.title(f'Boxplot of {col}')
            # Density Plot
            plt.subplot(1, 3, 3)
            sns.kdeplot(df[col].dropna())  # Density plot
            plt.title(f'Density Plot of {col}')
            plt.tight_layout()
            uni_plot = os.path.join(script_dir, f"univariate_{col}.png")
            plt.savefig(uni_plot)  # Save each plot
            plt.close()
            logger.info(f"Univariate plot for {col} saved to: {uni_plot}")
            # Comments on distribution
            logger.info(f"{col} Distribution: Mean={df[col].mean():.2f}, Median={df[col].median():.2f}, "
                        f"Spread={df[col].std():.2f}, Shape={'skewed' if abs(skew(df[col].dropna())) > 1 else 'symmetric'}")

    # --- Correlation Analysis ---
    logger.info("Performing correlation analysis.")
    corr_matrix = df[numerical_cols].corr()  # Calculate correlation between numerical columns
    plt.figure(figsize=(10, 8))  # Set plot size
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)  # Create heatmap
    plt.title('Correlation Matrix of Numerical Features')
    corr_plot = os.path.join(script_dir, "correlation_heatmap.png")
    plt.savefig(corr_plot)  # Save the heatmap
    plt.close()
    logger.info(f"Correlation heatmap saved to: {corr_plot}")
    print("Correlation Matrix:\n", corr_matrix)
    # Check for multicollinearity (correlation > 0.8 or < -0.8)
    high_corr = np.where(np.abs(corr_matrix) > 0.8)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                       for x, y in zip(*high_corr) if x != y and x < y]
    if high_corr_pairs:
        logger.warning(f"Potential multicollinearity detected: {high_corr_pairs}")
    else:
        logger.info("No significant multicollinearity detected.")

    # --- Advanced Time Series Techniques ---
    logger.info("Performing advanced time series analysis.")
    if len(df) > 1 and 'temperature_2m' in df.columns:  # Ensure enough data
        # Decomposition
        decomposition = seasonal_decompose(df.set_index('date')['temperature_2m'], period=12)  # Assume monthly seasonality
        decomposition.plot()
        decomp_plot = os.path.join(script_dir, "time_series_decomposition.png")
        plt.savefig(decomp_plot)  # Save decomposition plot
        plt.close()
        logger.info(f"Time series decomposition saved to: {decomp_plot}")

        # Stationarity Test (Augmented Dickey-Fuller)
        adf_test = adfuller(df['temperature_2m'].dropna())
        logger.info(f"ADF Test Results: Statistic={adf_test[0]:.2f}, p-value={adf_test[1]:.2f}, "
                    f"Critical Values={adf_test[4]}")
        if adf_test[1] < 0.05:
            logger.info("Time series is stationary (p-value < 0.05).")
        else:
            logger.warning("Time series is not stationary (p-value >= 0.05). Consider differencing.")
    else:
        logger.warning("Insufficient data or no 'temperature_2m' for advanced time series analysis.")

# Run the EDA
if __name__ == "__main__":
    if 'df' in locals() and not df.empty:  # Check if data is loaded and not empty
        perform_eda(df)  # Perform the analysis
        logger.info("EDA completed successfully.")
        print("EDA completed successfully. Check the script directory for output files.")
    else:
        logger.error("No valid data to perform EDA.")
        print("EDA failed due to no valid data.")