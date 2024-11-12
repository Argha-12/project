# Data Cleaning and Preparation: 
# Write a Python script to clean and prepare banking transaction data, handling missing values, and removing outliers based on a specified threshold.

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Loads banking transaction data from a CSV file.
    
    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv("C:\\Users\\argha\\Desktop\\argha\\git\\project\\banking_Analysis_project\\data_set\\Comprehensive_Banking_Database.csv")
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print("Error: The file was not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty. Please provide a valid CSV file.")
    except pd.errors.ParserError:
        print("Error: There was an error parsing the file. Ensure it is a proper CSV format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def clean_banking_data(df, outlier_threshold=3):
    """
    Cleans banking transaction data by handling missing values and removing outliers.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing banking transaction data.
    - outlier_threshold (int or float): The Z-score threshold to identify outliers (default is 3).

    Returns:
    - pd.DataFrame: A cleaned DataFrame.
    """
    
    # Step 1: Handle missing values
    # Fill numerical columns with the mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
    print(df[numeric_cols].head(10))
    
    # Fill categorical columns with the mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

    # Step 2: Remove outliers
    # Calculate Z-scores for numeric columns
    for col in numeric_cols:
        col_zscore = (df[col] - df[col].mean()) / df[col].std()
        df = df[(col_zscore.abs() <= outlier_threshold)]
    
    return df

# Example usage:
# Load the data
file_path = 'path/to/your/banking_data.csv'  # Replace with your actual file path
df = load_data(file_path)

# Clean the data if loading was successful
if df is not None:
    df_cleaned = clean_banking_data(df, outlier_threshold=3)
    print("Data cleaning complete.")

 # 1. Average Transaction Amount (across the entire dataset)

avg_transaction = df['Transaction Amount'].mean()
df['avg_transaction_amount'] = avg_transaction
print (df['avg_transaction_amount'] )