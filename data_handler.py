"""
Module for loading, altering, and managing weather datasets.
"""
import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load a weather dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame containing the weather data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def alter_dataset(df, parameter, percentage=10):
    """
    Alter a dataset by removing a percentage of values from a specific parameter.
    
    Args:
        df: Input DataFrame
        parameter: Column name of the parameter to alter
        percentage: Percentage of values to remove (default: 10%)
        
    Returns:
        tuple: (altered_df, removed_indices, true_values)
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_altered = df.copy()
    
    # Total number of data points
    n = len(df)
    
    # Number of points to remove
    n_remove = int(n * percentage / 100)
    
    # Generate random indices to remove
    # We'll use a fixed seed for reproducibility
    np.random.seed(42)
    removed_indices = np.random.choice(df.index, size=n_remove, replace=False)
    removed_indices = sorted(removed_indices)
    
    # Store the true values before removing them
    true_values = df.loc[removed_indices, parameter].values
    
    # Set the values to NaN in the altered dataset
    df_altered.loc[removed_indices, parameter] = np.nan
    
    print(f"Removed {n_remove} values ({percentage}%) from {parameter}")
    
    return df_altered, removed_indices, true_values

def save_results(results, output_dir='results'):
    """
    Save the results to CSV files.
    
    Args:
        results: Dictionary containing the results
        output_dir: Directory to save the results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Convert results to DataFrame and save
    for dataset_name, dataset_results in results.items():
        for param, param_results in dataset_results.items():
            # Create a DataFrame for current parameter
            errors_df = pd.DataFrame(param_results).T
            
            # Save to CSV
            filename = f"{output_dir}/{dataset_name}_{param}_errors.csv"
            errors_df.to_csv(filename)
            print(f"Saved results to {filename}")
