"""
Inference script for filling missing values in weather datasets
using various interpolation methods.

This script takes datasets from testdata/input/, creates an altered version
by removing 10-15% of the values, applies interpolation methods to fill the
missing values, and saves the results to testdata/output/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import re
import warnings
warnings.filterwarnings("ignore")

# Import interpolation methods
from interpolation import (
    lagrange_interpolation, 
    newton_interpolation,
    linear_spline, 
    cubic_spline, 
    polynomial_interpolation
)

# Dictionary of interpolation methods
METHODS = {
    'Lagrange': lagrange_interpolation,
    'Newton': newton_interpolation,
    'Linear': linear_spline,
    'Cubic': cubic_spline,
    'Polynomial': polynomial_interpolation
}

def sanitize_filename(name):
    """
    Sanitize a string to make it a valid filename.
    Removes special characters and replaces spaces with underscores.
    
    Args:
        name: The string to sanitize
        
    Returns:
        Sanitized string suitable for use as a filename
    """
    # Replace special characters with underscores
    sanitized = re.sub(r'[^\w\s-]', '_', name)
    # Replace spaces with underscores
    sanitized = re.sub(r'[\s]+', '_', sanitized)
    return sanitized

def ensure_dir_exists(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)

def detect_missing_values(df):
    """
    Detect columns with missing values and their counts.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with columns having missing values and their counts
    """
    missing_values = {}
    for column in df.columns:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            missing_values[column] = missing_count
    
    return missing_values

def apply_interpolation(df, method_func, parameter):
    """
    Apply an interpolation method to fill missing values in a specific parameter.
    
    Args:
        df: DataFrame with missing values
        method_func: Interpolation function to apply
        parameter: Column name to interpolate
        
    Returns:
        DataFrame with interpolated values
    """
    # Create a copy of the dataframe
    df_result = df.copy()
    
    # Get indices of missing values
    missing_indices = df_result.index[df_result[parameter].isna()].tolist()
    
    if not missing_indices:
        print(f"No missing values found in {parameter}")
        return df_result
    
    try:
        # Apply interpolation method to predict missing values
        predictions = method_func(df_result, parameter, missing_indices)
        
        # Fill in the missing values
        for idx, pred_value in zip(missing_indices, predictions):
            df_result.loc[idx, parameter] = pred_value
        
        return df_result
    
    except Exception as e:
        print(f"Error applying interpolation: {e}")
        return df  # Return original if interpolation fails

def alter_dataset(df, percentage=10):
    """
    Alter a dataset by removing a percentage of values from all columns.
    
    Args:
        df: Input DataFrame
        percentage: Percentage of values to remove (default: 10%)
        
    Returns:
        DataFrame with values removed and the indices of removed values
    """
    df_altered = df.copy()
    total_values = df.size
    n_remove = int(total_values * percentage / 100)
    
    # Ensure n_remove does not exceed the number of rows
    n_remove = min(n_remove, df.shape[0])
    
    # Generate random indices to remove
    np.random.seed(42)
    removed_indices = np.random.choice(df.index, size=n_remove, replace=False)
    
    # Set the values to NaN in the altered dataset
    for idx in removed_indices:
        col = np.random.choice(df.columns)
        df_altered.loc[idx, col] = np.nan
    
    return df_altered, removed_indices

def interpolate_dataset(input_file, output_dir):
    """
    Interpolate a single dataset file using all methods.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the results
    """
    print(f"\nProcessing file: {input_file}")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return
    
    # Alter the dataset by removing 10-15% of the values
    altered_percentage = np.random.uniform(10, 15)
    df_altered, removed_indices = alter_dataset(df, percentage=altered_percentage)
    
    # Save the altered dataset
    base_filename = os.path.basename(input_file)
    filename_no_ext = os.path.splitext(base_filename)[0]
    altered_filename = f"altered_{filename_no_ext}.csv"
    altered_path = os.path.join(output_dir, altered_filename)
    df_altered.to_csv(altered_path, index=False)
    print(f"Saved altered dataset to {altered_path}")
    
    # Detect missing values
    missing_values = detect_missing_values(df_altered)
    
    if not missing_values:
        print("No missing values found in the altered dataset")
        return
    
    print(f"Found missing values in {len(missing_values)} columns:")
    for col, count in missing_values.items():
        print(f"  - {col}: {count} missing values")
    
    # Dictionary to store predictions and errors
    predictions = {method_name: [] for method_name in METHODS.keys()}
    errors = {method_name: [] for method_name in METHODS.keys()}
    
    # Apply each interpolation method
    for method_name, method_func in METHODS.items():
        print(f"\nApplying {method_name} interpolation...")
        
        # Create a copy of the original dataframe
        df_interpolated = df_altered.copy()
        
        # Apply interpolation to each column with missing values
        for parameter in missing_values.keys():
            print(f"  Interpolating {parameter}...")
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df_altered[parameter]):
                print(f"  Skipping non-numeric column: {parameter}")
                continue
            
            # Apply interpolation for this parameter
            df_interpolated = apply_interpolation(df_interpolated, method_func, parameter)
            
            # Store predictions and errors
            missing_indices = df_altered.index[df_altered[parameter].isna()].tolist()
            for idx in missing_indices:
                original_value = df.loc[idx, parameter]
                predicted_value = df_interpolated.loc[idx, parameter]
                predictions[method_name].append(predicted_value)
                errors[method_name].append(np.abs(original_value - predicted_value))
        
        # Check if interpolation filled all values
        remaining_missing = 0
        for parameter in missing_values.keys():
            if pd.api.types.is_numeric_dtype(df_interpolated[parameter]):
                remaining_missing += df_interpolated[parameter].isna().sum()
        
        if remaining_missing > 0:
            print(f"  Warning: {remaining_missing} values could not be interpolated with {method_name}")
        else:
            print(f"  Successfully filled all missing values with {method_name}")
        
        # Save the interpolated dataset
        output_filename = f"{filename_no_ext}_{sanitize_filename(method_name)}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        df_interpolated.to_csv(output_path, index=False)
        print(f"  Saved interpolated dataset to {output_path}")
        
        # Create a plot visualizing the filled values for all columns
        create_visualization(df, df_altered, df_interpolated, list(missing_values.keys()), 
                            method_name, output_dir, filename_no_ext)
        
        # Create a plot focusing on index range 200-250
        create_detailed_visualization(df, df_altered, df_interpolated, list(missing_values.keys()), 
                                      method_name, output_dir, filename_no_ext, 200, 250)
    
    # Save the summary file
    summary_data = []
    for idx in missing_indices:
        row = {"Index": idx, "Original Value": df.loc[idx, parameter]}
        for method_name in METHODS.keys():
            row[f"{method_name} Predicted"] = predictions[method_name][missing_indices.index(idx)]
            row[f"{method_name} Error"] = errors[method_name][missing_indices.index(idx)]
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"{filename_no_ext}_summary.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary file to {summary_path}")
    
    # Calculate and print average errors
    avg_errors = {method_name: np.mean(errors[method_name]) for method_name in METHODS.keys()}
    for method_name, avg_error in avg_errors.items():
        print(f"Average error for {method_name}: {avg_error}")

def create_visualization(df_original, df_altered, df_interpolated, parameters, method_name, output_dir, filename_base):
    """
    Create visualizations of the interpolation results.
    
    Args:
        df_original: Original DataFrame with complete data
        df_altered: Altered DataFrame with missing values
        df_interpolated: DataFrame with interpolated values
        parameters: List of parameters that were interpolated
        method_name: Name of the interpolation method
        output_dir: Directory to save the visualizations
        filename_base: Base filename for the output files
    """
    # Ensure visualization directory exists
    viz_dir = os.path.join(output_dir, 'visualizations')
    ensure_dir_exists(viz_dir)
    
    for parameter in parameters:
        # Skip non-numeric columns or those without missing values
        if not pd.api.types.is_numeric_dtype(df_original[parameter]) or not df_altered[parameter].isna().any():
            continue
        
        plt.figure(figsize=(12, 6))
        
        # Plot original data (excluding NaN values)
        original_indices = df_original.index[~df_original[parameter].isna()]
        plt.plot(original_indices, df_original.loc[original_indices, parameter], 
                'go', label='Original Data Points')
        
        # Plot deleted data points
        deleted_indices = df_altered.index[df_altered[parameter].isna()]
        plt.plot(deleted_indices, df_original.loc[deleted_indices, parameter], 
                'ro', label='Deleted Data Points')
        
        # Plot interpolated values (only the points that were originally missing)
        missing_indices = df_altered.index[df_altered[parameter].isna()]
        plt.plot(missing_indices, df_interpolated.loc[missing_indices, parameter], 
                'rx', markersize=8, label=f'{method_name} Interpolated Points')
        
        # Plot the full interpolated series with a line to show the trend
        plt.plot(df_interpolated.index, df_interpolated[parameter], 'b-', alpha=0.3, 
                label='Complete Interpolated Series')
        
        plt.title(f'{method_name} Interpolation for {parameter}')
        plt.xlabel('Index')
        plt.ylabel(parameter)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Save the visualization
        safe_param = sanitize_filename(parameter)
        safe_method = sanitize_filename(method_name)
        filename = f"{filename_base}_{safe_param}_{safe_method}.png"
        plt.savefig(os.path.join(viz_dir, filename))
        plt.close()

def create_detailed_visualization(df_original, df_altered, df_interpolated, parameters, method_name, output_dir, filename_base, start_idx, end_idx):
    """
    Create detailed visualizations of the interpolation results for a specific index range.
    
    Args:
        df_original: Original DataFrame with complete data
        df_altered: Altered DataFrame with missing values
        df_interpolated: DataFrame with interpolated values
        parameters: List of parameters that were interpolated
        method_name: Name of the interpolation method
        output_dir: Directory to save the visualizations
        filename_base: Base filename for the output files
        start_idx: Start index for the detailed view
        end_idx: End index for the detailed view
    """
    # Ensure visualization directory exists
    viz_dir = os.path.join(output_dir, 'visualizations')
    ensure_dir_exists(viz_dir)
    
    for parameter in parameters:
        # Skip non-numeric columns or those without missing values
        if not pd.api.types.is_numeric_dtype(df_original[parameter]) or not df_altered[parameter].isna().any():
            continue
        
        plt.figure(figsize=(12, 6))
        
        # Plot original data (excluding NaN values)
        original_indices = df_original.index[(~df_original[parameter].isna()) & (df_original.index >= start_idx) & (df_original.index <= end_idx)]
        plt.plot(original_indices, df_original.loc[original_indices, parameter], 
                'go', label='Original Data Points')
        
        # Plot deleted data points
        deleted_indices = df_altered.index[(df_altered[parameter].isna()) & (df_altered.index >= start_idx) & (df_altered.index <= end_idx)]
        plt.plot(deleted_indices, df_original.loc[deleted_indices, parameter], 
                'ro', label='Deleted Data Points')
        
        # Plot interpolated values (only the points that were originally missing)
        missing_indices = df_altered.index[(df_altered[parameter].isna()) & (df_altered.index >= start_idx) & (df_altered.index <= end_idx)]
        plt.plot(missing_indices, df_interpolated.loc[missing_indices, parameter], 
                'rx', markersize=8, label=f'{method_name} Interpolated Points')
        
        # Plot the full interpolated series with a line to show the trend
        detailed_indices = df_interpolated.index[(df_interpolated.index >= start_idx) & (df_interpolated.index <= end_idx)]
        plt.plot(detailed_indices, df_interpolated.loc[detailed_indices, parameter], 'b-', alpha=0.3, 
                label='Complete Interpolated Series')
        
        plt.title(f'{method_name} Interpolation for {parameter} (Index {start_idx}-{end_idx})')
        plt.xlabel('Index')
        plt.ylabel(parameter)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Save the visualization
        safe_param = sanitize_filename(parameter)
        safe_method = sanitize_filename(method_name)
        filename = f"{filename_base}_{safe_param}_{safe_method}_detailed_{start_idx}_{end_idx}.png"
        plt.savefig(os.path.join(viz_dir, filename))
        plt.close()

def main():
    """Main function to process all datasets."""
    # Define input and output directories
    input_dir = 'testdata/input'
    output_dir = 'testdata/output'
    
    # Ensure directories exist
    ensure_dir_exists(input_dir)
    ensure_dir_exists(output_dir)
    
    # Get list of all CSV files in input directory
    input_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not input_files:
        print(f"No CSV files found in {input_dir}. Please add input files.")
        return
    
    print(f"Found {len(input_files)} input files to process.")
    
    # Process each input file
    for input_file in input_files:
        interpolate_dataset(input_file, output_dir)
    
    print("\nInterpolation complete! Results are saved in the 'testdata/output/' directory.")
    print("Visualizations are available in the 'testdata/output/visualizations/' directory.")

if __name__ == "__main__":
    # Record start time
    start_time = datetime.now()
    print(f"Starting inference at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Inference completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time: {duration}")
