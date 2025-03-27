"""
Inference script for filling missing values in weather datasets
using various interpolation methods.

This script takes datasets from testdata/input/, creates an altered version
by removing values from each column, applies interpolation methods to fill the
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
    cubic_spline
)

# Dictionary of interpolation methods
METHODS = {
    'Lagrange': lagrange_interpolation,
    'Newton': newton_interpolation,
    'Linear': linear_spline,
    'Cubic': cubic_spline
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
    Alter a dataset by removing a percentage of values from EACH column.
    
    Args:
        df: Input DataFrame
        percentage: Percentage of values to remove from each column (default: 10%)
        
    Returns:
        DataFrame with values removed and the indices of removed values
    """
    df_altered = df.copy()
    removed_indices = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Process each column separately
    for column in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
            
        # Calculate how many values to remove from this column
        col_length = len(df[column])
        n_remove = max(int(col_length * percentage / 100), 1)  # At least 1 value
        
        # Choose random indices to remove from this column
        # Ensure we don't remove too many (max 30% of column)
        n_remove = min(n_remove, int(col_length * 0.3))
        col_indices = np.random.choice(df.index, size=n_remove, replace=False)
        
        # Set these values to NaN
        for idx in col_indices:
            df_altered.loc[idx, column] = np.nan
            if idx not in removed_indices:
                removed_indices.append(idx)
    
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
    
    # Alter the dataset by removing values from each column
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
    
    # Dictionary to store results for each method
    results = {}
    
    # Apply each interpolation method
    for method_name, method_func in METHODS.items():
        print(f"\nApplying {method_name} interpolation...")
        
        # Create a copy of the original dataframe
        df_interpolated = df_altered.copy()
        
        # Track predictions and actual values for comparison
        predictions = []
        actual_values = []
        column_names = []
        indices = []
        
        # Apply interpolation to each column with missing values
        for parameter in missing_values.keys():
            print(f"  Interpolating {parameter}...")
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df_altered[parameter]):
                print(f"  Skipping non-numeric column: {parameter}")
                continue
            
            # Apply interpolation for this parameter
            df_interpolated = apply_interpolation(df_interpolated, method_func, parameter)
            
            # Store predictions and actual values for comparison
            missing_indices = df_altered.index[df_altered[parameter].isna()].tolist()
            for idx in missing_indices:
                actual_value = df.loc[idx, parameter]
                predicted_value = df_interpolated.loc[idx, parameter]
                
                predictions.append(predicted_value)
                actual_values.append(actual_value)
                column_names.append(parameter)
                indices.append(idx)
        
        # Store results for this method
        results[method_name] = {
            'dataframe': df_interpolated,
            'predictions': predictions,
            'actual_values': actual_values,
            'column_names': column_names,
            'indices': indices
        }
        
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
    
    # Create visualizations
    create_method_comparison_plot(df, df_altered, results, missing_values, output_dir, filename_no_ext)
    create_prediction_vs_actual_plots(results, output_dir, filename_no_ext)
    create_interpolation_curve_plots(df, df_altered, results, missing_values, output_dir, filename_no_ext)
    
    # Generate and save summary statistics
    create_summary_statistics(results, output_dir, filename_no_ext)

def create_method_comparison_plot(df_original, df_altered, results, parameters, output_dir, filename_base):
    """
    Create plots comparing all methods for each parameter.
    
    Args:
        df_original: Original DataFrame with complete data
        df_altered: Altered DataFrame with missing values
        results: Dictionary with results for each method
        parameters: Dict of parameters that were interpolated
        output_dir: Directory to save the visualizations
        filename_base: Base filename for the output files
    """
    # Ensure visualization directory exists
    viz_dir = os.path.join(output_dir, 'visualizations')
    ensure_dir_exists(viz_dir)
    
    for parameter in parameters.keys():
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df_original[parameter]):
            continue
        
        plt.figure(figsize=(14, 7))
        
        # Plot original data points
        plt.plot(df_original.index, df_original[parameter], 'ko-', alpha=0.3, label='Original Data')
        
        # Plot original values at missing points (for reference)
        missing_indices = df_altered.index[df_altered[parameter].isna()]
        plt.plot(missing_indices, df_original.loc[missing_indices, parameter], 
                'go', markersize=8, label='Actual Values (Missing)')
        
        # Plot the interpolated values from each method
        for method_name, method_results in results.items():
            df_interp = method_results['dataframe']
            plt.plot(missing_indices, df_interp.loc[missing_indices, parameter], 
                     'o', markersize=6, alpha=0.7, label=f'{method_name} Predicted')
        
        plt.title(f'Comparison of Interpolation Methods for {parameter}')
        plt.xlabel('Index')
        plt.ylabel(parameter)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        # Save the visualization
        safe_param = sanitize_filename(parameter)
        plt.savefig(os.path.join(viz_dir, f"{filename_base}_{safe_param}_method_comparison.png"))
        plt.close()

def create_prediction_vs_actual_plots(results, output_dir, filename_base):
    """
    Create scatter plots of predicted vs actual values for each method.
    
    Args:
        results: Dictionary with results for each method
        output_dir: Directory to save the visualizations
        filename_base: Base filename for the output files
    """
    # Ensure visualization directory exists
    viz_dir = os.path.join(output_dir, 'visualizations')
    ensure_dir_exists(viz_dir)
    
    # Create a combined plot with subplots for each method
    num_methods = len(results)
    if num_methods == 0:
        return  # No methods to plot
        
    fig, axs = plt.subplots(1, num_methods, figsize=(5*num_methods, 5))
    if num_methods == 1:
        axs = [axs]  # Make it iterable when there's only one subplot
    
    for i, (method_name, method_results) in enumerate(results.items()):
        predictions = method_results['predictions']
        actual_values = method_results['actual_values']
        
        if not predictions:  # Skip if no predictions were made
            continue
            
        # Create scatter plot
        axs[i].scatter(actual_values, predictions, alpha=0.6)
        
        # Add diagonal line (perfect predictions)
        if predictions and actual_values:  # Check if lists are not empty
            min_val = min(min(predictions), min(actual_values))
            max_val = max(max(predictions), max(actual_values))
            axs[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Calculate and display error metrics
        if predictions and actual_values:  # Check if lists are not empty
            mse = np.mean((np.array(actual_values) - np.array(predictions))**2)
            mae = np.mean(np.abs(np.array(actual_values) - np.array(predictions)))
            axs[i].set_title(f'{method_name}\nMSE: {mse:.4f}, MAE: {mae:.4f}')
        else:
            axs[i].set_title(f'{method_name}\nNo data to calculate metrics')
            
        axs[i].set_xlabel('Actual Values')
        axs[i].set_ylabel('Predicted Values' if i == 0 else '')
        axs[i].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{filename_base}_prediction_vs_actual.png"))
    plt.close()

def create_interpolation_curve_plots(df_original, df_altered, results, parameters, output_dir, filename_base):
    """
    Create plots showing the interpolation curve for each method and parameter.
    
    Args:
        df_original: Original DataFrame with complete data
        df_altered: Altered DataFrame with missing values
        results: Dictionary with results for each method
        parameters: Dict of parameters that were interpolated
        output_dir: Directory to save the visualizations
        filename_base: Base filename for the output files
    """
    from scipy import interpolate
    # Ensure visualization directory exists
    viz_dir = os.path.join(output_dir, 'visualizations')
    ensure_dir_exists(viz_dir)
    
    # Import required modules here to avoid potential import issues
    try:
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        has_inset_modules = True
    except ImportError:
        print("Warning: Could not import zoomed_inset_axes. Insets will be disabled.")
        has_inset_modules = False
    
    for parameter in parameters.keys():
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df_original[parameter]):
            continue
            
        # For each method, create a detailed plot of the interpolation curve
        for method_name, method_results in results.items():
            plt.figure(figsize=(15, 8))
            
            df_interp = method_results['dataframe']
            
            # Plot the original complete data as a thin gray line
            plt.plot(df_original.index, df_original[parameter], 'k-', alpha=0.2, 
                    linewidth=1, label='Original Data')
            
            # Plot known points (not missing in altered dataset)
            known_indices = df_altered.index[~df_altered[parameter].isna()]
            plt.plot(known_indices, df_altered.loc[known_indices, parameter], 
                    'bo', markersize=6, label='Known Data Points')
            
            # Plot missing points (actual values)
            missing_indices = df_altered.index[df_altered[parameter].isna()]
            plt.plot(missing_indices, df_original.loc[missing_indices, parameter], 
                    'go', markersize=8, label='Actual Values (Missing)')
            
            # Generate a higher density curve for smoother visualization
            # This helps show the true shape of the interpolation curve
            if len(df_interp.index) > 1:
                try:
                    # Create a smooth interpolation for visualization purposes
                    dense_x = np.linspace(min(df_interp.index), max(df_interp.index), 1000)
                    
                    if method_name == 'Cubic':
                        # Use the same method as in the cubic_spline function
                        valid_indices = df_interp.index[~df_interp[parameter].isna()].tolist()
                        valid_values = df_interp.loc[valid_indices, parameter].values
                        x_valid = np.array(valid_indices, dtype=float)
                        y_valid = np.array(valid_values, dtype=float)
                        
                        if len(x_valid) >= 4:
                            # Use UnivariateSpline for smooth visualization
                            s = len(x_valid) * 0.01  # Small smoothing factor
                            f = interpolate.UnivariateSpline(x_valid, y_valid, k=3, s=s)
                            dense_y = f(dense_x)
                            plt.plot(dense_x, dense_y, 'r-', linewidth=2, alpha=0.7, 
                                    label=f'{method_name} Interpolation Curve')
                        else:
                            # Not enough points for spline visualization
                            plt.plot(df_interp.index, df_interp[parameter], 'r-', linewidth=2, 
                                    alpha=0.7, label=f'{method_name} Interpolation Curve')
                    else:
                        # For other methods, use the regular curve
                        plt.plot(df_interp.index, df_interp[parameter], 'r-', linewidth=2, 
                                alpha=0.7, label=f'{method_name} Interpolation Curve')
                except Exception as e:
                    print(f"Warning: Couldn't create smooth visualization curve: {e}")
                    # Fall back to regular line plot
                    plt.plot(df_interp.index, df_interp[parameter], 'r-', linewidth=2, 
                            alpha=0.7, label=f'{method_name} Interpolation Curve')
            else:
                # Fall back for very small datasets
                plt.plot(df_interp.index, df_interp[parameter], 'r-', linewidth=2, 
                        alpha=0.7, label=f'{method_name} Interpolation Curve')
            
            # Plot the interpolated points
            plt.plot(missing_indices, df_interp.loc[missing_indices, parameter], 
                    'rx', markersize=8, label=f'{method_name} Predicted Points')
            
            plt.title(f'{method_name} Interpolation Curve for {parameter}')
            plt.xlabel('Index')
            plt.ylabel(parameter)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            # Try to create inset zoom, but with better error handling
            if has_inset_modules and len(missing_indices) >= 3:
                try:
                    # Find a reasonable zoom region
                    consecutive_groups = group_consecutive_indices(missing_indices)
                    if consecutive_groups:
                        largest_group = max(consecutive_groups, key=len)
                        if len(largest_group) >= 2:  # Need at least 2 consecutive missing values
                            # Calculate range for zoom (add buffer)
                            buffer = 5
                            zoom_start = max(0, min(largest_group) - buffer)
                            zoom_end = min(len(df_original), max(largest_group) + buffer)
                            
                            # Calculate sensible y-limits
                            y_values = df_original.loc[zoom_start:zoom_end, parameter].dropna()
                            if not y_values.empty:
                                y_min, y_max = y_values.min(), y_values.max()
                                y_range = max(y_max - y_min, 1e-9)  # Prevent zero range
                                y_min = y_min - 0.1*y_range
                                y_max = y_max + 0.1*y_range
                                
                                # Create the inset axes
                                axins = zoomed_inset_axes(plt.gca(), zoom=2.5, loc='upper left')
                                
                                # Plot data in the inset
                                axins.plot(df_original.index, df_original[parameter], 'k-', alpha=0.2, linewidth=1)
                                axins.plot(known_indices, df_altered.loc[known_indices, parameter], 'bo', markersize=5)
                                axins.plot(missing_indices, df_original.loc[missing_indices, parameter], 'go', markersize=6)
                                axins.plot(df_interp.index, df_interp[parameter], 'r-', linewidth=1.5, alpha=0.7)
                                axins.plot(missing_indices, df_interp.loc[missing_indices, parameter], 'rx', markersize=6)
                                
                                # Set limits for the inset
                                axins.set_xlim(zoom_start, zoom_end)
                                axins.set_ylim(y_min, y_max)
                                
                                # Style the inset
                                axins.grid(True, linestyle='--', alpha=0.4)
                                
                                # Add connecting lines between the inset and the main plot
                                mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")
                except Exception as e:
                    print(f"Warning: Couldn't create inset zoom: {e}")
            
            plt.tight_layout()
            
            # Save the visualization
            safe_param = sanitize_filename(parameter)
            safe_method = sanitize_filename(method_name)
            plt.savefig(os.path.join(viz_dir, f"{filename_base}_{safe_param}_{safe_method}_curve.png"))
            plt.close()

def group_consecutive_indices(indices):
    """
    Group consecutive indices together.
    
    Args:
        indices: List of indices to group
        
    Returns:
        List of lists containing consecutive indices
    """
    if not indices:
        return []
        
    sorted_indices = sorted(indices)
    groups = []
    current_group = [sorted_indices[0]]
    
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i-1] + 1:
            current_group.append(sorted_indices[i])
        else:
            groups.append(current_group)
            current_group = [sorted_indices[i]]
    
    if current_group:
        groups.append(current_group)
    
    return groups

def create_summary_statistics(results, output_dir, filename_base):
    """
    Create and save summary statistics for all interpolation methods.
    
    Args:
        results: Dictionary with results for each method
        output_dir: Directory to save the summary
        filename_base: Base filename for the output files
    """
    # Calculate error metrics for each method
    summary = []
    
    for method_name, method_results in results.items():
        predictions = method_results['predictions']
        actual_values = method_results['actual_values']
        
        # Skip if no predictions were made
        if not predictions:
            continue
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        # Calculate error metrics
        mse = np.mean((actual_values - predictions)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_values - predictions))
        
        # Handle potential division by zero in MAPE calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = np.nan  # Set to NaN if calculation fails
        
        summary.append({
            'Method': method_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape if not np.isnan(mape) else "N/A",
            'Number of Points': len(predictions)
        })
    
    # Convert to DataFrame and save as CSV
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, f"{filename_base}_error_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved error metrics to {summary_path}")
    else:
        print("No summary statistics to save - no predictions were made")
        return
    
    # Create a more detailed summary with per-point comparisons
    detailed_data = []
    
    # Get all unique (index, column) pairs across all methods
    all_pairs = set()
    for method_results in results.values():
        for idx, col in zip(method_results['indices'], method_results['column_names']):
            all_pairs.add((idx, col))
    
    # For each pair, collect predictions from all methods
    for idx, col in all_pairs:
        row = {'Index': idx, 'Column': col}
        
        # Get actual value
        for method_name, method_results in results.items():
            # Find position of this pair in the method's results
            try:
                positions = [(i, c) for i, c in zip(method_results['indices'], 
                                             method_results['column_names'])]
                if (idx, col) in positions:
                    pos = positions.index((idx, col))
                    row[f'{method_name}_Predicted'] = method_results['predictions'][pos]
                    row['Actual_Value'] = method_results['actual_values'][pos]
                else:
                    row[f'{method_name}_Predicted'] = np.nan
            except (ValueError, IndexError):
                row[f'{method_name}_Predicted'] = np.nan
        
        detailed_data.append(row)
    
    # Convert to DataFrame and save as CSV
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = os.path.join(output_dir, f"{filename_base}_detailed_comparison.csv")
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Saved detailed comparison to {detailed_path}")
    
    # Create a comparison visualization
    plt.figure(figsize=(10, 6))
    methods = [row['Method'] for row in summary]
    
    # Extract metrics that are numeric
    metrics = ['MSE', 'RMSE', 'MAE']  # Exclude MAPE as it might be N/A
    
    x = np.arange(len(methods))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [row[metric] for row in summary]
        plt.bar(x + i*width - width*1.5/2, values, width, label=metric)
    
    plt.xlabel('Interpolation Method')
    plt.ylabel('Error Value (lower is better)')
    plt.title('Comparison of Interpolation Methods')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the comparison visualization
    plt.savefig(os.path.join(output_dir, 'visualizations', f"{filename_base}_method_comparison_metrics.png"))
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
