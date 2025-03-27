"""
Inference script for filling missing values in weather datasets
using various interpolation methods.

This script takes datasets from testdata/input/, creates an altered version
by removing values from each column, applies interpolation methods to fill the
missing values, and saves the results to testdata/output/.

Author: Gaurav Jha
Last updated: 2025-03-27 18:23:08
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
        col_length = len(df[column].dropna())
        n_remove = max(int(col_length * percentage / 100), 1)  # At least 1 value
        
        # Choose random indices to remove from this column
        # Ensure we don't remove too many (max 30% of column)
        n_remove = min(n_remove, int(col_length * 0.3))
        available_indices = df.index[~df[column].isna()].tolist()
        
        if available_indices:  # Only proceed if there are valid values
            col_indices = np.random.choice(available_indices, size=min(n_remove, len(available_indices)), replace=False)
            
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
    
    # Dictionary to store results for each method and parameter
    results = {}
    parameter_results = {}
    
    # Initialize parameter results structure
    for parameter in missing_values.keys():
        if pd.api.types.is_numeric_dtype(df_altered[parameter]):
            parameter_results[parameter] = {
                method_name: {'predictions': [], 'actual_values': [], 'indices': []}
                for method_name in METHODS.keys()
            }
    
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
                if pd.isna(df.loc[idx, parameter]):
                    # Skip if original value was also NaN
                    continue
                    
                actual_value = df.loc[idx, parameter]
                predicted_value = df_interpolated.loc[idx, parameter]
                
                if pd.isna(predicted_value):
                    # Skip if prediction failed
                    continue
                
                # Store in overall results
                predictions.append(predicted_value)
                actual_values.append(actual_value)
                column_names.append(parameter)
                indices.append(idx)
                
                # Store in parameter-specific results
                parameter_results[parameter][method_name]['predictions'].append(predicted_value)
                parameter_results[parameter][method_name]['actual_values'].append(actual_value)
                parameter_results[parameter][method_name]['indices'].append(idx)
        
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
    
    # Create only the interpolation curve plots and error metrics plots
    create_interpolation_curve_plots(df, df_altered, results, missing_values, output_dir, filename_no_ext)
    create_error_metrics_by_parameter(parameter_results, output_dir, filename_no_ext)
    
    # Generate and save summary statistics
    create_summary_statistics(results, parameter_results, output_dir, filename_no_ext)

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
            known_indices = df_altered.index[~df_altered[parameter].isna()].tolist()
            plt.plot(known_indices, df_altered.loc[known_indices, parameter], 
                    'bo', markersize=6, label='Known Data Points')
            
            # Plot missing points (actual values)
            missing_indices = df_altered.index[df_altered[parameter].isna()].tolist()
            actual_values_indices = []
            for idx in missing_indices:
                if not pd.isna(df_original.loc[idx, parameter]):
                    actual_values_indices.append(idx)
            
            if actual_values_indices:
                plt.plot(actual_values_indices, df_original.loc[actual_values_indices, parameter], 
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
            interpolated_indices = []
            for idx in missing_indices:
                if not pd.isna(df_interp.loc[idx, parameter]) and not pd.isna(df_original.loc[idx, parameter]):
                    interpolated_indices.append(idx)
            
            if interpolated_indices:
                plt.plot(interpolated_indices, df_interp.loc[interpolated_indices, parameter], 
                        'rx', markersize=8, label=f'{method_name} Predicted Points')
            
            # Allow Lagrange and Newton to show the full Runge effect (no y-axis limits)
            plt.title(f'{method_name} Interpolation Curve for {parameter}')
            plt.xlabel('Index')
            plt.ylabel(parameter)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            # No inset zoom (removed as per request)
            
            plt.tight_layout()
            
            # Save the visualization
            safe_param = sanitize_filename(parameter)
            safe_method = sanitize_filename(method_name)
            plt.savefig(os.path.join(viz_dir, f"{filename_base}_{safe_param}_{safe_method}_curve.png"))
            plt.close()

def create_error_metrics_by_parameter(parameter_results, output_dir, filename_base):
    """
    Create plots showing RMSE and MSE for each parameter across different methods.
    
    Args:
        parameter_results: Dictionary with results for each parameter and method
        output_dir: Directory to save the visualizations
        filename_base: Base filename for the output files
    """
    # Ensure visualization directory exists
    viz_dir = os.path.join(output_dir, 'visualizations')
    ensure_dir_exists(viz_dir)
    
    # Calculate error metrics for each parameter and method
    error_metrics = {}
    
    for parameter, methods in parameter_results.items():
        error_metrics[parameter] = {}
        
        for method_name, data in methods.items():
            if not data['predictions'] or not data['actual_values']:
                continue
                
            predictions = np.array(data['predictions'], dtype=float)
            actual_values = np.array(data['actual_values'], dtype=float)
            
            # Skip if arrays are empty
            if len(predictions) == 0 or len(actual_values) == 0:
                continue
            
            # Check for NaN or infinite values
            valid_mask = ~(np.isnan(predictions) | np.isnan(actual_values) | 
                          np.isinf(predictions) | np.isinf(actual_values))
            
            if not np.any(valid_mask):
                continue
                
            # Filter out invalid values
            valid_predictions = predictions[valid_mask]
            valid_actuals = actual_values[valid_mask]
            
            if len(valid_predictions) == 0:
                continue
            
            # Calculate metrics
            mse = np.mean((valid_predictions - valid_actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(valid_predictions - valid_actuals))
            
            error_metrics[parameter][method_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'Count': len(valid_predictions)
            }
    
    # Create bar plots for each parameter showing the metrics for all methods
    for parameter, methods in error_metrics.items():
        if not methods:  # Skip if no methods
            continue
            
        # Create separate plots for RMSE and MSE
        for metric_name in ['RMSE', 'MSE']:
            plt.figure(figsize=(10, 6))
            
            method_names = list(methods.keys())
            values = [methods[m][metric_name] for m in method_names]
            
            # Determine reasonable y-axis limits (exclude extreme outliers)
            if values:
                sorted_values = sorted(values)
                if len(sorted_values) >= 4:  # If we have enough values to determine quartiles
                    q1, q3 = np.percentile(sorted_values, [25, 75])
                    iqr = q3 - q1
                    upper_bound = q3 + 1.5 * iqr
                    non_outliers = [v for v in values if v <= upper_bound]
                    if non_outliers:
                        max_value = max(non_outliers)
                        y_limit = max_value * 1.2  # Add 20% padding
                    else:
                        y_limit = max(values) * 1.2
                else:
                    y_limit = max(values) * 1.2
                
                # Create bar chart
                bars = plt.bar(method_names, values, alpha=0.7)
                
                # Add value labels on top of each bar
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, 
                            min(value + 0.02 * y_limit, y_limit * 0.95),
                            f'{value:.4f}', 
                            ha='center', va='bottom', 
                            fontweight='bold')
                
                # Set consistent y-axis limit for better comparison
                plt.ylim(0, y_limit)
            
            plt.title(f'{metric_name} by Method for {parameter}')
            plt.xlabel('Interpolation Method')
            plt.ylabel(f'{metric_name} (lower is better)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the plot
            safe_param = sanitize_filename(parameter)
            plt.savefig(os.path.join(viz_dir, f"{filename_base}_{safe_param}_{metric_name.lower()}_comparison.png"))
            plt.close()
    
    # Create a combined heatmap for all parameters and methods
    create_error_heatmap(error_metrics, viz_dir, filename_base)

def create_error_heatmap(error_metrics, viz_dir, filename_base):
    """
    Create heatmaps showing error metrics for all parameters and methods.
    
    Args:
        error_metrics: Dictionary with metrics for each parameter and method
        viz_dir: Directory to save the visualizations
        filename_base: Base filename for the output files
    """
    # Create heatmaps for RMSE and MSE
    for metric_name in ['RMSE', 'MSE']:
        # Extract data for heatmap
        parameters = list(error_metrics.keys())
        if not parameters:
            continue
            
        # Get all methods across all parameters
        all_methods = set()
        for param_metrics in error_metrics.values():
            all_methods.update(param_metrics.keys())
        methods = sorted(all_methods)
        
        if not methods:
            continue
        
        # Create data array for heatmap
        data = np.zeros((len(parameters), len(methods)))
        data[:] = np.nan  # Set to NaN initially
        
        # Fill in data
        for i, param in enumerate(parameters):
            for j, method in enumerate(methods):
                if method in error_metrics[param] and metric_name in error_metrics[param][method]:
                    data[i, j] = error_metrics[param][method][metric_name]
        
        # Skip if all NaN
        if np.all(np.isnan(data)):
            continue
            
        # Replace NaN with zeros for plotting
        data_for_plot = np.where(np.isnan(data), 0, data)
        
        # Create heatmap
        plt.figure(figsize=(12, len(parameters)*0.8 + 2))
        im = plt.imshow(data_for_plot, cmap='YlOrRd')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label(metric_name)
        
        # Configure axes and labels
        plt.yticks(range(len(parameters)), parameters)
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        plt.xlabel('Interpolation Method')
        plt.ylabel('Parameter')
        plt.title(f'{metric_name} for Each Parameter and Method (lower is better)')
        
        # Add text annotations with values
        for i in range(len(parameters)):
            for j in range(len(methods)):
                if not np.isnan(data[i, j]):
                    plt.text(j, i, f'{data[i, j]:.4f}', 
                            ha="center", va="center", color="black" if data_for_plot[i, j] < np.nanmax(data_for_plot)/2 else "white")
                else:
                    plt.text(j, i, "N/A", ha="center", va="center", color="black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{filename_base}_{metric_name.lower()}_heatmap.png"))
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
        
    # Convert to Python list if it's a pandas Index
    if isinstance(indices, pd.Index):
        indices = indices.tolist()
        
    sorted_indices = sorted(indices)
    groups = []
    
    if not sorted_indices:
        return []
        
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

def create_summary_statistics(results, parameter_results, output_dir, filename_base):
    """
    Create and save summary statistics for all interpolation methods.
    
    Args:
        results: Dictionary with overall results for each method
        parameter_results: Dictionary with results for each parameter and method
        output_dir: Directory to save the summary
        filename_base: Base filename for the output files
    """
    # Calculate overall error metrics for each method
    summary = []
    
    for method_name, method_results in results.items():
        predictions = method_results['predictions']
        actual_values = method_results['actual_values']
        
        # Skip if no predictions were made
        if not predictions:
            continue
        
        # Convert to numpy arrays
        predictions = np.array(predictions, dtype=float)
        actual_values = np.array(actual_values, dtype=float)
        
        # Filter out NaN or Inf values
        valid_mask = ~(np.isnan(predictions) | np.isnan(actual_values) | 
                      np.isinf(predictions) | np.isinf(actual_values))
        
        if not np.any(valid_mask):
            continue
            
        valid_preds = predictions[valid_mask]
        valid_actuals = actual_values[valid_mask]
        
        if len(valid_preds) == 0:
            continue
        
        # Calculate error metrics
        mse = np.mean((valid_actuals - valid_preds)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(valid_actuals - valid_preds))
        
        # Handle potential division by zero in MAPE calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((valid_actuals - valid_preds) / valid_actuals)) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = np.nan  # Set to NaN if calculation fails
        
        summary.append({
            'Method': method_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape if not np.isnan(mape) else "N/A",
            'Number of Points': len(valid_preds)
        })
    
    # Convert to DataFrame and save as CSV
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, f"{filename_base}_error_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved error metrics to {summary_path}")
    
    # Create parameter-specific error metrics CSV
    param_summary = []
    
    for param, methods in parameter_results.items():
        for method_name, data in methods.items():
            if not data['predictions']:
                continue
                
            predictions = np.array(data['predictions'], dtype=float)
            actual_values = np.array(data['actual_values'], dtype=float)
            
            # Filter out NaN or Inf values
            valid_mask = ~(np.isnan(predictions) | np.isnan(actual_values) | 
                          np.isinf(predictions) | np.isinf(actual_values))
            
            if not np.any(valid_mask):
                continue
                
            valid_preds = predictions[valid_mask]
            valid_actuals = actual_values[valid_mask]
            
            if len(valid_preds) == 0:
                continue
            
            # Calculate metrics
            mse = np.mean((valid_actuals - valid_preds) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(valid_actuals - valid_preds))
            
            param_summary.append({
                'Parameter': param,
                'Method': method_name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'Number of Points': len(valid_preds)
            })
    
    if param_summary:
        param_df = pd.DataFrame(param_summary)
        param_path = os.path.join(output_dir, f"{filename_base}_parameter_metrics.csv")
        param_df.to_csv(param_path, index=False)
        print(f"Saved parameter-specific metrics to {param_path}")
    
    # Create a comparison visualization
    if summary:
        plt.figure(figsize=(10, 6))
        methods = [row['Method'] for row in summary]
        
        # Extract metrics that are numeric
        metrics = ['MSE', 'RMSE', 'MAE']  # Exclude MAPE as it might be N/A
        values_by_metric = {metric: [row[metric] for row in summary] for metric in metrics}
        
        # Set consistent y-axis limit based on reasonable values (exclude extreme outliers)
        y_limits = []
        for metric in metrics:
            values = values_by_metric[metric]
            sorted_values = sorted(values)
            if len(sorted_values) >= 4:  # If we have enough values to determine quartiles
                q1, q3 = np.percentile(sorted_values, [25, 75])
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                non_outliers = [v for v in values if v <= upper_bound]
                if non_outliers:
                    y_limits.append(max(non_outliers) * 1.2)
                else:
                    y_limits.append(max(values) * 1.2)
            else:
                y_limits.append(max(values) * 1.2)
        
        y_limit = max(y_limits) if y_limits else None
        
        x = np.arange(len(methods))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = values_by_metric[metric]
            bars = plt.bar(x + i*width - width*1.5/2, values, width, label=metric)
            
            # Add value labels on top of each bar
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, 
                        min(value + 0.02 * y_limit, y_limit * 0.95) if y_limit else value * 1.1,
                        f'{value:.4f}', 
                        ha='center', va='bottom', 
                        fontweight='bold', rotation=90 if value > y_limit/2 else 0)
        
        if y_limit:
            plt.ylim(0, y_limit)
            
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
