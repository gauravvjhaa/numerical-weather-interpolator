"""
Module for evaluating the performance of interpolation methods.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_errors(true_values, predicted_values):
    """
    Calculate various error metrics between true and predicted values.
    
    Args:
        true_values: Array of actual values
        predicted_values: Array of predicted values
        
    Returns:
        dict: Dictionary containing error metrics
    """
    # Calculate absolute errors
    absolute_errors = np.abs(true_values - predicted_values)
    
    # Calculate mean absolute error (MAE)
    mae = np.mean(absolute_errors)
    
    # Calculate mean squared error (MSE)
    mse = np.mean((true_values - predicted_values) ** 2)
    
    # Calculate root mean squared error (RMSE)
    rmse = np.sqrt(mse)
    
    # Calculate mean absolute percentage error (MAPE)
    # Avoid division by zero
    non_zero_mask = (true_values != 0)
    if np.any(non_zero_mask):
        mape = 100 * np.mean(absolute_errors[non_zero_mask] / np.abs(true_values[non_zero_mask]))
    else:
        mape = np.nan
    
    # Calculate coefficient of determination (R²)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    ss_res = np.sum((true_values - predicted_values) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    # Calculate maximum absolute error
    max_error = np.max(absolute_errors)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'max_error': max_error
    }

def generate_error_report(results_all, output_dir='results'):
    """
    Generate comprehensive error reports and visualizations.
    
    Args:
        results_all: Dictionary containing results for all datasets
        output_dir: Directory to save the reports
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract method names
    methods = next(iter(next(iter(results_all.values())).values())).keys()
    
    # Create summary DataFrame for each error metric
    for metric in ['rmse', 'mae', 'mape', 'r2']:
        # Initialize DataFrame for current metric
        df_metric = pd.DataFrame(index=results_all.keys())
        
        # Populate DataFrame with values for each method
        for dataset_name, dataset_results in results_all.items():
            # For each parameter in the dataset
            for param, param_results in dataset_results.items():
                # For each method
                for method in methods:
                    if method in param_results:
                        col_name = f"{param}_{method}"
                        df_metric.loc[dataset_name, col_name] = param_results[method][metric]
        
        # Save to CSV
        df_metric.to_csv(f"{output_dir}/summary_{metric}.csv")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        ax = df_metric.mean().plot(kind='bar')
        plt.title(f'Average {metric.upper()} Across All Datasets')
        plt.ylabel(metric.upper())
        plt.xlabel('Parameter and Method')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/average_{metric}.png")
        plt.close()
    
    # Create a comprehensive ranking of methods
    ranking_df = pd.DataFrame(columns=['Dataset', 'Parameter', 'Best Method', 'RMSE'])
    
    row_idx = 0
    for dataset_name, dataset_results in results_all.items():
        for param, param_results in dataset_results.items():
            # Find the method with the lowest RMSE
            best_method = min(param_results.items(), key=lambda x: x[1]['rmse'])
            
            ranking_df.loc[row_idx] = [
                dataset_name,
                param,
                best_method[0],
                best_method[1]['rmse']
            ]
            row_idx += 1
    
    # Save ranking to CSV
    ranking_df.to_csv(f"{output_dir}/method_ranking.csv", index=False)
    
    # Create a summary table of best methods
    method_counts = ranking_df['Best Method'].value_counts()
    plt.figure(figsize=(10, 6))
    method_counts.plot(kind='bar')
    plt.title('Number of Times Each Method Performed Best')
    plt.ylabel('Count')
    plt.xlabel('Method')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/best_method_counts.png")
    plt.close()

def plot_interpolation_results(df_original, df_altered, parameter, predictions, 
                              method_name, indices_to_predict, output_dir='results'):
    """
    Plot the original data, altered data, and interpolation results.
    
    Args:
        df_original: Original DataFrame
        df_altered: DataFrame with missing values
        parameter: Column name of the parameter visualized
        predictions: Predicted values
        method_name: Name of the interpolation method
        indices_to_predict: Indices where predictions were made
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 6))
    
    # Plot original data
    plt.plot(df_original.index, df_original[parameter], 'b-', label='Original Data')
    
    # Plot remaining data points
    valid_indices = df_altered.index[~df_altered[parameter].isna()].tolist()
    plt.plot(valid_indices, df_altered.loc[valid_indices, parameter], 'go', label='Remaining Data')
    
    # Plot predicted values
    plt.plot(indices_to_predict, predictions, 'rx', label='Predicted Values')
    
    # Plot true values at prediction points
    plt.plot(indices_to_predict, df_original.loc[indices_to_predict, parameter], 'kx', label='True Values')
    
    plt.title(f'{method_name} Interpolation for {parameter}')
    plt.xlabel('Index')
    plt.ylabel(parameter)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    filename = f"{output_dir}/{parameter}_{method_name}_interpolation.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Plot saved to {filename}")
