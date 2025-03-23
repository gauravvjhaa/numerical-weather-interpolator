"""
Module for evaluating the performance of interpolation methods with enhanced 
visualization and detailed analysis capabilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

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
        'max_error': max_error,
        'individual_errors': absolute_errors  # Store individual errors for detailed analysis
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

def generate_error_report(results_all, output_dir='results'):
    """
    Generate comprehensive error reports and visualizations with improved clarity.
    
    Args:
        results_all: Dictionary containing results for all datasets
        output_dir: Directory to save the reports
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Create subdirectories for different types of visualizations
    subdirs = ['per_method', 'per_parameter', 'per_dataset', 'summary', 'error_distribution', 'examples']
    for subdir in subdirs:
        ensure_dir_exists(os.path.join(output_dir, subdir))
    
    # Extract method names and parameter names
    methods = []
    parameters = []
    
    for dataset_results in results_all.values():
        for param, param_results in dataset_results.items():
            if param not in parameters:
                parameters.append(param)
            for method in param_results.keys():
                if method not in methods:
                    methods.append(method)
    
    # 1. Generate per-method analysis
    analyze_per_method(results_all, methods, parameters, output_dir)
    
    # 2. Generate per-parameter analysis
    analyze_per_parameter(results_all, methods, parameters, output_dir)
    
    # 3. Generate per-dataset analysis
    analyze_per_dataset(results_all, methods, parameters, output_dir)
    
    # 4. Generate overall summary
    generate_summary(results_all, methods, parameters, output_dir)
    
    # 5. Generate method ranking
    generate_method_ranking(results_all, output_dir)
    
    # 6. Generate error distribution analysis
    analyze_error_distributions(results_all, output_dir)
    
    print(f"\nDetailed analysis complete. Results saved to the '{output_dir}' directory.")

def analyze_per_method(results_all, methods, parameters, output_dir):
    """Generate analysis comparing each method across all datasets and parameters."""
    print("\nGenerating per-method analysis...")
    
    # Ensure directory exists
    method_dir = os.path.join(output_dir, 'per_method')
    ensure_dir_exists(method_dir)
    
    # For each method, create a summary of performance across all parameters and datasets
    for method in methods:
        # Create a dataframe to store RMSE values for this method
        rmse_data = []
        
        for dataset_name, dataset_results in results_all.items():
            for param, param_results in dataset_results.items():
                if method in param_results:
                    rmse_data.append({
                        'Dataset': dataset_name,
                        'Parameter': param,
                        'RMSE': param_results[method]['rmse']
                    })
        
        if not rmse_data:
            continue
            
        df_method = pd.DataFrame(rmse_data)
        
        # 1. Single plot for this method's RMSE across parameters
        plt.figure(figsize=(10, 6))
        # Updated to fix deprecation warning
        sns.barplot(x='Parameter', y='RMSE', data=df_method, errorbar=None)
        plt.title(f'RMSE for {method} Interpolation Across Parameters')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Safe filename
        filename = os.path.join(method_dir, f"{sanitize_filename(method)}_rmse_by_parameter.png")
        plt.savefig(filename)
        plt.close()
        
        # 2. Save method summary to CSV
        csv_filename = os.path.join(method_dir, f"{sanitize_filename(method)}_summary.csv")
        df_method.to_csv(csv_filename, index=False)
        
        # 3. Create boxplot of RMSE distribution for this method
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Parameter', y='RMSE', data=df_method)
        plt.title(f'Distribution of RMSE for {method} Interpolation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = os.path.join(method_dir, f"{sanitize_filename(method)}_rmse_distribution.png")
        plt.savefig(filename)
        plt.close()

def analyze_per_parameter(results_all, methods, parameters, output_dir):
    """Generate analysis for each parameter across all methods and datasets."""
    print("\nGenerating per-parameter analysis...")
    
    # Ensure directory exists
    param_dir = os.path.join(output_dir, 'per_parameter')
    ensure_dir_exists(param_dir)
    
    for param in parameters:
        # Create a dataframe to store RMSE values for this parameter
        rmse_data = []
        
        for dataset_name, dataset_results in results_all.items():
            if param in dataset_results:
                for method, method_results in dataset_results[param].items():
                    rmse_data.append({
                        'Dataset': dataset_name,
                        'Method': method,
                        'RMSE': method_results['rmse']
                    })
        
        if not rmse_data:
            continue
            
        df_param = pd.DataFrame(rmse_data)
        
        # Sanitize parameter name for filenames
        safe_param = sanitize_filename(param)
        
        # 1. Single line plot showing average RMSE by method for this parameter
        plt.figure(figsize=(10, 6))
        mean_rmse = df_param.groupby('Method')['RMSE'].mean().reset_index()
        sns.lineplot(x='Method', y='RMSE', data=mean_rmse, marker='o', markersize=10)
        plt.title(f'Average RMSE by Method for {param}')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        filename = os.path.join(param_dir, f"{safe_param}_avg_rmse_by_method.png")
        plt.savefig(filename)
        plt.close()
        
        # 2. Method comparison bar chart (fixed deprecation warning)
        plt.figure(figsize=(10, 6))
        # Correctly using hue instead of palette directly
        ax = sns.barplot(x='Method', y='RMSE', hue='Method', data=mean_rmse, legend=False)
        plt.title(f'Method Comparison for {param} Parameter')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        filename = os.path.join(param_dir, f"{safe_param}_method_comparison.png")
        plt.savefig(filename)
        plt.close()
        
        # 3. Best method analysis for this parameter
        best_method_by_dataset = df_param.loc[df_param.groupby('Dataset')['RMSE'].idxmin()]
        method_counts = best_method_by_dataset['Method'].value_counts()
        
        plt.figure(figsize=(8, 6))
        method_counts.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.tab10.colors)
        plt.title(f'Best Performing Methods for {param}')
        plt.ylabel('')
        plt.tight_layout()
        
        filename = os.path.join(param_dir, f"{safe_param}_best_methods_pie.png")
        plt.savefig(filename)
        plt.close()
        
        # 4. Save parameter summary to CSV
        csv_filename = os.path.join(param_dir, f"{safe_param}_summary.csv")
        df_param.to_csv(csv_filename, index=False)

def analyze_per_dataset(results_all, methods, parameters, output_dir):
    """Generate analysis for each dataset across all methods and parameters."""
    print("\nGenerating per-dataset analysis...")
    
    # Ensure directory exists
    dataset_dir = os.path.join(output_dir, 'per_dataset')
    ensure_dir_exists(dataset_dir)
    
    for dataset_name, dataset_results in results_all.items():
        # Skip empty datasets
        if not dataset_results:
            continue
            
        # Create heatmap data for this dataset
        heatmap_data = []
        
        for param, param_results in dataset_results.items():
            for method, method_results in param_results.items():
                heatmap_data.append({
                    'Parameter': param,
                    'Method': method,
                    'RMSE': method_results['rmse']
                })
        
        if not heatmap_data:
            continue
            
        df_heatmap = pd.DataFrame(heatmap_data)
        
        # Sanitize dataset name
        safe_dataset = sanitize_filename(dataset_name)
        
        # Pivot the data for the heatmap
        pivot_data = df_heatmap.pivot(index='Parameter', columns='Method', values='RMSE')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
        plt.title(f'RMSE Heatmap for {dataset_name}')
        plt.tight_layout()
        
        filename = os.path.join(dataset_dir, f"{safe_dataset}_heatmap.png")
        plt.savefig(filename)
        plt.close()
        
        # Create method ranking for this dataset
        for param in pivot_data.index:
            # Get the method ranking for this parameter
            method_ranking = pivot_data.loc[param].sort_values()
            
            # Sanitize parameter name
            safe_param = sanitize_filename(param)
            
            # Create horizontal bar chart with best method on top
            plt.figure(figsize=(10, 6))
            method_ranking.plot(kind='barh', color=plt.cm.viridis(np.linspace(0, 1, len(method_ranking))))
            plt.title(f'Method Ranking for {param} in {dataset_name}')
            plt.xlabel('RMSE (lower is better)')
            plt.grid(True, axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            filename = os.path.join(dataset_dir, f"{safe_dataset}_{safe_param}_method_ranking.png")
            plt.savefig(filename)
            plt.close()
        
        # Save dataset summary to CSV
        csv_filename = os.path.join(dataset_dir, f"{safe_dataset}_summary.csv")
        df_heatmap.to_csv(csv_filename, index=False)

def generate_summary(results_all, methods, parameters, output_dir):
    """Generate overall summary visualizations and reports."""
    print("\nGenerating overall summary...")
    
    # Ensure directory exists
    summary_dir = os.path.join(output_dir, 'summary')
    ensure_dir_exists(summary_dir)
    
    # Create dataframe with all results
    all_results = []
    
    for dataset_name, dataset_results in results_all.items():
        for param, param_results in dataset_results.items():
            for method, method_results in param_results.items():
                all_results.append({
                    'Dataset': dataset_name,
                    'Parameter': param,
                    'Method': method,
                    'RMSE': method_results['rmse'],
                    'MAE': method_results['mae'],
                    'R2': method_results['r2'],
                    'MAPE': method_results['mape']
                })
    
    if not all_results:
        print("No results to summarize!")
        return
        
    df_all = pd.DataFrame(all_results)
    
    # 1. Average RMSE by method (one clean line with points)
    plt.figure(figsize=(10, 6))
    mean_rmse_by_method = df_all.groupby('Method')['RMSE'].mean().reset_index()
    mean_rmse_by_method = mean_rmse_by_method.sort_values('RMSE')
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_rmse_by_method['Method'], mean_rmse_by_method['RMSE'], 
             marker='o', linestyle='-', linewidth=2, markersize=10)
    
    plt.title('Average RMSE by Method Across All Datasets and Parameters')
    plt.ylabel('RMSE (lower is better)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = os.path.join(summary_dir, "average_rmse_by_method.png")
    plt.savefig(filename)
    plt.close()
    
    # 2. One metric per visualization - MAE comparison (updated with hue)
    plt.figure(figsize=(10, 6))
    mean_mae_by_method = df_all.groupby('Method')['MAE'].mean().reset_index()
    mean_mae_by_method = mean_mae_by_method.sort_values('MAE')
    
    sns.barplot(x='Method', y='MAE', hue='Method', data=mean_mae_by_method, legend=False)
    plt.title('Average MAE by Method')
    plt.ylabel('MAE (lower is better)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = os.path.join(summary_dir, "average_mae_by_method.png")
    plt.savefig(filename)
    plt.close()
    
    # 3. One metric per visualization - R² comparison (updated with hue)
    plt.figure(figsize=(10, 6))
    mean_r2_by_method = df_all.groupby('Method')['R2'].mean().reset_index()
    mean_r2_by_method = mean_r2_by_method.sort_values('R2', ascending=False)
    
    sns.barplot(x='Method', y='R2', hue='Method', data=mean_r2_by_method, legend=False)
    plt.title('Average R² by Method')
    plt.ylabel('R² (higher is better)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = os.path.join(summary_dir, "average_r2_by_method.png")
    plt.savefig(filename)
    plt.close()
    
    # 4. Summary table of all metrics
    summary_table = pd.DataFrame({
        'Method': methods,
        'Avg RMSE': [df_all[df_all['Method'] == method]['RMSE'].mean() for method in methods],
        'Avg MAE': [df_all[df_all['Method'] == method]['MAE'].mean() for method in methods],
        'Avg R²': [df_all[df_all['Method'] == method]['R2'].mean() for method in methods],
        'Avg MAPE (%)': [df_all[df_all['Method'] == method]['MAPE'].mean() for method in methods]
    })
    
    summary_table = summary_table.sort_values('Avg RMSE')
    csv_filename = os.path.join(summary_dir, "method_performance_summary.csv")
    summary_table.to_csv(csv_filename, index=False)
    
    # 5. Method performance by parameter type (separate visualization)
    plt.figure(figsize=(12, 8))
    parameter_performance = df_all.groupby(['Parameter', 'Method'])['RMSE'].mean().reset_index()
    
    for i, param in enumerate(parameters):
        plt.figure(figsize=(10, 6))
        param_data = parameter_performance[parameter_performance['Parameter'] == param]
        param_data = param_data.sort_values('RMSE')
        
        safe_param = sanitize_filename(param)
        
        # Updated to use hue instead of direct color assignment
        sns.barplot(x='Method', y='RMSE', hue='Method', data=param_data, legend=False)
        plt.title(f'Method Performance for {param}')
        plt.ylabel('RMSE (lower is better)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = os.path.join(summary_dir, f"method_performance_{safe_param}.png")
        plt.savefig(filename)
        plt.close()

def generate_method_ranking(results_all, output_dir):
    """Generate comprehensive method ranking across all datasets and parameters."""
    print("\nGenerating method ranking...")
    
    # Ensure directory exists
    summary_dir = os.path.join(output_dir, 'summary')
    ensure_dir_exists(summary_dir)
    
    # Create a dataframe to store the best method for each dataset and parameter
    ranking_data = []
    
    for dataset_name, dataset_results in results_all.items():
        for param, param_results in dataset_results.items():
            # Find method with lowest RMSE
            if not param_results:
                continue
                
            best_method = min(param_results.items(), key=lambda x: x[1]['rmse'])
            ranking_data.append({
                'Dataset': dataset_name,
                'Parameter': param,
                'Best Method': best_method[0],
                'RMSE': best_method[1]['rmse']
            })
    
    if not ranking_data:
        print("No data for method ranking!")
        return
        
    ranking_df = pd.DataFrame(ranking_data)
    
    # 1. Count of best method occurrences (one simple bar chart)
    method_counts = ranking_df['Best Method'].value_counts().reset_index()
    method_counts.columns = ['Method', 'Count']
    
    plt.figure(figsize=(10, 6))
    # Updated to use hue instead of direct color assignment
    sns.barplot(x='Method', y='Count', hue='Method', data=method_counts, legend=False)
    plt.title('Number of Times Each Method Performed Best')
    plt.ylabel('Count')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = os.path.join(summary_dir, "best_method_counts.png")
    plt.savefig(filename)
    plt.close()
    
    # 2. Percentage of best method occurrences (pie chart)
    plt.figure(figsize=(10, 8))
    method_counts['Percentage'] = method_counts['Count'] / method_counts['Count'].sum() * 100
    plt.pie(method_counts['Count'], labels=method_counts['Method'], 
            autopct='%1.1f%%', colors=plt.cm.tab10.colors)
    plt.title('Percentage of Times Each Method Performed Best')
    plt.tight_layout()
    
    filename = os.path.join(summary_dir, "best_method_percentage.png")
    plt.savefig(filename)
    plt.close()
    
    # 3. Best method by parameter
    param_best_methods = ranking_df.groupby('Parameter')['Best Method'].value_counts().unstack().fillna(0)
    
    for param in param_best_methods.index:
        plt.figure(figsize=(10, 6))
        param_best_methods.loc[param].plot(kind='bar', 
                                          color=plt.cm.viridis(np.linspace(0, 1, len(param_best_methods.columns))))
        plt.title(f'Best Methods for {param}')
        plt.ylabel('Count')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        safe_param = sanitize_filename(param)
        filename = os.path.join(summary_dir, f"best_methods_for_{safe_param}.png")
        plt.savefig(filename)
        plt.close()
    
    # 4. Save ranking to CSV
    csv_filename = os.path.join(summary_dir, "method_ranking.csv")
    ranking_df.to_csv(csv_filename, index=False)
    
    # 5. Generate a comprehensive report
    report_path = os.path.join(summary_dir, "method_ranking_report.txt")
    with open(report_path, 'w') as f:
        f.write("# INTERPOLATION METHOD RANKING REPORT #\n\n")
        
        f.write("## OVERALL BEST METHODS ##\n")
        for i, (method, count) in enumerate(zip(method_counts['Method'], method_counts['Count'])):
            percentage = method_counts.loc[i, 'Percentage']
            f.write(f"{i+1}. {method}: {count} occurrences ({percentage:.1f}%)\n")
        
        f.write("\n## BEST METHOD BY PARAMETER ##\n")
        for param in param_best_methods.index:
            f.write(f"\n{param}:\n")
            best_for_param = param_best_methods.loc[param].sort_values(ascending=False)
            for method, count in best_for_param.items():
                if count > 0:
                    f.write(f"  - {method}: {int(count)} datasets\n")

def analyze_error_distributions(results_all, output_dir):
    """Analyze the distribution of errors for each method."""
    print("\nAnalyzing error distributions...")
    
    # Create directory for error distribution plots
    error_dir = os.path.join(output_dir, 'error_distribution')
    ensure_dir_exists(error_dir)
    
    # Collect all methods
    all_methods = set()
    for dataset_results in results_all.values():
        for param_results in dataset_results.values():
            all_methods.update(param_results.keys())
    
    # For each method, create a boxplot of error metrics across all datasets and parameters
    for metric in ['rmse', 'mae', 'mape', 'r2']:
        metric_data = []
        
        for dataset_name, dataset_results in results_all.items():
            for param, param_results in dataset_results.items():
                for method, method_results in param_results.items():
                    if metric in method_results:
                        metric_data.append({
                            'Dataset': dataset_name,
                            'Parameter': param,
                            'Method': method,
                            'Value': method_results[metric]
                        })
        
        if not metric_data:
            continue
            
        df_metric = pd.DataFrame(metric_data)
        
        # Create one plot per metric (clean, focused visualization)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Method', y='Value', data=df_metric)
        
        metric_name = metric.upper()
        plt.title(f'{metric_name} Distribution by Method')
        plt.ylabel(metric_name)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = os.path.join(error_dir, f"{metric}_distribution.png")
        plt.savefig(filename)
        plt.close()
        
        # Create individual method comparison for this metric
        for method in all_methods:
            method_data = df_metric[df_metric['Method'] == method]
            
            if method_data.empty:
                continue
                
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Parameter', y='Value', data=method_data)
            plt.title(f'{metric_name} Distribution for {method} Method')
            plt.ylabel(metric_name)
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            safe_method = sanitize_filename(method)
            filename = os.path.join(error_dir, f"{safe_method}_{metric}_by_parameter.png")
            plt.savefig(filename)
            plt.close()

def plot_interpolation_example(df_original, df_altered, parameter, predictions_all, 
                              indices_to_predict, output_dir='results/examples'):
    """
    Plot examples of interpolation results with each method shown separately for clarity.
    
    Args:
        df_original: Original DataFrame with complete data
        df_altered: DataFrame with missing values
        parameter: Column name of the parameter visualized
        predictions_all: Dictionary with predictions from each method
        indices_to_predict: Indices where predictions were made
        output_dir: Directory to save the example plots
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Sanitize parameter name for filenames
    safe_param = sanitize_filename(parameter)
    
    # Get original values for comparison
    true_values = df_original.loc[indices_to_predict, parameter].values
    
    # Plot each method separately with original data for comparison
    for method_name, predictions in predictions_all.items():
        plt.figure(figsize=(12, 6))
        
        # Plot complete original data
        plt.plot(df_original.index, df_original[parameter], 'b-', alpha=0.3, label='Complete Data')
        
        # Plot remaining data points
        valid_indices = df_altered.index[~df_altered[parameter].isna()].tolist()
        plt.plot(valid_indices, df_altered.loc[valid_indices, parameter], 'go', label='Known Data Points')
        
        # Plot true values at prediction points
        plt.plot(indices_to_predict, true_values, 'ko', label='True Values')
        
        # Plot predicted values
        plt.plot(indices_to_predict, predictions, 'rx', label=f'{method_name} Predictions')
        
        # Add a legend, title and labels
        plt.title(f'{method_name} Interpolation for {parameter}')
        plt.xlabel('Index')
        plt.ylabel(parameter)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the plot with sanitized filename
        safe_method = sanitize_filename(method_name)
        filename = os.path.join(output_dir, f"{safe_param}_{safe_method}_example.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"Example plot saved to {filename}")
    
    # Create a method comparison plot for a small section (for clarity)
    # If there are many prediction points, select a small window to show detail
    if len(indices_to_predict) > 10:
        # Take a window of 10 points
        window_size = 10
        window_start = len(indices_to_predict) // 2  # Start at middle
        window_end = min(window_start + window_size, len(indices_to_predict))
        
        window_indices = indices_to_predict[window_start:window_end]
        window_true = true_values[window_start:window_end]
        
        plt.figure(figsize=(12, 8))
        plt.plot(window_indices, window_true, 'ko-', label='True Values')
        
        for method_name, predictions in predictions_all.items():
            window_pred = predictions[window_start:window_end]
            plt.plot(window_indices, window_pred, 'o-', label=f'{method_name}')
        
        plt.title(f'Method Comparison Detail for {parameter}')
        plt.xlabel('Index')
        plt.ylabel(parameter)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        filename = os.path.join(output_dir, f"{safe_param}_method_comparison_detail.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
