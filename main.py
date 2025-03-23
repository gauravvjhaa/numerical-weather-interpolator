"""
Main script for weather data interpolation project.
This script orchestrates the entire workflow for all datasets.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_handler import load_data, alter_dataset
from interpolation import (lagrange_interpolation, newton_interpolation, 
                          linear_spline, cubic_spline, polynomial_interpolation)
from evaluation import calculate_errors, generate_error_report

# List of interpolation methods to evaluate
METHODS = {
    'Lagrange': lagrange_interpolation,
    'Newton': newton_interpolation,
    'Linear Spline': linear_spline,
    'Cubic Spline': cubic_spline,
    'Polynomial': polynomial_interpolation
}

# Let's check the first dataset to see what parameters are available
def get_common_parameters():
    """Find common numerical columns across all datasets."""
    data_dir = 'datasets'
    all_datasets = []
    
    # Get list of all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the datasets directory!")
        return []
    
    # Load each dataset and get column names
    for filename in csv_files:
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_datasets.append(set(numeric_cols))
    
    # Find columns common to all datasets
    if all_datasets:
        common_cols = all_datasets[0]
        for cols in all_datasets[1:]:
            common_cols = common_cols.intersection(cols)
        
        # Convert to list and sort
        common_cols = sorted(list(common_cols))
        
        print(f"Found {len(common_cols)} common numerical parameters across all datasets:")
        for col in common_cols:
            print(f"  - {col}")
        
        return common_cols
    else:
        return []

def process_dataset(file_path, parameters, alteration_percentage=10):
    """Process a single dataset with all interpolation methods."""
    print(f"\nProcessing file: {file_path}")
    
    # Load the dataset
    df = load_data(file_path)
    
    # Results dictionary to store errors for each method and parameter
    results = {}
    
    for param in parameters:
        if param not in df.columns:
            print(f"Parameter {param} not found in dataset, skipping...")
            continue
            
        # Skip parameters with missing or non-numeric values
        if not pd.api.types.is_numeric_dtype(df[param]) or df[param].isna().any():
            print(f"Parameter {param} contains non-numeric or missing values, skipping...")
            continue
            
        print(f"\nAnalyzing parameter: {param}")
        results[param] = {}
        
        # Alter dataset (remove some values)
        df_altered, removed_indices, true_values = alter_dataset(
            df, param, percentage=alteration_percentage
        )
        
        # Apply each interpolation method
        for method_name, method_func in METHODS.items():
            print(f"  Applying {method_name} interpolation...")
            
            # Generate predictions using current method
            try:
                predictions = method_func(df_altered, param, removed_indices)
                
                # Calculate errors
                errors = calculate_errors(true_values, predictions)
                results[param][method_name] = errors
                
                print(f"    RMSE: {errors['rmse']:.4f}")
            except Exception as e:
                print(f"    Error with {method_name} interpolation: {e}")
                results[param][method_name] = {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'max_error': np.nan}
    
    return results

def main():
    """Main function to process all datasets."""
    # Find common parameters across all datasets
    common_parameters = get_common_parameters()
    
    if not common_parameters:
        print("No common numerical parameters found across datasets. Exiting.")
        return
    
    # Ask user to select parameters or use defaults
    print("\nPlease select which parameters to analyze:")
    for i, param in enumerate(common_parameters):
        print(f"{i+1}. {param}")
    
    # For now, let's use all common parameters
    selected_parameters = common_parameters
    print(f"\nAnalyzing these parameters: {', '.join(selected_parameters)}")
    
    data_dir = 'datasets'
    results_all = {}
    
    # Process all data files
    for i, filename in enumerate(sorted(os.listdir(data_dir))):
        if not filename.endswith('.csv'):  # Assuming CSV format
            continue
            
        file_path = os.path.join(data_dir, filename)
        results_all[f"dataset_{i+1}"] = process_dataset(file_path, selected_parameters)
    
    # Generate comprehensive error report
    generate_error_report(results_all)
    
    print("\nAnalysis complete! Check the 'results' directory for reports and visualizations.")

if __name__ == "__main__":
    main()
