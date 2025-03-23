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

# Parameters to analyze (example - adjust based on your actual data)
PARAMETERS = ['temperature', 'humidity', 'pressure', 'wind_speed']

def process_dataset(file_path, alteration_percentage=10):
    """Process a single dataset with all interpolation methods."""
    print(f"\nProcessing file: {file_path}")
    
    # Load the dataset
    df = load_data(file_path)
    
    # Results dictionary to store errors for each method and parameter
    results = {}
    
    for param in PARAMETERS:
        if param not in df.columns:
            print(f"Parameter {param} not found in dataset, skipping...")
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
            predictions = method_func(df_altered, param, removed_indices)
            
            # Calculate errors
            errors = calculate_errors(true_values, predictions)
            results[param][method_name] = errors
            
            print(f"    RMSE: {errors['rmse']:.4f}")
    
    return results

def main():
    """Main function to process all datasets."""
    data_dir = 'datasets'
    results_all = {}
    
    # Process all data files
    for i, filename in enumerate(sorted(os.listdir(data_dir))):
        if not filename.endswith('.csv'):  # Assuming CSV format
            continue
            
        file_path = os.path.join(data_dir, filename)
        results_all[f"dataset_{i+1}"] = process_dataset(file_path)
    
    # Generate comprehensive error report
    generate_error_report(results_all)
    
    print("\nAnalysis complete! Check the 'results' directory for reports and visualizations.")

if __name__ == "__main__":
    main()
