"""
Module implementing various interpolation methods for weather data.
Optimized for inference on datasets with actual missing values.
"""
import numpy as np
from scipy import interpolate
import pandas as pd

def lagrange_interpolation(df, parameter, indices_to_predict):
    """
    Implement Lagrange interpolation to predict missing values.
    
    Args:
        df: DataFrame with missing values
        parameter: Column name of the parameter to interpolate
        indices_to_predict: Indices where predictions are needed
        
    Returns:
        numpy array of predicted values
    """
    # Get non-NaN indices and values
    valid_indices = df.index[~df[parameter].isna()].tolist()
    valid_values = df.loc[valid_indices, parameter].values
    
    # If no valid data points, we can't interpolate
    if len(valid_indices) < 2:
        raise ValueError(f"Not enough valid points to perform Lagrange interpolation for {parameter}")
    
    # Create time or position axis (using index as x-values)
    x_valid = np.array(valid_indices)
    y_valid = valid_values
    
    # Function to calculate Lagrange polynomial
    def lagrange_basis(x, i, x_points):
        """Calculate the i-th Lagrange basis polynomial at x."""
        n = len(x_points)
        result = 1.0
        
        for j in range(n):
            if j != i:
                result *= (x - x_points[j]) / (x_points[i] - x_points[j])
                
        return result
    
    # Function to evaluate Lagrange polynomial at a point
    def lagrange_polynomial(x, x_points, y_points):
        """Evaluate the Lagrange polynomial at x."""
        n = len(x_points)
        result = 0.0
        
        for i in range(n):
            result += y_points[i] * lagrange_basis(x, i, x_points)
            
        return result
    
    # Predict values at missing indices
    predictions = []
    for idx in indices_to_predict:
        # Using only a subset of points for efficiency and numerical stability
        # Find nearest 10 points to the prediction point
        distances = np.abs(x_valid - idx)
        nearest_indices = np.argsort(distances)[:min(10, len(distances))]
        
        x_subset = x_valid[nearest_indices]
        y_subset = y_valid[nearest_indices]
        
        # Calculate prediction using Lagrange polynomial
        pred = lagrange_polynomial(idx, x_subset, y_subset)
        predictions.append(pred)
    
    return np.array(predictions)

def newton_interpolation(df, parameter, indices_to_predict):
    """
    Implement Newton's divided difference interpolation to predict missing values.
    
    Args:
        df: DataFrame with missing values
        parameter: Column name of the parameter to interpolate
        indices_to_predict: Indices where predictions are needed
        
    Returns:
        numpy array of predicted values
    """
    # Get non-NaN indices and values
    valid_indices = df.index[~df[parameter].isna()].tolist()
    valid_values = df.loc[valid_indices, parameter].values
    
    # If no valid data points, we can't interpolate
    if len(valid_indices) < 2:
        raise ValueError(f"Not enough valid points to perform Newton interpolation for {parameter}")
    
    # Create time or position axis (using index as x-values)
    x_valid = np.array(valid_indices)
    y_valid = valid_values
    
    # Function to calculate divided differences
    def divided_diff(x, y):
        """Calculate the divided differences table."""
        n = len(y)
        coef = np.zeros([n, n])
        
        # First column is y
        coef[:,0] = y
        
        for j in range(1, n):
            for i in range(n-j):
                coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
                
        return coef
    
    # Function to evaluate Newton's polynomial at a point
    def newton_polynomial(coef, x_data, x):
        """Evaluate the Newton polynomial at x."""
        n = len(x_data) - 1
        p = coef[0][0]
        
        for i in range(1, n+1):
            term = coef[0][i]
            for j in range(i):
                term *= (x - x_data[j])
            p += term
            
        return p
    
    # Predict values at missing indices
    predictions = []
    for idx in indices_to_predict:
        # Using only a subset of points for efficiency and numerical stability
        # Find nearest 10 points to the prediction point
        distances = np.abs(x_valid - idx)
        nearest_indices = np.argsort(distances)[:min(10, len(distances))]
        
        x_subset = x_valid[nearest_indices]
        y_subset = y_valid[nearest_indices]
        
        # Calculate divided differences
        coef = divided_diff(x_subset, y_subset)
        
        # Calculate prediction using Newton's polynomial
        pred = newton_polynomial(coef, x_subset, idx)
        predictions.append(pred)
    
    return np.array(predictions)

def linear_spline(df, parameter, indices_to_predict):
    """
    Implement linear spline interpolation to predict missing values.
    
    Args:
        df: DataFrame with missing values
        parameter: Column name of the parameter to interpolate
        indices_to_predict: Indices where predictions are needed
        
    Returns:
        numpy array of predicted values
    """
    # Get non-NaN indices and values
    valid_indices = df.index[~df[parameter].isna()].tolist()
    valid_values = df.loc[valid_indices, parameter].values
    
    # If no valid data points, we can't interpolate
    if len(valid_indices) < 2:
        raise ValueError(f"Not enough valid points to perform linear spline interpolation for {parameter}")
    
    # Create time or position axis (using index as x-values)
    x_valid = np.array(valid_indices)
    y_valid = valid_values
    
    # Create linear spline interpolation function
    f = interpolate.interp1d(x_valid, y_valid, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    
    # Predict values at missing indices
    predictions = f(indices_to_predict)
    
    return predictions

def cubic_spline(df, parameter, indices_to_predict):
    """
    Implement cubic spline interpolation to predict missing values.
    
    Args:
        df: DataFrame with missing values
        parameter: Column name of the parameter to interpolate
        indices_to_predict: Indices where predictions are needed
        
    Returns:
        numpy array of predicted values
    """
    # Get non-NaN indices and values
    valid_indices = df.index[~df[parameter].isna()].tolist()
    valid_values = df.loc[valid_indices, parameter].values
    
    # If no valid data points, we can't interpolate
    if len(valid_indices) < 4:  # Cubic spline needs at least 4 points for stability
        # Fall back to linear spline if we don't have enough points
        if len(valid_indices) >= 2:
            print(f"Warning: Not enough points for cubic spline for {parameter}, falling back to linear")
            return linear_spline(df, parameter, indices_to_predict)
        else:
            raise ValueError(f"Not enough valid points to perform cubic spline interpolation for {parameter}")
    
    # Create time or position axis (using index as x-values)
    x_valid = np.array(valid_indices)
    y_valid = valid_values
    
    # Create cubic spline interpolation function
    try:
        f = interpolate.interp1d(x_valid, y_valid, kind='cubic', 
                                bounds_error=False, fill_value='extrapolate')
        
        # Predict values at missing indices
        predictions = f(indices_to_predict)
        
    except ValueError:
        # If cubic spline fails (e.g., duplicate x values), fall back to linear
        print(f"Warning: Cubic spline failed for {parameter}, falling back to linear")
        return linear_spline(df, parameter, indices_to_predict)
    
    return predictions

def polynomial_interpolation(df, parameter, indices_to_predict, degree=3):
    """
    Implement polynomial interpolation to predict missing values.
    
    Args:
        df: DataFrame with missing values
        parameter: Column name of the parameter to interpolate
        indices_to_predict: Indices where predictions are needed
        degree: Degree of the polynomial (default: 3)
        
    Returns:
        numpy array of predicted values
    """
    # Get non-NaN indices and values
    valid_indices = df.index[~df[parameter].isna()].tolist()
    valid_values = df.loc[valid_indices, parameter].values
    
    # If no valid data points, we can't interpolate
    if len(valid_indices) <= degree:
        # Fall back to lower degree if possible
        if len(valid_indices) >= 2:
            fallback_degree = len(valid_indices) - 1
            print(f"Warning: Not enough points for degree {degree} polynomial for {parameter}, "
                  f"falling back to degree {fallback_degree}")
            return polynomial_interpolation(df, parameter, indices_to_predict, fallback_degree)
        else:
            raise ValueError(f"Not enough valid points to perform polynomial interpolation for {parameter}")
    
    # Create time or position axis (using index as x-values)
    x_valid = np.array(valid_indices)
    y_valid = valid_values
    
    # Fit polynomial
    coeffs = np.polyfit(x_valid, y_valid, degree)
    p = np.poly1d(coeffs)
    
    # Predict values at missing indices
    predictions = p(indices_to_predict)
    
    return predictions
