"""
Module implementing various interpolation methods for weather data.
Optimized for inference on datasets with actual missing values.

Author: Gaurav Jha
Last updated: 2025-03-27 18:09:23
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
    
    # Predict values at missing indices using all valid points
    predictions = [lagrange_polynomial(idx, x_valid, y_valid) for idx in indices_to_predict]
    
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
    x_valid = np.array(valid_indices, dtype=float)
    y_valid = np.array(valid_values, dtype=float)
    
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
    
    # Calculate divided differences using all valid points
    coef = divided_diff(x_valid, y_valid)
    
    # Predict values at missing indices using all valid points
    predictions = [newton_polynomial(coef, x_valid, idx) for idx in indices_to_predict]
    
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
    x_valid = np.array(valid_indices, dtype=float)
    y_valid = np.array(valid_values, dtype=float)
    
    # Create linear spline interpolation function
    f = interpolate.interp1d(x_valid, y_valid, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    
    # Predict values at missing indices using all valid points
    predictions = f(indices_to_predict)
    
    return predictions

def cubic_spline(df, parameter, indices_to_predict):
    """
    Implement cubic spline interpolation with smoothing to predict missing values.
    
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
    x_valid = np.array(valid_indices, dtype=float)
    y_valid = np.array(valid_values, dtype=float)
    
    try:
        # Create proper cubic spline with smoothness
        # Use UnivariateSpline with small smoothing factor to reduce sharpness
        s = len(x_valid) * 0.01  # Small smoothing factor (0 = exact interpolation)
        f = interpolate.UnivariateSpline(x_valid, y_valid, k=3, s=s)
        
        # Predict values at missing indices
        predictions = f(indices_to_predict)
        
    except Exception as e:
        print(f"Warning: Advanced cubic spline failed for {parameter}: {e}, falling back to basic spline")
        try:
            # Try with regular CubicSpline as fallback
            f = interpolate.CubicSpline(x_valid, y_valid, bc_type='natural')
            predictions = f(indices_to_predict)
        except:
            # If that fails too, fall back to linear
            print(f"Warning: Cubic spline failed for {parameter}, falling back to linear")
            return linear_spline(df, parameter, indices_to_predict)
    
    return predictions