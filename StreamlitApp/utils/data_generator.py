"""
Data Generation Utilities
==========================
Functions for generating synthetic datasets and loading real data.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_correlated_data(n=100, p=2, correlation=0.95, noise_std=0.5, random_state=None):
    """
    Generate dataset with controlled multicollinearity.
    
    Parameters:
    -----------
    n : int
        Number of observations
    p : int
        Number of predictors
    correlation : float
        Target correlation between predictors (0 to 0.99)
    noise_std : float
        Standard deviation of noise in response
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : ndarray, shape (n, p)
        Predictor matrix
    y : ndarray, shape (n,)
        Response vector
    beta_true : ndarray, shape (p,)
        True coefficients
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create correlation matrix
    corr_matrix = np.eye(p)
    for i in range(p):
        for j in range(p):
            if i != j:
                corr_matrix[i, j] = correlation ** abs(i - j)
    
    # Generate correlated predictors
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean, corr_matrix, n)
    
    # True coefficients - decreasing from 2.0 to 0.5
    beta_true = np.linspace(2.0, 0.5, p)
    
    # Generate response with noise
    y = X @ beta_true + noise_std * np.random.normal(0, 1, n)
    
    return X, y, beta_true


def generate_linear_data(n=100, slope=2.5, intercept=1.0, noise_std=1.0, x_range=(0, 10), random_state=None):
    """
    Generate simple linear regression data.
    
    Parameters:
    -----------
    n : int
        Number of samples
    slope : float
        True slope
    intercept : float
        True intercept
    noise_std : float
        Standard deviation of noise
    x_range : tuple
        (min, max) range for X values
    random_state : int
        Random seed
    
    Returns:
    --------
    X : ndarray, shape (n,)
        Predictor values
    y : ndarray, shape (n,)
        Response values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.linspace(x_range[0], x_range[1], n)
    y = slope * X + intercept + noise_std * np.random.randn(n)
    
    return X, y


def generate_rotated_data(angle=45, var_parallel=4.0, var_perp=0.5, n_samples=100, random_state=None):
    """
    Generate 2D data with specified rotation angle and variances.
    Useful for PCA demonstrations.
    
    Parameters:
    -----------
    angle : float
        Rotation angle in degrees
    var_parallel : float
        Variance along principal direction
    var_perp : float
        Variance perpendicular to principal direction
    n_samples : int
        Number of samples
    random_state : int
        Random seed
    
    Returns:
    --------
    data : ndarray, shape (n_samples, 2)
        Generated 2D data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate data in aligned axes
    x_aligned = np.random.normal(0, np.sqrt(var_parallel), n_samples)
    y_aligned = np.random.normal(0, np.sqrt(var_perp), n_samples)
    
    # Rotation matrix
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate data
    data_aligned = np.column_stack([x_aligned, y_aligned])
    data = data_aligned @ rotation_matrix.T
    
    return data


def generate_process_sensor_data(n=100, random_state=None):
    """
    Generate simulated process sensor data with 5 correlated temperature sensors.
    
    Parameters:
    -----------
    n : int
        Number of observations
    random_state : int
        Random seed
    
    Returns:
    --------
    df : DataFrame
        DataFrame with columns: x1_Inlet, x2_Mid, x3_Outlet, x4_Jacket, x5_Ambient, product_strength
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Base temperature (overall level)
    base_temp = np.random.normal(100, 5, n)
    
    # Sensor readings with correlation
    x1_inlet = base_temp + np.random.normal(0, 1, n)
    x2_mid = 0.95 * x1_inlet + np.random.normal(0, 1.5, n)
    x3_outlet = 0.90 * x2_mid + np.random.normal(0, 2, n)
    x4_jacket = base_temp + np.random.normal(0, 1.5, n)
    x5_ambient = 0.85 * base_temp + np.random.normal(0, 2.5, n)
    
    # Product strength depends on overall temperature
    product_strength = 0.5 * base_temp + 0.3 * x1_inlet + 0.2 * x2_mid + np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        'x1_Inlet': x1_inlet,
        'x2_Mid': x2_mid,
        'x3_Outlet': x3_outlet,
        'x4_Jacket': x4_jacket,
        'x5_Ambient': x5_ambient,
        'product_strength': product_strength
    })
    
    return df


def load_citytemp_data():
    """
    Load CityTemp dataset if available.
    
    Returns:
    --------
    df : DataFrame or None
        Temperature data or None if file not found
    """
    data_path = Path(__file__).parent.parent / 'data' / 'CityTemp.xlsx'
    
    try:
        df = pd.read_excel(data_path)
        return df
    except FileNotFoundError:
        return None


def load_macdonald_data():
    """
    Load McDonald's dataset if available.
    
    Returns:
    --------
    df : DataFrame or None
        McDonald's data or None if file not found
    """
    data_path = Path(__file__).parent.parent / 'data' / 'macdonald.csv'
    
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        return None


def generate_classification_data(n=200, n_features=2, n_classes=2, class_sep=1.0, random_state=None):
    """
    Generate synthetic classification dataset.
    
    Parameters:
    -----------
    n : int
        Number of samples
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    class_sep : float
        Separation between classes (larger = easier)
    random_state : int
        Random seed
    
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Class labels
    """
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state
    )
    
    return X, y


# Quick test
if __name__ == "__main__":
    # Test data generation
    X, y, beta = generate_correlated_data(n=50, p=3, correlation=0.9)
    print(f"Generated data: X.shape={X.shape}, y.shape={y.shape}")
    print(f"True coefficients: {beta}")
    print(f"Correlation matrix:\n{np.corrcoef(X.T)}")
