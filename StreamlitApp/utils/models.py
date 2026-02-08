"""
Model Implementation Utilities
===============================
ML model implementations and wrappers.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score


class PCRFromScratch:
    """
    Principal Component Regression implemented from scratch.
    """
    
    def __init__(self, n_components=2):
        """
        Parameters:
        -----------
        n_components : int
            Number of principal components to use
        """
        self.n_components = n_components
        self.pca = None
        self.scaler_X = None
        self.scaler_y = None
        self.beta_scores = None
        self.theta_original = None
        
    def fit(self, X, y):
        """
        Fit PCR model.
        
        Parameters:
        -----------
        X : ndarray, shape (n, p)
            Training predictors
        y : ndarray, shape (n,)
            Training response
        
        Returns:
        --------
        self
        """
        # Step 1: Standardize
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_std = self.scaler_X.fit_transform(X)
        y_std = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Step 2: PCA
        self.pca = PCA(n_components=self.n_components)
        T = self.pca.fit_transform(X_std)  # Scores
        
        # Step 3: Regression on scores
        self.beta_scores = np.linalg.lstsq(T, y_std, rcond=None)[0]
        
        # Step 4: Back-project to original space
        V = self.pca.components_.T  # Loadings
        self.theta_original = V @ self.beta_scores
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : ndarray
            Test predictors
        
        Returns:
        --------
        y_pred : ndarray
            Predictions
        """
        X_std = self.scaler_X.transform(X)
        T = self.pca.transform(X_std)
        y_pred_std = T @ self.beta_scores
        y_pred = self.scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def score(self, X, y):
        """
        Calculate R² score.
        
        Parameters:
        -----------
        X : ndarray
            Test predictors
        y : ndarray
            True response
        
        Returns:
        --------
        r2 : float
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2


def pcr_cross_validation(X, y, max_components=None, n_folds=5, random_state=42):
    """
    Perform cross-validation for PCR with proper handling of PCA in each fold.
    
    Parameters:
    -----------
    X : ndarray
        Predictor matrix
    y : ndarray
        Response vector
    max_components : int
        Maximum number of components to try (default: min(n_samples, n_features))
    n_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed
    
    Returns:
    --------
    results : dict
        Dictionary with keys 'n_components', 'mean_rmse', 'std_rmse'
    """
    if max_components is None:
        max_components = min(X.shape[0], X.shape[1])
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    rmse_by_k = {}
    
    for k in range(1, max_components + 1):
        fold_rmse = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit PCR on training fold
            pcr = PCRFromScratch(n_components=k)
            pcr.fit(X_train, y_train)
            
            # Predict on validation fold
            y_pred = pcr.predict(X_val)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            fold_rmse.append(rmse)
        
        rmse_by_k[k] = {
            'mean': np.mean(fold_rmse),
            'std': np.std(fold_rmse)
        }
    
    # Format results
    n_components = list(range(1, max_components + 1))
    mean_rmse = [rmse_by_k[k]['mean'] for k in n_components]
    std_rmse = [rmse_by_k[k]['std'] for k in n_components]
    
    results = {
        'n_components': n_components,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse
    }
    
    return results


def bootstrap_stability_analysis(X, y, n_bootstrap=100, model_type='ols', **model_kwargs):
    """
    Perform bootstrap analysis to assess coefficient stability.
    
    Parameters:
    -----------
    X : ndarray
        Predictor matrix
    y : ndarray
        Response vector
    n_bootstrap : int
        Number of bootstrap samples
    model_type : str
        'ols', 'ridge', 'lasso', or 'pcr'
    model_kwargs : dict
        Additional arguments for the model
    
    Returns:
    --------
    bootstrap_coeffs : ndarray
        Bootstrap coefficient estimates
    """
    n_samples = X.shape[0]
    bootstrap_coeffs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X[idx], y[idx]
        
        # Fit model
        if model_type == 'ols':
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            coeffs = model.coef_
        
        elif model_type == 'ridge':
            alpha = model_kwargs.get('alpha', 1.0)
            model = Ridge(alpha=alpha)
            model.fit(X_boot, y_boot)
            coeffs = model.coef_
        
        elif model_type == 'lasso':
            alpha = model_kwargs.get('alpha', 1.0)
            model = Lasso(alpha=alpha)
            model.fit(X_boot, y_boot)
            coeffs = model.coef_
        
        elif model_type == 'pcr':
            n_components = model_kwargs.get('n_components', 1)
            model = PCRFromScratch(n_components=n_components)
            model.fit(X_boot, y_boot)
            coeffs = model.theta_original
        
        bootstrap_coeffs.append(coeffs)
    
    return np.array(bootstrap_coeffs)


def compare_regularization_methods(X, y, alphas=None):
    """
    Compare OLS, Ridge, and Lasso on given data.
    
    Parameters:
    -----------
    X : ndarray
        Predictor matrix
    y : ndarray
        Response vector
    alphas : list
        List of regularization parameters to try
    
    Returns:
    --------
    results : dict
        Comparison results
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0]
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = {}
    
    # OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    results['OLS'] = {
        'train_r2': ols.score(X_train, y_train),
        'test_r2': ols.score(X_test, y_test),
        'coeffs': ols.coef_,
        'l2_norm': np.linalg.norm(ols.coef_)
    }
    
    # Ridge
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        results[f'Ridge(α={alpha})'] = {
            'train_r2': ridge.score(X_train, y_train),
            'test_r2': ridge.score(X_test, y_test),
            'coeffs': ridge.coef_,
            'l2_norm': np.linalg.norm(ridge.coef_)
        }
    
    # Lasso
    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        results[f'Lasso(α={alpha})'] = {
            'train_r2': lasso.score(X_train, y_train),
            'test_r2': lasso.score(X_test, y_test),
            'coeffs': lasso.coef_,
            'l2_norm': np.linalg.norm(lasso.coef_),
            'n_nonzero': np.sum(lasso.coef_ != 0)
        }
    
    return results


# Quick test
if __name__ == "__main__":
    # Test PCR
    from data_generator import generate_correlated_data
    
    X, y, beta_true = generate_correlated_data(n=100, p=5, correlation=0.95)
    
    pcr = PCRFromScratch(n_components=2)
    pcr.fit(X, y)
    
    y_pred = pcr.predict(X)
    r2 = pcr.score(X, y)
    
    print(f"PCR Test: R² = {r2:.4f}")
    print(f"True coefficients: {beta_true}")
    print(f"PCR coefficients: {pcr.theta_original}")
