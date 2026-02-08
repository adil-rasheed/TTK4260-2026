"""
Metrics Utilities
=================
Performance metrics for regression and classification.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)


# ===== REGRESSION METRICS =====

def compute_regression_metrics(y_true, y_pred):
    """
    Compute all regression metrics.
    
    Parameters:
    -----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
    
    Returns:
    --------
    metrics : dict
        Dictionary of metric values
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
        'Max Error': np.max(np.abs(y_true - y_pred)),
        'Mean Residual': np.mean(y_true - y_pred)
    }
    
    return metrics


# ===== CLASSIFICATION METRICS =====

def compute_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute all classification metrics.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    y_pred_proba : ndarray (optional)
        Predicted probabilities for ROC/AUC
    
    Returns:
    --------
    metrics : dict
        Dictionary of metric values
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix with labels.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    
    Returns:
    --------
    cm : ndarray
        Confusion matrix
    labels : dict
        Dictionary with TN, FP, FN, TP
    """
    cm = confusion_matrix(y_true, y_pred)
    
    labels = {
        'TN': cm[0, 0],
        'FP': cm[0, 1],
        'FN': cm[1, 0],
        'TP': cm[1, 1]
    }
    
    return cm, labels


def compute_roc_curve(y_true, y_pred_proba):
    """
    Compute ROC curve data.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred_proba : ndarray
        Predicted probabilities
    
    Returns:
    --------
    fpr : ndarray
        False positive rates
    tpr : ndarray
        True positive rates
    thresholds : ndarray
        Thresholds
    auc : float
        Area under ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    return fpr, tpr, thresholds, auc


def compute_pr_curve(y_true, y_pred_proba):
    """
    Compute Precision-Recall curve data.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred_proba : ndarray
        Predicted probabilities
    
    Returns:
    --------
    precision : ndarray
        Precision values
    recall : ndarray
        Recall values
    thresholds : ndarray
        Thresholds
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    return precision, recall, thresholds


# Quick test
if __name__ == "__main__":
    # Test regression metrics
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    
    reg_metrics = compute_regression_metrics(y_true, y_pred)
    print("Regression Metrics:")
    for metric, value in reg_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test classification metrics
    y_true_clf = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred_clf = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    
    clf_metrics = compute_classification_metrics(y_true_clf, y_pred_clf)
    print("\nClassification Metrics:")
    for metric, value in clf_metrics.items():
        print(f"  {metric}: {value:.4f}")
