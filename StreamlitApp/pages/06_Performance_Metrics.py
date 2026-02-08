"""
Performance Metrics for ML Models
=================================
Classification and regression metrics with interactive visualizations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import compute_regression_metrics, compute_classification_metrics

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

st.set_page_config(page_title="Performance Metrics", page_icon="📊", layout="wide")

st.title("📊 Performance Metrics")
st.markdown("### Evaluating Machine Learning Models")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Regression Metrics",
    "🎯 Classification Metrics",
    "📉 ROC & PR Curves",
    "🔍 Threshold Tuning"
])

# ===== TAB 1: Regression Metrics =====
with tab1:
    st.markdown("## Regression Performance Metrics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎲 Generate Predictions")
        
        n_samples = st.slider("Number of samples", 20, 200, 100, 10)
        r2_target = st.slider("Target R² score", 0.0, 1.0, 0.8, 0.05)
        
        if st.button("Generate Data", type="primary"):
            np.random.seed(42)
            # Generate true values
            y_true = np.random.randn(n_samples) * 10 + 50
            
            # Generate predictions with controlled R²
            noise_factor = np.sqrt(1 - r2_target) / np.sqrt(r2_target) if r2_target > 0 else 10
            y_pred = y_true + np.random.randn(n_samples) * np.std(y_true) * noise_factor
            
            # Compute metrics
            metrics = compute_regression_metrics(y_true, y_pred)
            
            st.session_state.reg_y_true = y_true
            st.session_state.reg_y_pred = y_pred
            st.session_state.reg_metrics = metrics
            
            st.markdown("### 📊 Metrics")
            st.metric("MAE", f"{metrics['mae']:.3f}")
            st.metric("RMSE", f"{metrics['rmse']:.3f}")
            st.metric("R² Score", f"{metrics['r2']:.3f}")
            st.metric("Adj. R²", f"{metrics['adjusted_r2']:.3f}")
    
    with col2:
        if 'reg_y_true' in st.session_state:
            y_true = st.session_state.reg_y_true
            y_pred = st.session_state.reg_y_pred
            
            # Create prediction plot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Predicted vs Actual", "Residual Plot")
            )
            
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                marker=dict(color='blue', size=6),
                name='Predictions'
            ), row=1, col=1)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Fit'
            ), row=1, col=1)
            
            # Residuals
            residuals = y_true - y_pred
            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(color='green', size=6),
                name='Residuals'
            ), row=1, col=2)
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            fig.update_xaxes(title_text="Actual Values", row=1, col=1)
            fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
            fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
            fig.update_yaxes(title_text="Residuals", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📖 Metric Explanations"):
                st.markdown("""
                **MAE (Mean Absolute Error):** Average absolute difference
                
                $$MAE = \\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|$$
                
                **RMSE (Root Mean Squared Error):** Square root of average squared error
                
                $$RMSE = \\sqrt{\\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2}$$
                
                **R² Score:** Proportion of variance explained
                
                $$R^2 = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$$
                """)

# ===== TAB 2: Classification Metrics =====
with tab2:
    st.markdown("## Classification Performance Metrics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎲 Generate Predictions")
        
        n_samples = st.slider("Number of samples", 50, 500, 200, 10, key='class_n')
        class_balance = st.slider("Class balance (% positive)", 20, 80, 50, 5)
        accuracy_target = st.slider("Target accuracy", 0.5, 1.0, 0.85, 0.05)
        
        if st.button("Generate Data", type="primary", key='class_btn'):
            np.random.seed(42)
            
            # Generate true labels
            n_positive = int(n_samples * class_balance / 100)
            y_true = np.array([1] * n_positive + [0] * (n_samples - n_positive))
            np.random.shuffle(y_true)
            
            # Generate predictions with target accuracy
            y_pred = y_true.copy()
            n_errors = int(n_samples * (1 - accuracy_target))
            error_indices = np.random.choice(n_samples, n_errors, replace=False)
            y_pred[error_indices] = 1 - y_pred[error_indices]
            
            # Generate probability scores
            y_prob = np.zeros(n_samples)
            y_prob[y_true == 1] = np.random.beta(8, 2, sum(y_true == 1))
            y_prob[y_true == 0] = np.random.beta(2, 8, sum(y_true == 0))
            
            st.session_state.class_y_true = y_true
            st.session_state.class_y_pred = y_pred
            st.session_state.class_y_prob = y_prob
            
            # Compute metrics
            metrics = compute_classification_metrics(y_true, y_pred, y_prob)
            
            st.markdown("### 📊 Metrics")
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("Precision", f"{metrics['precision']:.3f}")
            st.metric("Recall", f"{metrics['recall']:.3f}")
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
            if 'roc_auc' in metrics:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    
    with col2:
        if 'class_y_true' in st.session_state:
            y_true = st.session_state.class_y_true
            y_pred = st.session_state.class_y_pred
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                height=400,
                xaxis_title="",
                yaxis_title=""
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show metrics breakdown
            tn, fp, fn, tp = cm.ravel()
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("True Neg", tn)
            col_b.metric("False Pos", fp)
            col_c.metric("False Neg", fn)
            col_d.metric("True Pos", tp)

# ===== TAB 3: ROC & PR Curves =====
with tab3:
    st.markdown("## ROC and Precision-Recall Curves")
    
    if 'class_y_prob' in st.session_state:
        y_true = st.session_state.class_y_true
        y_prob = st.session_state.class_y_prob
        
        # Compute ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        # Compute PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        
        # Create plots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"ROC Curve (AUC = {roc_auc:.3f})", "Precision-Recall Curve")
        )
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            line=dict(color='blue', width=3),
            name='ROC'
        ), row=1, col=1)
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Random'
        ), row=1, col=1)
        
        # PR curve
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            line=dict(color='green', width=3),
            name='PR'
        ), row=1, col=2)
        
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📖 Understanding the Curves"):
            st.markdown("""
            **ROC Curve:**
            - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
            - Y-axis: True Positive Rate (TPR) = TP / (TP + FN)
            - AUC = 1.0: Perfect classifier
            - AUC = 0.5: Random classifier
            
            **Precision-Recall Curve:**
            - X-axis: Recall = TP / (TP + FN)
            - Y-axis: Precision = TP / (TP + FP)
            - Better for imbalanced datasets
            """)
    else:
        st.info("👆 Generate classification data in the previous tab first!")

# ===== TAB 4: Threshold Tuning =====
with tab4:
    st.markdown("## Interactive Threshold Tuning")
    
    if 'class_y_prob' in st.session_state:
        y_true = st.session_state.class_y_true
        y_prob = st.session_state.class_y_prob
        
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.01)
        
        # Apply threshold
        y_pred_threshold = (y_prob >= threshold).astype(int)
        
        # Compute metrics
        cm = confusion_matrix(y_true, y_pred_threshold)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")
        col4.metric("F1 Score", f"{f1:.3f}")
        
        # Show how metrics change with threshold
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            cm_t = confusion_matrix(y_true, y_pred_t)
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            
            acc = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t)
            prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1_t)
        
        # Plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=accuracies,
            mode='lines',
            name='Accuracy',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=precisions,
            mode='lines',
            name='Precision',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=recalls,
            mode='lines',
            name='Recall',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            name='F1 Score',
            line=dict(color='red')
        ))
        
        # Current threshold
        fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                     annotation_text=f"Current: {threshold:.2f}")
        
        fig.update_layout(
            title="Metrics vs Classification Threshold",
            xaxis_title="Threshold",
            yaxis_title="Metric Value",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Adjust threshold based on your use case: Low threshold → higher recall, High threshold → higher precision")
    else:
        st.info("👆 Generate classification data in Tab 2 first!")

with st.sidebar:
    st.markdown("### 📖 About Metrics")
    st.info("""
    **Regression:**
    - MAE: Average error magnitude
    - RMSE: Penalizes large errors
    - R²: Variance explained
    
    **Classification:**
    - Accuracy: Overall correctness
    - Precision: Positive prediction accuracy
    - Recall: Positive detection rate
    - F1: Harmonic mean of P & R
    
    **ROC-AUC:**
    - Threshold-independent
    - Measures separability
    """)
