"""
Principal Component Regression Interactive Page
================================================
Explore multicollinearity and PCR solution.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_generator import generate_correlated_data, generate_process_sensor_data
from utils.models import PCRFromScratch, bootstrap_stability_analysis, compare_regularization_methods
from utils.visualizations import plot_ols_vs_pcr_contours, plot_bootstrap_comparison, plot_correlation_heatmap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PCR Interactive", page_icon="🔵", layout="wide")

st.title("🔵 Principal Component Regression")
st.markdown("### Handling Multicollinearity Through Dimensionality Reduction")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Multicollinearity Demo",
    "🎯 PCR vs OLS Comparison",
    "📈 Bootstrap Stability",
    "⚖️ Method Comparison"
])

# ===== TAB 1: Multicollinearity Demo =====
with tab1:
    st.markdown("## Understanding Multicollinearity")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Data Controls")
        
        n_samples = st.slider("Number of samples", 50, 500, 100, 10)
        correlation = st.slider("Correlation between predictors", 0.5, 0.99, 0.95, 0.01,
                               help="Higher correlation = more severe multicollinearity")
        noise_level = st.slider("Noise level", 0.1, 2.0, 0.5, 0.1)
        
        random_seed = st.number_input("Random seed", 0, 9999, 42)
        
        if st.button("Generate Data", type="primary"):
            st.session_state.data_generated = True
    
    with col2:
        if 'data_generated' in st.session_state or st.session_state.get('data_generated', False):
            # Generate data
            X, y, beta_true = generate_correlated_data(
                n=n_samples, p=2, correlation=correlation,
                noise_std=noise_level, random_state=random_seed
            )
            
            # Compute condition number
            cond_num = np.linalg.cond(X.T @ X)
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Condition Number", f"{cond_num:.2f}",
                         delta="Severe!" if cond_num > 30 else "Moderate" if cond_num > 10 else "Good")
            with col_b:
                corr_matrix = np.corrcoef(X.T)
                if corr_matrix.shape == (2, 2):
                    actual_corr = corr_matrix[0, 1]
                else:
                    actual_corr = 0.0
                st.metric("Actual Correlation", f"{actual_corr:.3f}")
            with col_c:
                st.metric("Sample Size", n_samples)
            
            # Show correlation heatmap
            fig = plot_correlation_heatmap(X, ['X₁', 'X₂'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Data preview
            with st.expander("📋 View Data"):
                df = pd.DataFrame(X, columns=['X₁', 'X₂'])
                df['y'] = y
                st.dataframe(df.head(20), use_container_width=True)

# ===== TAB 2: PCR vs OLS Comparison =====
with tab2:
    st.markdown("## Loss Surface Comparison")
    st.markdown("See how PCR transforms the unstable elongated valleys of OLS into stable circular contours")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Settings")
        
        correlation_demo = st.slider("Correlation", 0.90, 0.99, 0.99, 0.01, key='corr_demo')
        n_perturbations = st.slider("Number of perturbations", 5, 50, 20, 5,
                                    help="Number of noisy datasets to generate")
        noise_scale = st.slider("Perturbation noise", 0.1, 1.0, 0.3, 0.1)
        
        if st.button("Run Analysis", type="primary", key='run_pcr'):
            with st.spinner("Computing OLS and PCR solutions..."):
                # Generate data
                X, y, beta_true = generate_correlated_data(n=100, p=2, correlation=correlation_demo, random_state=42)
                
                # OLS solution
                theta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
                cond_num = np.linalg.cond(X.T @ X)
                
                # PCR solution
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_std = scaler_X.fit_transform(X)
                y_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
                
                pca = PCA(n_components=2)
                T = pca.fit_transform(X_std)
                V = pca.components_.T
                beta_scores = np.linalg.lstsq(T, y_std, rcond=None)[0]
                theta_pcr = V @ beta_scores
                
                # Generate perturbed solutions
                theta_ols_perturbed_list = []
                beta_scores_perturbed_list = []
                
                for i in range(n_perturbations):
                    y_pert = y + np.random.normal(0, noise_scale, len(y))
                    
                    # OLS perturbed
                    theta_ols_pert = np.linalg.lstsq(X, y_pert, rcond=None)[0]
                    theta_ols_perturbed_list.append(theta_ols_pert)
                    
                    # PCR perturbed
                    y_pert_std = scaler_y.fit_transform(y_pert.reshape(-1, 1)).ravel()
                    beta_pert = np.linalg.lstsq(T, y_pert_std, rcond=None)[0]
                    beta_scores_perturbed_list.append(beta_pert)
                
                theta_ols_perturbed = np.array(theta_ols_perturbed_list)
                beta_scores_perturbed = np.array(beta_scores_perturbed_list)
                
                # Store in session state
                st.session_state.pcr_results = {
                    'theta_ols': theta_ols,
                    'theta_pcr': theta_pcr,
                    'beta_scores': beta_scores,
                    'theta_ols_perturbed': theta_ols_perturbed,
                    'beta_scores_perturbed': beta_scores_perturbed,
                    'cond_num': cond_num,
                    'ols_std': theta_ols_perturbed.std(axis=0),
                    'pcr_std': beta_scores_perturbed.std(axis=0)
                }
    
    with col2:
        if 'pcr_results' in st.session_state:
            results = st.session_state.pcr_results
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Condition Number", f"{results['cond_num']:.1f}")
            with col_b:
                ols_instability = np.mean(results['ols_std'])
                st.metric("OLS Instability", f"{ols_instability:.3f}", delta="High", delta_color="inverse")
            with col_c:
                pcr_instability = np.mean(results['pcr_std'])
                st.metric("PCR Instability", f"{pcr_instability:.3f}", delta="Low", delta_color="normal")
            
            # Visualization
            fig = plot_ols_vs_pcr_contours(
                results['theta_ols'],
                results['theta_pcr'],
                results['beta_scores'],
                results['theta_ols_perturbed'],
                results['beta_scores_perturbed']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            with st.expander("📊 Detailed Results"):
                col_x, col_y = st.columns(2)
                with col_x:
                    st.markdown("**OLS Coefficients:**")
                    st.write(f"θ₁ = {results['theta_ols'][0]:.4f} (std = {results['ols_std'][0]:.4f})")
                    st.write(f"θ₂ = {results['theta_ols'][1]:.4f} (std = {results['ols_std'][1]:.4f})")
                
                with col_y:
                    st.markdown("**PCR Score Coefficients:**")
                    st.write(f"β₁ = {results['beta_scores'][0]:.4f} (std = {results['pcr_std'][0]:.4f})")
                    st.write(f"β₂ = {results['beta_scores'][1]:.4f} (std = {results['pcr_std'][1]:.4f})")

# ===== TAB 3: Bootstrap Stability =====
with tab3:
    st.markdown("## Bootstrap Stability Analysis")
    st.markdown("Compare coefficient stability between OLS and PCR using bootstrap resampling")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎲 Bootstrap Settings")
        
        n_boot = st.slider("Bootstrap samples", 50, 500, 100, 50)
        corr_boot = st.slider("Correlation", 0.80, 0.99, 0.95, 0.01, key='corr_boot')
        
        if st.button("Run Bootstrap Analysis", type="primary"):
            with st.spinner(f"Running {n_boot} bootstrap iterations..."):
                X, y, beta_true = generate_correlated_data(n=50, p=2, correlation=corr_boot, random_state=42)
                
                # Bootstrap OLS
                bootstrap_ols = bootstrap_stability_analysis(X, y, n_bootstrap=n_boot, model_type='ols')
                
                # Bootstrap PCR
                bootstrap_pcr = bootstrap_stability_analysis(X, y, n_bootstrap=n_boot, model_type='pcr', n_components=1)
                
                st.session_state.bootstrap_results = {
                    'ols': bootstrap_ols,
                    'pcr': bootstrap_pcr,
                    'beta_true': beta_true,
                    'ols_std': bootstrap_ols.std(axis=0),
                    'pcr_std': bootstrap_pcr.std(axis=0)
                }
    
    with col2:
        if 'bootstrap_results' in st.session_state:
            results = st.session_state.bootstrap_results
            
            # Stability comparison
            stability_ratio = results['ols_std'][0] / results['pcr_std'][0]
            
            st.success(f"✅ PCR is **{stability_ratio:.1f}×** more stable than OLS!")
            
            # Visualization
            fig = plot_bootstrap_comparison(
                results['ols'],
                results['pcr'],
                results['beta_true']
            )
            st.plotly_chart(fig, use_container_width=True)

# ===== TAB 4: Method Comparison =====
with tab4:
    st.markdown("## Compare Multiple Regularization Methods")
    st.markdown("See how PCR stacks up against Ridge and Lasso regression")
    
    if st.button("Run Comprehensive Comparison", type="primary"):
        with st.spinner("Comparing methods..."):
            # Generate high-dimensional data
            X, y, beta_true = generate_correlated_data(n=100, p=12, correlation=0.9, random_state=42)
            
            # Compare methods
            results = compare_regularization_methods(X, y, alphas=[0.1, 1.0, 10.0])
            
            # Create comparison dataframe
            comparison_data = []
            for method, metrics in results.items():
                comparison_data.append({
                    'Method': method,
                    'Train R²': f"{metrics['train_r2']:.4f}",
                    'Test R²': f"{metrics['test_r2']:.4f}",
                    '||θ||₂': f"{metrics['l2_norm']:.4f}",
                    'Nonzero': metrics.get('n_nonzero', len(metrics['coeffs']))
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            st.dataframe(df_comparison, use_container_width=True)
            
            st.info("💡 **Key Insights:** PCR provides regularization through dimensionality reduction, while Ridge/Lasso use penalty terms. Each has different strengths depending on the data structure.")

# Sidebar info
with st.sidebar:
    st.markdown("### 📖 About PCR")
    st.info("""
    **Principal Component Regression** handles multicollinearity by:
    
    1. **PCA**: Transform to uncorrelated components
    2. **Regression**: Fit on PC scores
    3. **Back-project**: Return to original space
    
    **Benefits:**
    - Stable coefficients
    - Interpretable components
    - Automatic regularization
    """)
