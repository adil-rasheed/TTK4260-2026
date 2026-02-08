"""
Ordinary Least Squares Fundamentals
===================================
3D cost surface, gradient descent, and optimizer comparison.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.visualizations import plot_3d_cost_surface, plot_gradient_descent_path, plot_outlier_effect

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Least Squares", page_icon="📉", layout="wide")

st.title("📉 Ordinary Least Squares")
st.markdown("### Understanding Cost Functions and Optimization")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏔️ Cost Surface",
    "🎯 Gradient Descent",
    "⚡ Optimizer Comparison",
    "📐 Closed Form vs Iterative",
    "🎯 Outlier Effects",
    "🔄 Non-Linear Least Squares",
    "⚖️ Weighted Least Squares",
    "🔧 Feature Engineering"
])

# ===== TAB 1: Cost Surface =====
with tab1:
    st.markdown("## 3D Cost Surface Visualization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Problem Setup")
        
        n_samples = st.slider("Number of samples", 10, 200, 50, 10)
        theta1_true = st.slider("True θ₁", -5.0, 5.0, 2.0, 0.5)
        theta2_true = st.slider("True θ₂", -5.0, 5.0, -1.0, 0.5)
        noise_std = st.slider("Noise std", 0.0, 2.0, 0.5, 0.1)
        
        show_solution = st.checkbox("Show true minimum", value=True)
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(n_samples, 2)
        y = X[:, 0] * theta1_true + X[:, 1] * theta2_true + np.random.randn(n_samples) * noise_std
        
        # Compute analytical solution
        theta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        
        st.markdown("### 📊 OLS Solution")
        st.metric("θ₁ (OLS)", f"{theta_ols[0]:.4f}")
        st.metric("θ₂ (OLS)", f"{theta_ols[1]:.4f}")
    
    with col2:
        # Plot cost surface
        fig = plot_3d_cost_surface(X, y)
        
        if show_solution:
            # Add true minimum
            cost_min = np.mean((y - X @ theta_ols)**2)
            fig.add_trace(go.Scatter3d(
                x=[theta_ols[0]],
                y=[theta_ols[1]],
                z=[cost_min],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='OLS Solution'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        The **cost function** measures error:
        
        $$J(\\theta) = \\frac{1}{2n}\\sum_{i=1}^n (y_i - \\mathbf{x}_i^T\\theta)^2$$
        
        The minimum is where $\\nabla J(\\theta) = 0$
        """)

# ===== TAB 2: Gradient Descent =====
with tab2:
    st.markdown("## Gradient Descent Animation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Optimizer Settings")
        
        learning_rate = st.slider("Learning rate", 0.001, 0.5, 0.1, 0.01)
        n_iterations = st.slider("Max iterations", 10, 200, 50, 10)
        init_theta1 = st.slider("Initial θ₁", -5.0, 5.0, -3.0, 0.5, key='init1')
        init_theta2 = st.slider("Initial θ₂", -5.0, 5.0, 3.0, 0.5, key='init2')
        
        if st.button("Run Gradient Descent", type="primary"):
            # Generate data
            np.random.seed(42)
            X = np.random.randn(50, 2)
            y = 2*X[:, 0] - X[:, 1] + np.random.randn(50) * 0.5
            
            # Gradient descent
            theta = np.array([init_theta1, init_theta2])
            history = [theta.copy()]
            costs = []
            
            for i in range(n_iterations):
                # Predictions
                y_pred = X @ theta
                error = y_pred - y
                
                # Gradient
                gradient = (1/len(y)) * (X.T @ error)
                
                # Update
                theta = theta - learning_rate * gradient
                history.append(theta.copy())
                
                # Cost
                cost = np.mean(error**2) / 2
                costs.append(cost)
            
            st.session_state.gd_history = np.array(history)
            st.session_state.gd_costs = costs
            st.session_state.gd_X = X
            st.session_state.gd_y = y
            
            st.success(f"✅ Converged to θ = [{theta[0]:.3f}, {theta[1]:.3f}]")
    
    with col2:
        if 'gd_history' in st.session_state:
            history = st.session_state.gd_history
            costs = st.session_state.gd_costs
            X = st.session_state.gd_X
            y = st.session_state.gd_y
            
            # Create animation
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Gradient Descent Path", "Cost over Iterations")
            )
            
            # Contour plot
            theta1_range = np.linspace(-5, 5, 100)
            theta2_range = np.linspace(-5, 5, 100)
            T1, T2 = np.meshgrid(theta1_range, theta2_range)
            
            Z = np.zeros_like(T1)
            for i in range(T1.shape[0]):
                for j in range(T1.shape[1]):
                    theta_test = np.array([T1[i, j], T2[i, j]])
                    Z[i, j] = np.mean((y - X @ theta_test)**2) / 2
            
            fig.add_trace(go.Contour(
                x=theta1_range,
                y=theta2_range,
                z=Z,
                colorscale='Viridis',
                showscale=False,
                contours=dict(start=0, end=50, size=2)
            ), row=1, col=1)
            
            # Path
            fig.add_trace(go.Scatter(
                x=history[:, 0],
                y=history[:, 1],
                mode='lines+markers',
                marker=dict(color='red', size=6),
                line=dict(color='red', width=2),
                showlegend=False
            ), row=1, col=1)
            
            # Cost curve
            fig.add_trace(go.Scatter(
                x=list(range(len(costs))),
                y=costs,
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ), row=1, col=2)
            
            fig.update_xaxes(title_text="θ₁", row=1, col=1)
            fig.update_yaxes(title_text="θ₂", row=1, col=1)
            fig.update_xaxes(title_text="Iteration", row=1, col=2)
            fig.update_yaxes(title_text="Cost J(θ)", row=1, col=2)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: Optimizer Comparison =====
with tab3:
    st.markdown("## Compare Different Optimizers")
    
    st.markdown("Compare vanilla gradient descent with momentum and adaptive methods")
    
    if st.button("Compare Optimizers", type="primary"):
        # Generate challenging problem
        np.random.seed(42)
        X = np.random.randn(80, 2)
        X[:, 1] = X[:, 1] * 0.1  # Make one direction harder
        y = 2*X[:, 0] - X[:, 1] + np.random.randn(80) * 0.3
        
        def gradient_descent(X, y, lr, n_iter):
            theta = np.array([-3.0, 3.0])
            history = [theta.copy()]
            for _ in range(n_iter):
                grad = (1/len(y)) * X.T @ (X @ theta - y)
                theta = theta - lr * grad
                history.append(theta.copy())
            return np.array(history)
        
        def momentum_gd(X, y, lr, n_iter, beta=0.9):
            theta = np.array([-3.0, 3.0])
            v = np.zeros(2)
            history = [theta.copy()]
            for _ in range(n_iter):
                grad = (1/len(y)) * X.T @ (X @ theta - y)
                v = beta * v + (1 - beta) * grad
                theta = theta - lr * v
                history.append(theta.copy())
            return np.array(history)
        
        # Run optimizers
        gd_path = gradient_descent(X, y, 0.5, 50)
        momentum_path = momentum_gd(X, y, 0.5, 50)
        
        # Plot
        fig = go.Figure()
        
        # Contour
        theta1_range = np.linspace(-5, 5, 100)
        theta2_range = np.linspace(-5, 5, 100)
        T1, T2 = np.meshgrid(theta1_range, theta2_range)
        Z = np.zeros_like(T1)
        for i in range(T1.shape[0]):
            for j in range(T1.shape[1]):
                theta_test = np.array([T1[i, j], T2[i, j]])
                Z[i, j] = np.mean((y - X @ theta_test)**2)
        
        fig.add_trace(go.Contour(
            x=theta1_range,
            y=theta2_range,
            z=Z,
            colorscale='Greys',
            showscale=False
        ))
        
        # Paths
        fig.add_trace(go.Scatter(
            x=gd_path[:, 0],
            y=gd_path[:, 1],
            mode='lines+markers',
            name='Vanilla GD',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=momentum_path[:, 0],
            y=momentum_path[:, 1],
            mode='lines+markers',
            name='Momentum',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Optimizer Comparison",
            xaxis_title="θ₁",
            yaxis_title="θ₂",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Momentum** helps accelerate convergence and dampen oscillations")

# ===== TAB 4: Closed Form vs Iterative =====
with tab4:
    st.markdown("## Closed Form vs Iterative Solutions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Closed Form (Analytical)")
        st.latex(r"\theta^* = (X^T X)^{-1} X^T y")
        st.markdown("""
        **Advantages:**
        - Exact solution
        - No hyperparameters
        - Single computation
        
        **Disadvantages:**
        - O(p³) complexity
        - Requires matrix inversion
        - Memory intensive for large p
        """)
    
    with col2:
        st.markdown("### ⚡ Iterative (Gradient Descent)")
        st.latex(r"\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)")
        st.markdown("""
        **Advantages:**
        - Scalable to large datasets
        - Works with streaming data
        - Lower memory footprint
        
        **Disadvantages:**
        - Requires tuning (α)
        - Approximate solution
        - Multiple iterations
        """)
    
    st.markdown("---")
    
    if st.button("Benchmark Both Methods", type="primary"):
        results = []
        
        for n_features in [10, 50, 100, 500]:
            # Generate data
            np.random.seed(42)
            X = np.random.randn(100, n_features)
            y = np.random.randn(100)
            
            # Closed form
            start = time.time()
            theta_closed = np.linalg.lstsq(X, y, rcond=None)[0]
            time_closed = time.time() - start
            
            # Gradient descent
            start = time.time()
            theta_gd = np.zeros(n_features)
            lr = 0.01
            for _ in range(100):
                grad = (1/len(y)) * X.T @ (X @ theta_gd - y)
                theta_gd = theta_gd - lr * grad
            time_gd = time.time() - start
            
            results.append({
                'Features': n_features,
                'Closed Form (ms)': time_closed * 1000,
                'Gradient Descent (ms)': time_gd * 1000,
                'Solution Diff': np.linalg.norm(theta_closed - theta_gd)
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Plot timing
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Features'],
            y=df['Closed Form (ms)'],
            mode='lines+markers',
            name='Closed Form'
        ))
        fig.add_trace(go.Scatter(
            x=df['Features'],
            y=df['Gradient Descent (ms)'],
            mode='lines+markers',
            name='Gradient Descent'
        ))
        fig.update_layout(
            title="Computation Time vs Problem Size",
            xaxis_title="Number of Features",
            yaxis_title="Time (ms)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ===== TAB 5: Outlier Effects =====
with tab5:
    st.markdown("## 🎯 Impact of Outliers on Least Squares")
    
    st.markdown("""
    OLS is **sensitive to outliers** because it minimizes **squared** errors.
    Large errors from outliers get squared, giving them disproportionate influence.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Data Configuration")
        
        n_points = st.slider("Number of points", 20, 100, 50, 10, key='outlier_n')
        outlier_x = st.slider("Outlier X position", -3.0, 10.0, 8.0, 0.5, key='outlier_x')
        outlier_y = st.slider("Outlier Y position", -5.0, 10.0, -3.0, 0.5, key='outlier_y')
        add_outlier = st.checkbox("Include outlier", value=True, key='add_outlier')
        
        # Generate data
        np.random.seed(42)
        X_clean = np.linspace(0, 5, n_points)
        y_clean = 2 * X_clean + 1 + np.random.randn(n_points) * 0.5
        
        if add_outlier:
            X_with = np.append(X_clean, outlier_x)
            y_with = np.append(y_clean, outlier_y)
            outlier_idx = len(y_clean)
        else:
            X_with = X_clean
            y_with = y_clean
            outlier_idx = None
        
        # Fit models
        from sklearn.linear_model import LinearRegression
        
        model_clean = LinearRegression()
        model_clean.fit(X_clean.reshape(-1, 1), y_clean)
        
        model_with = LinearRegression()
        model_with.fit(X_with.reshape(-1, 1), y_with)
        
        st.markdown("### 📊 Results")
        
        if add_outlier:
            coef_diff = abs(model_with.coef_[0] - model_clean.coef_[0])
            intercept_diff = abs(model_with.intercept_ - model_clean.intercept_)
            
            st.metric("Slope (clean)", f"{model_clean.coef_[0]:.3f}")
            st.metric("Slope (with outlier)", f"{model_with.coef_[0]:.3f}", 
                     delta=f"{coef_diff:.3f}", delta_color="inverse")
            
            st.metric("Intercept (clean)", f"{model_clean.intercept_:.3f}")
            st.metric("Intercept (with outlier)", f"{model_with.intercept_:.3f}",
                     delta=f"{intercept_diff:.3f}", delta_color="inverse")
            
            if coef_diff > 0.5 or intercept_diff > 0.5:
                st.warning("⚠️ Outlier has significant impact!")
        else:
            st.metric("Slope", f"{model_clean.coef_[0]:.3f}")
            st.metric("Intercept", f"{model_clean.intercept_:.3f}")
    
    with col2:
        # Plot using visualization function
        fig = plot_outlier_effect(X_with, y_with, outlier_idx=[outlier_idx] if outlier_idx is not None else None)
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational content
    st.markdown("### 🎓 Understanding Outlier Sensitivity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Why OLS is Sensitive:**
        - Uses **squared errors**: $(y_i - \\hat{y}_i)^2$
        - Large errors get **amplified**
        - Single outlier can dominate cost
        - Fit pulled toward outliers
        """)
    
    with col2:
        st.markdown("""
        **Alternatives:**
        - **Robust regression** (e.g., RANSAC, Huber loss)
        - **Outlier detection** before fitting
        - **Regularization** (can help somewhat)
        - **Median-based methods** (L1 regression)
        """)
    
    st.info("""
    💡 **Key Takeaway**: Always **visualize your data** and check for outliers before applying OLS. 
    A single extreme point can dramatically change your regression line and lead to poor predictions!
    """)

# ===== TAB 6: Non-Linear Least Squares =====
with tab6:
    st.markdown("## Non-Linear Least Squares")
    
    st.markdown("""
    When the relationship is **non-linear in parameters**, we can't use the closed-form solution.
    
    **Example:** Exponential decay model: $y = a \\cdot e^{-b \\cdot x} + c$
    
    **Solution:** Use iterative optimization (e.g., Levenberg-Marquardt, Trust Region)
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Setup")
        
        model_type = st.selectbox(
            "Model type",
            ["Exponential Decay", "Gaussian", "Logistic Growth", "Power Law"]
        )
        
        n_samples = st.slider("Number of samples", 20, 200, 50, 10, key='nls_n')
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, 0.05, key='nls_noise')
        
        if st.button("Fit Non-Linear Model", type="primary", key='nls_btn'):
            from scipy.optimize import curve_fit
            np.random.seed(42)
            
            x = np.linspace(0, 10, n_samples)
            
            if model_type == "Exponential Decay":
                # True model: y = a * exp(-b * x) + c
                true_params = [5.0, 0.5, 1.0]
                y_true = true_params[0] * np.exp(-true_params[1] * x) + true_params[2]
                y_noisy = y_true + np.random.randn(n_samples) * noise_level
                
                def model(x, a, b, c):
                    return a * np.exp(-b * x) + c
                
                param_names = ['a', 'b', 'c']
                initial_guess = [1.0, 0.1, 0.0]
                
            elif model_type == "Gaussian":
                # y = a * exp(-(x-mu)^2 / (2*sigma^2))
                true_params = [3.0, 5.0, 1.5]
                y_true = true_params[0] * np.exp(-((x - true_params[1])**2) / (2 * true_params[2]**2))
                y_noisy = y_true + np.random.randn(n_samples) * noise_level
                
                def model(x, a, mu, sigma):
                    return a * np.exp(-((x - mu)**2) / (2 * sigma**2))
                
                param_names = ['a', 'μ', 'σ']
                initial_guess = [1.0, 5.0, 1.0]
                
            elif model_type == "Logistic Growth":
                # y = L / (1 + exp(-k*(x-x0)))
                true_params = [10.0, 0.5, 5.0]
                y_true = true_params[0] / (1 + np.exp(-true_params[1] * (x - true_params[2])))
                y_noisy = y_true + np.random.randn(n_samples) * noise_level
                
                def model(x, L, k, x0):
                    return L / (1 + np.exp(-k * (x - x0)))
                
                param_names = ['L', 'k', 'x₀']
                initial_guess = [5.0, 0.1, 5.0]
                
            else:  # Power Law
                # y = a * x^b
                x = np.linspace(0.1, 10, n_samples)  # Avoid x=0
                true_params = [2.0, 0.8]
                y_true = true_params[0] * (x ** true_params[1])
                y_noisy = y_true + np.random.randn(n_samples) * noise_level
                
                def model(x, a, b):
                    return a * (x ** b)
                
                param_names = ['a', 'b']
                initial_guess = [1.0, 0.5]
            
            # Fit model
            try:
                fitted_params, _ = curve_fit(model, x, y_noisy, p0=initial_guess)
                y_fitted = model(x, *fitted_params)
                
                st.session_state.nls_x = x
                st.session_state.nls_y_true = y_true
                st.session_state.nls_y_noisy = y_noisy
                st.session_state.nls_y_fitted = y_fitted
                st.session_state.nls_true_params = true_params
                st.session_state.nls_fitted_params = fitted_params
                st.session_state.nls_param_names = param_names
                st.session_state.nls_model_type = model_type
                
                st.markdown("### 📊 Fitted Parameters")
                for name, true_val, fit_val in zip(param_names, true_params, fitted_params):
                    col_a, col_b = st.columns(2)
                    col_a.metric(f"True {name}", f"{true_val:.4f}")
                    col_b.metric(f"Fitted {name}", f"{fit_val:.4f}")
                    
            except Exception as e:
                st.error(f"Fitting failed: {e}")
                st.info("Try different initial guess or less noise")
    
    with col2:
        if 'nls_x' in st.session_state:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=st.session_state.nls_x,
                y=st.session_state.nls_y_noisy,
                mode='markers',
                name='Data',
                marker=dict(size=8, color='lightblue', opacity=0.6)
            ))
            
            fig.add_trace(go.Scatter(
                x=st.session_state.nls_x,
                y=st.session_state.nls_y_true,
                mode='lines',
                name='True Model',
                line=dict(color='green', width=3, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=st.session_state.nls_x,
                y=st.session_state.nls_y_fitted,
                mode='lines',
                name='Fitted Model',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title=f"{st.session_state.nls_model_type} - Non-Linear Fit",
                xaxis_title="x",
                yaxis_title="y",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Non-Linear LS uses iterative optimization:**
            1. Start with initial parameter guess
            2. Linearize model around current parameters (Taylor expansion)
            3. Solve linear LS problem
            4. Update parameters
            5. Repeat until convergence
            
            **Common algorithms:**
            - Gauss-Newton
            - Levenberg-Marquardt
            - Trust Region Reflective
            """)

# ===== TAB 7: Weighted Least Squares =====
with tab7:
    st.markdown("## Weighted Least Squares (WLS)")
    
    st.markdown("""
    When observations have **different variances** (heteroscedasticity), give more weight to precise measurements.
    
    **WLS minimizes:** $\\sum_{i=1}^n w_i (y_i - \\hat{y}_i)^2$
    
    where $w_i = 1/\\sigma_i^2$ (inverse variance weighting)
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Setup")
        
        n_samples = st.slider("Number of samples", 50, 200, 100, 10, key='wls_n')
        het_strength = st.slider("Heteroscedasticity strength", 0.0, 3.0, 1.5, 0.1, key='wls_het')
        
        if st.button("Compare OLS vs WLS", type="primary", key='wls_btn'):
            np.random.seed(42)
            
            # Generate data with increasing variance
            x = np.linspace(0, 10, n_samples)
            y_true = 2 + 3 * x
            
            # Heteroscedastic noise (variance increases with x)
            noise_std = 0.5 + het_strength * x
            y_noisy = y_true + np.random.randn(n_samples) * noise_std
            
            # OLS (ignores variance differences)
            X = np.column_stack([np.ones(n_samples), x])
            theta_ols = np.linalg.lstsq(X, y_noisy, rcond=None)[0]
            y_ols = X @ theta_ols
            
            # WLS (weights by inverse variance)
            weights = 1 / (noise_std ** 2)
            W = np.diag(weights)
            theta_wls = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y_noisy
            y_wls = X @ theta_wls
            
            st.session_state.wls_x = x
            st.session_state.wls_y_true = y_true
            st.session_state.wls_y_noisy = y_noisy
            st.session_state.wls_y_ols = y_ols
            st.session_state.wls_y_wls = y_wls
            st.session_state.wls_noise_std = noise_std
            st.session_state.wls_theta_ols = theta_ols
            st.session_state.wls_theta_wls = theta_wls
            
            st.markdown("### 📊 Results")
            
            st.markdown("**OLS:**")
            col_a, col_b = st.columns(2)
            col_a.metric("Intercept", f"{theta_ols[0]:.4f}")
            col_b.metric("Slope", f"{theta_ols[1]:.4f}")
            
            st.markdown("**WLS:**")
            col_c, col_d = st.columns(2)
            col_c.metric("Intercept", f"{theta_wls[0]:.4f}")
            col_d.metric("Slope", f"{theta_wls[1]:.4f}")
            
            st.markdown("**True:**")
            st.write("Intercept: 2.0, Slope: 3.0")
    
    with col2:
        if 'wls_x' in st.session_state:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Data with Heteroscedastic Noise", "Comparison of Fits")
            )
            
            # Left: Show noise variance
            x = st.session_state.wls_x
            y_noisy = st.session_state.wls_y_noisy
            noise_std = st.session_state.wls_noise_std
            
            # Color by uncertainty
            fig.add_trace(go.Scatter(
                x=x, y=y_noisy,
                mode='markers',
                name='Data',
                marker=dict(
                    size=8,
                    color=noise_std,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Noise σ", x=0.45),
                    opacity=0.6
                )
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=x, y=st.session_state.wls_y_true,
                mode='lines',
                name='True',
                line=dict(color='green', width=2, dash='dash'),
                showlegend=False
            ), row=1, col=1)
            
            # Right: Compare OLS vs WLS
            fig.add_trace(go.Scatter(
                x=x, y=y_noisy,
                mode='markers',
                name='Data',
                marker=dict(size=6, color='lightgray', opacity=0.4),
                showlegend=False
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=x, y=st.session_state.wls_y_true,
                mode='lines',
                name='True',
                line=dict(color='green', width=2, dash='dash')
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=x, y=st.session_state.wls_y_ols,
                mode='lines',
                name='OLS',
                line=dict(color='blue', width=2)
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=x, y=st.session_state.wls_y_wls,
                mode='lines',
                name='WLS',
                line=dict(color='red', width=2)
            ), row=1, col=2)
            
            fig.update_xaxes(title_text="x", row=1, col=1)
            fig.update_xaxes(title_text="x", row=1, col=2)
            fig.update_yaxes(title_text="y", row=1, col=1)
            fig.update_yaxes(title_text="y", row=1, col=2)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Key Insight:**
            - **Red data** (high x) has higher variance → less reliable
            - **OLS** gives equal weight to all points → influenced by noisy high-x points
            - **WLS** down-weights unreliable points → better estimates!
            
            **When to use WLS:**
            - Measurement precision varies across observations
            - Variance increases/decreases with predictor
            - Financial data (different company sizes)
            - Scientific measurements with varying equipment precision
            """)

# ===== TAB 8: Feature Engineering =====
with tab8:
    st.markdown("## Feature Engineering for Better Fits")
    
    st.markdown("""
    **Feature Engineering:** Transform or create new features to capture non-linear relationships
    
    **Common transformations:**
    - Polynomial features: $x, x^2, x^3, ...$
    - Interaction terms: $x_1 \\cdot x_2$
    - Logarithmic: $\\log(x)$
    - Trigonometric: $\\sin(x), \\cos(x)$
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Setup")
        
        true_function = st.selectbox(
            "True relationship",
            ["Linear", "Quadratic", "Cubic", "Sine Wave", "Exponential"]
        )
        
        n_samples = st.slider("Number of samples", 30, 200, 50, 10, key='fe_n')
        noise = st.slider("Noise level", 0.0, 2.0, 0.3, 0.1, key='fe_noise')
        
        poly_degree = st.slider("Polynomial degree for fit", 1, 10, 2, 1, key='fe_poly')
        
        if st.button("Fit with Features", type="primary", key='fe_btn'):
            np.random.seed(42)
            
            x = np.linspace(0, 10, n_samples)
            
            if true_function == "Linear":
                y_true = 2 + 3 * x
            elif true_function == "Quadratic":
                y_true = 1 + 2 * x - 0.3 * x**2
            elif true_function == "Cubic":
                y_true = 0.1 * x**3 - 1.5 * x**2 + 5 * x + 1
            elif true_function == "Sine Wave":
                y_true = 5 + 3 * np.sin(x)
            else:  # Exponential
                y_true = np.exp(0.3 * x)
            
            y_noisy = y_true + np.random.randn(n_samples) * noise
            
            # Fit polynomial
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
            X_poly = poly.fit_transform(x.reshape(-1, 1))
            
            model = LinearRegression(fit_intercept=False)  # Bias already in features
            model.fit(X_poly, y_noisy)
            
            x_fine = np.linspace(0, 10, 300)
            X_fine_poly = poly.transform(x_fine.reshape(-1, 1))
            y_fitted = model.predict(X_fine_poly)
            
            st.session_state.fe_x = x
            st.session_state.fe_y_true = y_true
            st.session_state.fe_y_noisy = y_noisy
            st.session_state.fe_x_fine = x_fine
            st.session_state.fe_y_fitted = y_fitted
            st.session_state.fe_true_fn = true_function
            st.session_state.fe_degree = poly_degree
            st.session_state.fe_coefs = model.coef_
            
            st.markdown("### 📊 Model Coefficients")
            st.write(f"Degree {poly_degree} polynomial:")
            for i, coef in enumerate(model.coef_[:min(6, len(model.coef_))]):
                st.write(f"$\\theta_{i}$: {coef:.4f}")
            if len(model.coef_) > 6:
                st.write(f"... ({len(model.coef_)} coefficients total)")
    
    with col2:
        if 'fe_x' in st.session_state:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=st.session_state.fe_x,
                y=st.session_state.fe_y_noisy,
                mode='markers',
                name='Data',
                marker=dict(size=8, color='lightblue', opacity=0.6)
            ))
            
            fig.add_trace(go.Scatter(
                x=st.session_state.fe_x,
                y=st.session_state.fe_y_true,
                mode='lines',
                name=f'True ({st.session_state.fe_true_fn})',
                line=dict(color='green', width=3, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=st.session_state.fe_x_fine,
                y=st.session_state.fe_y_fitted,
                mode='lines',
                name=f'Polynomial Fit (deg={st.session_state.fe_degree})',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Feature Engineering: Polynomial Regression",
                xaxis_title="x",
                yaxis_title="y",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Model:** $\\hat{y} = \\theta_0 + \\theta_1 x + \\theta_2 x^2 + ... + \\theta_d x^d$
            
            **Still linear in parameters** → Can use OLS!
            
            **Cautions:**
            - **Underfitting:** Degree too low → misses true pattern
            - **Overfitting:** Degree too high → fits noise
            - **Extrapolation:** Polynomials behave badly outside training range
            
            **Better alternatives for non-linearity:**
            - Splines (piecewise polynomials)
            - Gaussian processes
            - Neural networks
            """)

with st.sidebar:
    st.markdown("### 📖 About Least Squares")
    st.info("""
    **OLS minimizes:**
    
    $$\\min_\\theta \\|y - X\\theta\\|^2$$
    
    **Analytical solution:**
    
    $$\\theta^* = (X^T X)^{-1} X^T y$$
    
    **Gradient:**
    
    $$\\nabla J = \\frac{1}{n}X^T(X\\theta - y)$$
    
    Key property: Convex optimization
    """)
