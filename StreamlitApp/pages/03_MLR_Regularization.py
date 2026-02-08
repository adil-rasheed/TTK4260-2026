"""
Multiple Linear Regression with Regularization
===============================================
Explore Ridge, Lasso, and ElasticNet regularization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_generator import generate_correlated_data
from utils.visualizations import plot_correlation_heatmap, plot_l1_l2_geometry, plot_regularization_paths

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MLR & Regularization", page_icon="📏", layout="wide")

st.title("📏 Multiple Linear Regression & Regularization")
st.markdown("### L1, L2, and ElasticNet Penalty Methods")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Regularization Comparison",
    "🎯 Coefficient Paths",
    "💎 L1 vs L2 Geometry",
    "⚖️ Bias-Variance Tradeoff",
    "🔍 Feature Selection",
    "🏷️ Categorical Variables",
    "🔗 Multicollinearity & VIF"
])

# ===== TAB 1: Regularization Comparison =====
with tab1:
    st.markdown("## Compare OLS, Ridge, Lasso, and ElasticNet")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Data Settings")
        
        n_samples = st.slider("Number of samples", 50, 500, 100, 10)
        n_features = st.slider("Number of features", 2, 20, 10, 1)
        correlation = st.slider("Feature correlation", 0.0, 0.95, 0.7, 0.05)
        noise_std = st.slider("Noise level", 0.1, 3.0, 0.5, 0.1)
        
        st.markdown("### 🎚️ Regularization")
        
        alpha_ridge = st.slider("Ridge α (λ)", 0.01, 10.0, 1.0, 0.1, key='ridge_alpha')
        alpha_lasso = st.slider("Lasso α", 0.01, 10.0, 0.1, 0.01, key='lasso_alpha')
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Generate data
                X, y, beta_true = generate_correlated_data(
                    n=n_samples, p=n_features, correlation=correlation,
                    noise_std=noise_std, random_state=42
                )
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                # Standardize
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                models = {
                    'OLS': LinearRegression(),
                    'Ridge': Ridge(alpha=alpha_ridge),
                    'Lasso': Lasso(alpha=alpha_lasso),
                    'ElasticNet': ElasticNet(alpha=alpha_lasso, l1_ratio=0.5)
                }
                
                results = {}
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    results[name] = {
                        'train_r2': model.score(X_train_scaled, y_train),
                        'test_r2': model.score(X_test_scaled, y_test),
                        'coeffs': model.coef_,
                        'l2_norm': np.linalg.norm(model.coef_),
                        'l1_norm': np.sum(np.abs(model.coef_)),
                        'n_nonzero': np.sum(np.abs(model.coef_) > 1e-5)
                    }
                
                st.session_state.mlr_results = results
                st.session_state.beta_true = beta_true
    
    with col2:
        if 'mlr_results' in st.session_state:
            results = st.session_state.mlr_results
            
            # Create comparison table
            comparison_data = []
            for method, metrics in results.items():
                comparison_data.append({
                    'Method': method,
                    'Train R²': f"{metrics['train_r2']:.4f}",
                    'Test R²': f"{metrics['test_r2']:.4f}",
                    '||θ||₂': f"{metrics['l2_norm']:.3f}",
                    '||θ||₁': f"{metrics['l1_norm']:.3f}",
                    'Nonzero': metrics['n_nonzero']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Plot coefficients
            fig = go.Figure()
            
            x_pos = np.arange(len(results['OLS']['coeffs']))
            
            for method in ['OLS', 'Ridge', 'Lasso', 'ElasticNet']:
                fig.add_trace(go.Scatter(
                    x=x_pos,
                    y=results[method]['coeffs'],
                    mode='lines+markers',
                    name=method
                ))
            
            fig.update_layout(
                title="Coefficient Comparison",
                xaxis_title="Feature Index",
                yaxis_title="Coefficient Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ===== TAB 2: Coefficient Paths =====
with tab2:
    st.markdown("## Regularization Paths")
    st.markdown("See how coefficients shrink as regularization strength increases")
    
    method = st.selectbox("Select method", ["Ridge", "Lasso"])
    
    if st.button("Compute Paths", type="primary", key='paths'):
        with st.spinner("Computing regularization paths..."):
            # Generate data
            X, y, beta_true = generate_correlated_data(n=100, p=8, correlation=0.7, random_state=42)
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Alpha range
            alphas = np.logspace(-3, 2, 50)
            coeffs_path = []
            
            for alpha in alphas:
                if method == "Ridge":
                    model = Ridge(alpha=alpha)
                else:
                    model = Lasso(alpha=alpha, max_iter=10000)
                
                model.fit(X_scaled, y)
                coeffs_path.append(model.coef_)
            
            coeffs_path = np.array(coeffs_path)
            
            # Plot using enhanced function
            feature_names = [f'β{i+1}' for i in range(coeffs_path.shape[1])]
            fig = plot_regularization_paths(alphas, coeffs_path, feature_names)
            
            # Update title
            fig.update_layout(title=f"{method} Regularization Path")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Observations")
                n_zero_at_max = np.sum(np.abs(coeffs_path[-1]) < 1e-5)
                st.write(f"- At max α: **{n_zero_at_max}/{len(coeffs_path[-1])}** coefficients ≈ 0")
                st.write(f"- Method: **{method}**")
                if method == 'Lasso':
                    st.write("- **Sparsity achieved** ✓")
                else:
                    st.write("- All coefficients remain non-zero")
            
            with col2:
                st.markdown("### 📊 Statistics")
                st.write(f"- Min α: {alphas.min():.4f}")
                st.write(f"- Max α: {alphas.max():.1f}")
                st.write(f"- Number of features: {coeffs_path.shape[1]}")
            
            st.info(f"💡 **{method}**: As α increases, coefficients {'shrink smoothly toward zero but rarely reach exactly zero' if method == 'Ridge' else 'become exactly zero (automatic feature selection)'}")

# ===== TAB 3: L1 vs L2 Geometry =====
with tab3:
    st.markdown("## 💎 Geometric Interpretation: Why L1 Produces Sparsity")
    
    st.markdown("""
    The **constraint regions** explain why:
    - **L2 (Ridge)**: Circle constraint → solutions rarely on axes
    - **L1 (Lasso)**: Diamond constraint → solutions often on axes (sparse)
    
    The contours show OLS cost function (ellipses centered at OLS solution).
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Settings")
        
        # Generate simple 2D problem
        ols_1 = st.slider("OLS β₁", -3.0, 3.0, 2.0, 0.5, key='geom_ols1')
        ols_2 = st.slider("OLS β₂", -3.0, 3.0, 1.5, 0.5, key='geom_ols2')
        lambda_geom = st.slider("Regularization λ", 0.1, 5.0, 1.0, 0.1, key='geom_lambda')
        
        # Simulate Ridge and Lasso solutions
        # Ridge: shrinks proportionally
        ridge_factor = 1 / (1 + lambda_geom)
        ridge_1 = ols_1 * ridge_factor
        ridge_2 = ols_2 * ridge_factor
        
        # Lasso: soft thresholding (simplified)
        lasso_1 = np.sign(ols_1) * max(0, abs(ols_1) - lambda_geom * 0.5)
        lasso_2 = np.sign(ols_2) * max(0, abs(ols_2) - lambda_geom * 0.5)
        
        st.markdown("### 📊 Solutions")
        st.write(f"**OLS**: ({ols_1:.2f}, {ols_2:.2f})")
        st.write(f"**Ridge**: ({ridge_1:.2f}, {ridge_2:.2f})")
        st.write(f"**Lasso**: ({lasso_1:.2f}, {lasso_2:.2f})")
        
        # Check sparsity
        if abs(lasso_1) < 0.01 or abs(lasso_2) < 0.01:
            st.success("✅ Lasso achieved sparsity!")
        else:
            st.info("No sparsity yet (increase λ)")
    
    with col2:
        # Plot geometry
        beta_ols = np.array([ols_1, ols_2])
        beta_ridge = np.array([ridge_1, ridge_2])
        beta_lasso = np.array([lasso_1, lasso_2])
        
        fig = plot_l1_l2_geometry(beta_ols, beta_ridge, beta_lasso, lambda_geom)
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational explanation
    st.markdown("### 🎓 Why This Matters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **L2 (Ridge) - Circle**
        - Smooth penalty: $\\lambda \\sum \\beta_i^2$
        - Constraint: $\\|\\beta\\|_2 \\leq c$
        - Circle has no corners
        - Ellipse rarely touches axis
        - **All coefficients non-zero**
        """)
    
    with col2:
        st.markdown("""
        **L1 (Lasso) - Diamond**
        - Sharp penalty: $\\lambda \\sum |\\beta_i|$
        - Constraint: $\\|\\beta\\|_1 \\leq c$
        - Diamond has corners on axes
        - Ellipse often touches corner
        - **Coefficients become exactly zero**
        """)
    
    st.info("""
    💡 **Key Insight**: The diamond shape of L1 constraint has corners on the coordinate axes. 
    When the elliptical contours of the OLS cost function touch the constraint region, 
    they are much more likely to touch at a corner (axis) for L1 than for L2 (circle). 
    This is why Lasso produces sparse solutions!
    """)

# ===== TAB 4: Bias-Variance Tradeoff =====
with tab4:
    st.markdown("## Bias-Variance Tradeoff")
    
    st.markdown("""
    Regularization balances:
    - **Low α**: Low bias, high variance (overfitting)
    - **High α**: High bias, low variance (underfitting)
    - **Optimal α**: Best generalization
    """)
    
    if st.button("Demonstrate Tradeoff", type="primary"):
        # Generate data
        X, y, _ = generate_correlated_data(n=80, p=10, correlation=0.8, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try different alphas
        alphas = np.logspace(-2, 2, 30)
        train_scores = []
        test_scores = []
        
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)
            train_scores.append(model.score(X_train_scaled, y_train))
            test_scores.append(model.score(X_test_scaled, y_test))
        
        # Plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=alphas,
            y=train_scores,
            mode='lines',
            name='Training R²',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=alphas,
            y=test_scores,
            mode='lines',
            name='Test R²',
            line=dict(color='red', width=2)
        ))
        
        # Find optimal alpha
        optimal_idx = np.argmax(test_scores)
        optimal_alpha = alphas[optimal_idx]
        
        fig.add_vline(x=optimal_alpha, line_dash="dash", line_color="green",
                     annotation_text=f"Optimal α={optimal_alpha:.3f}")
        
        fig.update_layout(
            title="Bias-Variance Tradeoff",
            xaxis_title="α (log scale)",
            yaxis_title="R² Score",
            xaxis_type="log",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"✅ Optimal α = {optimal_alpha:.3f} (maximizes test R²)")

# ===== TAB 5: Feature Selection =====
with tab5:
    st.markdown("## Automatic Feature Selection with Lasso")
    
    st.markdown("Lasso can zero out irrelevant features, performing automatic feature selection")
    
    n_relevant = st.slider("Number of relevant features", 2, 10, 5, 1)
    n_noise = st.slider("Number of noise features", 0, 20, 10, 1)
    
    if st.button("Run Feature Selection", type="primary"):
        # Generate data with noise features
        X_relevant, y, beta_relevant = generate_correlated_data(
            n=100, p=n_relevant, correlation=0.3, random_state=42
        )
        
        # Add noise features
        X_noise = np.random.randn(100, n_noise)
        X_full = np.hstack([X_relevant, X_noise])
        
        # True coefficients
        beta_true_full = np.hstack([beta_relevant, np.zeros(n_noise)])
        
        # Train Lasso
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_full, y)
        
        # Plot coefficients
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(beta_true_full) + 1)),
            y=beta_true_full,
            name='True Coefficients',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(lasso.coef_) + 1)),
            y=lasso.coef_,
            name='Lasso Estimates',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.add_vline(x=n_relevant + 0.5, line_dash="dash",
                     annotation_text="Noise features →")
        
        fig.update_layout(
            title="Feature Selection Results",
            xaxis_title="Feature Index",
            yaxis_title="Coefficient Value",
            height=400,
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        n_selected = np.sum(np.abs(lasso.coef_) > 1e-5)
        st.info(f"💡 Lasso selected **{n_selected}/{len(lasso.coef_)}** features (true relevant: {n_relevant})")

# ===== TAB 6: Categorical Variables =====
with tab6:
    st.markdown("## Categorical Variables and Dummy Encoding")
    
    st.markdown("""
    **How to include categorical variables in regression:**
    - Convert categories to numerical "dummy" variables (0/1)
    - Use **k-1** dummies for k categories (one is baseline)
    - Coefficients represent difference from baseline category
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎲 Generate Data")
        
        n_samples = st.slider("Number of samples", 50, 300, 100, 10, key='cat_n')
        noise = st.slider("Noise level", 0.1, 2.0, 0.5, 0.1, key='cat_noise')
        
        if st.button("Generate & Fit", type="primary", key='cat_btn'):
            np.random.seed(42)
            
            # Create categorical variable: A, B, C (e.g., product types)
            categories = np.random.choice(['A', 'B', 'C'], size=n_samples)
            
            # Create continuous predictor
            x_cont = np.random.randn(n_samples)
            
            # True effects:
            # Baseline (A): intercept = 5
            # B: +3 more than A
            # C: -2 less than A
            # x_cont: slope = 2
            
            y = 5 + 2 * x_cont + np.where(categories == 'B', 3, 0) + \
                np.where(categories == 'C', -2, 0) + np.random.randn(n_samples) * noise
            
            # Create DataFrame
            df = pd.DataFrame({
                'Y': y,
                'X_continuous': x_cont,
                'Category': categories
            })
            
            # Dummy encoding
            df_encoded = pd.get_dummies(df, columns=['Category'], drop_first=True, dtype=float)
            
            # Fit model
            from sklearn.linear_model import LinearRegression
            X = df_encoded.drop('Y', axis=1)
            y_vals = df_encoded['Y']
            
            model = LinearRegression()
            model.fit(X, y_vals)
            
            st.session_state.cat_df = df
            st.session_state.cat_df_encoded = df_encoded
            st.session_state.cat_model = model
            st.session_state.cat_coefs = dict(zip(X.columns, model.coef_))
            st.session_state.cat_intercept = model.intercept_
            
            st.markdown("### 📊 Model Coefficients")
            st.metric("Intercept (Category A baseline)", f"{model.intercept_:.3f}")
            for col, coef in zip(X.columns, model.coef_):
                st.metric(col, f"{coef:.3f}")
    
    with col2:
        if 'cat_df' in st.session_state:
            df = st.session_state.cat_df
            model = st.session_state.cat_model
            
            # Show dummy encoding
            st.markdown("### Dummy Encoding Example")
            st.dataframe(st.session_state.cat_df_encoded.head(10), height=300)
            
            # Visualization: Compare predictions by category
            fig = go.Figure()
            
            categories = ['A', 'B', 'C']
            colors = {'A': 'blue', 'B': 'green', 'C': 'red'}
            
            for cat in categories:
                mask = df['Category'] == cat
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, 'X_continuous'],
                    y=df.loc[mask, 'Y'],
                    mode='markers',
                    name=f'Category {cat}',
                    marker=dict(color=colors[cat], size=8, opacity=0.6)
                ))
                
                # Fitted lines
                x_range = np.linspace(df['X_continuous'].min(), df['X_continuous'].max(), 50)
                
                # Predict for this category
                if cat == 'A':  # Baseline
                    y_pred = st.session_state.cat_intercept + \
                            st.session_state.cat_coefs['X_continuous'] * x_range
                elif cat == 'B':
                    y_pred = st.session_state.cat_intercept + \
                            st.session_state.cat_coefs['X_continuous'] * x_range + \
                            st.session_state.cat_coefs['Category_B']
                else:  # C
                    y_pred = st.session_state.cat_intercept + \
                            st.session_state.cat_coefs['X_continuous'] * x_range + \
                            st.session_state.cat_coefs['Category_C']
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_pred,
                    mode='lines',
                    name=f'Fit {cat}',
                    line=dict(color=colors[cat], width=3),
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Parallel Regression Lines (Different Intercepts)",
                xaxis_title="X (Continuous)",
                yaxis_title="Y",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - All lines have the **same slope** (effect of X_continuous)
            - Lines are **parallel** with different intercepts
            - `Category_B` coefficient: how much higher than A
            - `Category_C` coefficient: how much lower/higher than A
            
            **Formula:** $y = \\beta_0 + \\beta_1 X_{cont} + \\beta_2 D_B + \\beta_3 D_C$
            
            where $D_B = 1$ if category B, else 0 (and similarly for $D_C$)
            """)

# ===== TAB 7: Multicollinearity & VIF =====
with tab7:
    st.markdown("## Multicollinearity Detection with VIF")
    
    st.markdown("""
    **Multicollinearity:** When predictors are highly correlated with each other
    
    **Problems:**
    - Unstable coefficient estimates
    - Large standard errors
    - Difficult interpretation ("holding others fixed" doesn't make sense)
    
    **Variance Inflation Factor (VIF):**
    $$VIF_j = \\frac{1}{1 - R^2_j}$$
    
    where $R^2_j$ is the R² from regressing $X_j$ on all other predictors.
    
    **Rule of thumb:**
    - VIF < 5: No problem
    - VIF 5-10: Moderate multicollinearity
    - VIF > 10: Severe multicollinearity (problematic!)
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Data Settings")
        
        n_samples = st.slider("Sample size", 100, 500, 200, 50, key='vif_n')
        correlation = st.slider("Feature correlation", 0.0, 0.98, 0.8, 0.02, key='vif_corr')
        n_features = st.slider("Number of features", 3, 10, 5, 1, key='vif_p')
        
        if st.button("Analyze Multicollinearity", type="primary", key='vif_btn'):
            np.random.seed(42)
            
            # Generate correlated features
            X, y, _ = generate_correlated_data(
                n=n_samples, p=n_features, correlation=correlation, random_state=42
            )
            
            # Calculate VIF for each feature
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            vif_data = pd.DataFrame()
            vif_data["Feature"] = [f"X{i+1}" for i in range(X.shape[1])]
            vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            
            st.session_state.vif_data = vif_data
            st.session_state.vif_X = X
            st.session_state.vif_corr_set = correlation
            
            st.markdown("### 📊 VIF Values")
            
            for _, row in vif_data.iterrows():
                vif_val = row['VIF']
                if vif_val < 5:
                    status = "🟢 Good"
                    color = "normal"
                elif vif_val < 10:
                    status = "🟡 Moderate"
                    color = "normal"
                else:
                    status = "🔴 Severe"
                    color = "inverse"
                
                st.metric(row['Feature'], f"{vif_val:.2f}", status, delta_color=color)
    
    with col2:
        if 'vif_data' in st.session_state:
            vif_data = st.session_state.vif_data
            X = st.session_state.vif_X
            
            # VIF Bar chart
            fig1 = go.Figure()
            
            colors = ['green' if v < 5 else 'orange' if v < 10 else 'red' 
                     for v in vif_data['VIF']]
            
            fig1.add_trace(go.Bar(
                x=vif_data['Feature'],
                y=vif_data['VIF'],
                marker_color=colors,
                text=vif_data['VIF'].round(2),
                textposition='outside'
            ))
            
            fig1.add_hline(y=5, line_dash="dash", line_color="orange",
                          annotation_text="VIF = 5 (moderate threshold)")
            fig1.add_hline(y=10, line_dash="dash", line_color="red",
                          annotation_text="VIF = 10 (severe threshold)")
            
            fig1.update_layout(
                title="Variance Inflation Factors",
                xaxis_title="Feature",
                yaxis_title="VIF",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Correlation matrix
            st.markdown("### Correlation Matrix")
            
            corr_matrix = np.corrcoef(X.T)
            
            fig2 = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=[f"X{i+1}" for i in range(X.shape[1])],
                y=[f"X{i+1}" for i in range(X.shape[1])],
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig2.update_layout(
                title="Feature Correlation Matrix",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("""
            **How to fix multicollinearity:**
            1. **Remove** highly correlated predictors
            2. **Combine** correlated features (e.g., PCA)
            3. Use **regularization** (Ridge, Lasso)
            4. Collect **more data** (doesn't always help)
            
            **Note:** Multicollinearity affects interpretation, not prediction!
            """)

with st.sidebar:
    st.markdown("### 📖 About Regularization")
    st.info("""
    **Regularization** adds penalty to loss function:
    
    **Ridge (L2):**
    - Penalty: λ∑θ²
    - Shrinks coefficients smoothly
    - Keeps all features
    
    **Lasso (L1):**
    - Penalty: λ∑|θ|
    - Can zero out coefficients
    - Feature selection
    
    **ElasticNet:**
    - Combination of L1 and L2
    - Balanced approach
    """)
