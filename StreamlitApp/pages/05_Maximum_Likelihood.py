"""
Maximum Likelihood Estimation
=============================
Interactive MLE demonstrations for different distributions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize

st.set_page_config(page_title="Maximum Likelihood", page_icon="📊", layout="wide")

st.title("📊 Maximum Likelihood Estimation")
st.markdown("### Finding Parameters that Maximize Data Likelihood")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 MLE for Normal Distribution",
    "🎲 Other Distributions",
    "🏔️ Likelihood Surface",
    "🆚 MLE vs Least Squares",
    "📊 Fisher Information & CI",
    "🔬 Advanced Distributions"
])

# ===== TAB 1: Normal Distribution =====
with tab1:
    st.markdown("## MLE for Normal Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎲 Generate Data")
        
        true_mu = st.slider("True μ", -5.0, 5.0, 1.0, 0.5)
        true_sigma = st.slider("True σ", 0.5, 5.0, 2.0, 0.5)
        n_samples = st.slider("Sample size", 10, 500, 100, 10)
        
        if st.button("Generate & Estimate", type="primary"):
            # Generate data
            np.random.seed(42)
            data = np.random.normal(true_mu, true_sigma, n_samples)
            
            # MLE estimates
            mu_mle = np.mean(data)
            sigma_mle = np.std(data, ddof=0)  # MLE uses N, not N-1
            
            st.session_state.normal_data = data
            st.session_state.mu_mle = mu_mle
            st.session_state.sigma_mle = sigma_mle
            st.session_state.true_mu = true_mu
            st.session_state.true_sigma = true_sigma
            
            st.markdown("### 📊 MLE Results")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("True μ", f"{true_mu:.3f}")
                st.metric("MLE μ̂", f"{mu_mle:.3f}")
            with col_b:
                st.metric("True σ", f"{true_sigma:.3f}")
                st.metric("MLE σ̂", f"{sigma_mle:.3f}")
    
    with col2:
        if 'normal_data' in st.session_state:
            data = st.session_state.normal_data
            mu_mle = st.session_state.mu_mle
            sigma_mle = st.session_state.sigma_mle
            true_mu = st.session_state.true_mu
            true_sigma = st.session_state.true_sigma
            
            # Plot histogram with fitted distribution
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=data,
                name='Data',
                histnorm='probability density',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # True distribution
            x_range = np.linspace(data.min() - 1, data.max() + 1, 200)
            true_pdf = stats.norm.pdf(x_range, true_mu, true_sigma)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=true_pdf,
                mode='lines',
                name='True Distribution',
                line=dict(color='green', width=3, dash='dash')
            ))
            
            # MLE distribution
            mle_pdf = stats.norm.pdf(x_range, mu_mle, sigma_mle)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=mle_pdf,
                mode='lines',
                name='MLE Fit',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Data Histogram with True and MLE Distributions",
                xaxis_title="Value",
                yaxis_title="Density",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show log-likelihood
            log_likelihood = np.sum(stats.norm.logpdf(data, mu_mle, sigma_mle))
            st.info(f"💡 **Log-Likelihood:** {log_likelihood:.2f}")
            
            st.markdown("""
            **MLE formulas for Normal distribution:**
            
            $$\\hat{\\mu} = \\frac{1}{n}\\sum_{i=1}^n x_i$$
            
            $$\\hat{\\sigma}^2 = \\frac{1}{n}\\sum_{i=1}^n (x_i - \\hat{\\mu})^2$$
            """)

# ===== TAB 2: Other Distributions =====
with tab2:
    st.markdown("## MLE for Different Distributions")
    
    distribution = st.selectbox(
        "Select distribution",
        ["Exponential", "Poisson", "Bernoulli", "Uniform"]
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if distribution == "Exponential":
            true_lambda = st.slider("True λ (rate)", 0.1, 5.0, 1.0, 0.1)
            n_samples = st.slider("Sample size", 20, 500, 100, 10, key='exp_n')
            
            if st.button("Estimate", type="primary", key='exp_btn'):
                data = np.random.exponential(1/true_lambda, n_samples)
                lambda_mle = 1 / np.mean(data)
                
                st.metric("True λ", f"{true_lambda:.3f}")
                st.metric("MLE λ̂", f"{lambda_mle:.3f}")
                
                st.session_state.exp_data = data
                st.session_state.exp_mle = lambda_mle
                st.session_state.exp_true = true_lambda
        
        elif distribution == "Poisson":
            true_lambda = st.slider("True λ (mean)", 1, 20, 5, 1)
            n_samples = st.slider("Sample size", 20, 500, 100, 10, key='poi_n')
            
            if st.button("Estimate", type="primary", key='poi_btn'):
                data = np.random.poisson(true_lambda, n_samples)
                lambda_mle = np.mean(data)
                
                st.metric("True λ", f"{true_lambda}")
                st.metric("MLE λ̂", f"{lambda_mle:.3f}")
                
                st.session_state.poi_data = data
                st.session_state.poi_mle = lambda_mle
                st.session_state.poi_true = true_lambda
        
        elif distribution == "Bernoulli":
            true_p = st.slider("True p", 0.1, 0.9, 0.6, 0.05)
            n_samples = st.slider("Sample size", 20, 500, 100, 10, key='ber_n')
            
            if st.button("Estimate", type="primary", key='ber_btn'):
                data = np.random.binomial(1, true_p, n_samples)
                p_mle = np.mean(data)
                
                st.metric("True p", f"{true_p:.3f}")
                st.metric("MLE p̂", f"{p_mle:.3f}")
                
                st.session_state.ber_data = data
                st.session_state.ber_mle = p_mle
                st.session_state.ber_true = true_p
    
    with col2:
        if distribution == "Exponential" and 'exp_data' in st.session_state:
            data = st.session_state.exp_data
            lambda_mle = st.session_state.exp_mle
            true_lambda = st.session_state.exp_true
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data,
                name='Data',
                histnorm='probability density',
                marker_color='lightblue'
            ))
            
            x = np.linspace(0, data.max(), 200)
            fig.add_trace(go.Scatter(
                x=x,
                y=stats.expon.pdf(x, scale=1/lambda_mle),
                mode='lines',
                name='MLE Fit',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(title="Exponential Distribution MLE", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.latex(r"\hat{\lambda} = \frac{n}{\sum_{i=1}^n x_i}")
        
        elif distribution == "Poisson" and 'poi_data' in st.session_state:
            data = st.session_state.poi_data
            lambda_mle = st.session_state.poi_mle
            
            counts = np.bincount(data)
            x = np.arange(len(counts))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x,
                y=counts / len(data),
                name='Data',
                marker_color='lightblue'
            ))
            
            x_full = np.arange(0, data.max() + 1)
            fig.add_trace(go.Scatter(
                x=x_full,
                y=stats.poisson.pmf(x_full, lambda_mle),
                mode='lines+markers',
                name='MLE Fit',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(title="Poisson Distribution MLE", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.latex(r"\hat{\lambda} = \frac{1}{n}\sum_{i=1}^n x_i")

# ===== TAB 3: Likelihood Surface =====
with tab3:
    st.markdown("## Visualizing the Likelihood Surface")
    
    st.markdown("For Normal distribution with 2 parameters (μ, σ), we can visualize the likelihood surface")
    
    if st.button("Create Likelihood Surface", type="primary"):
        # Generate data
        np.random.seed(42)
        data = np.random.normal(2.0, 1.5, 50)
        
        # Grid for surface
        mu_range = np.linspace(0, 4, 50)
        sigma_range = np.linspace(0.5, 3, 50)
        MU, SIGMA = np.meshgrid(mu_range, sigma_range)
        
        # Compute log-likelihood
        LL = np.zeros_like(MU)
        for i in range(MU.shape[0]):
            for j in range(MU.shape[1]):
                LL[i, j] = np.sum(stats.norm.logpdf(data, MU[i, j], SIGMA[i, j]))
        
        # MLE
        mu_mle = np.mean(data)
        sigma_mle = np.std(data, ddof=0)
        ll_mle = np.sum(stats.norm.logpdf(data, mu_mle, sigma_mle))
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            x=mu_range,
            y=sigma_range,
            z=LL,
            colorscale='Viridis',
            name='Log-Likelihood'
        )])
        
        # Add MLE point
        fig.add_trace(go.Scatter3d(
            x=[mu_mle],
            y=[sigma_mle],
            z=[ll_mle],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='MLE'
        ))
        
        fig.update_layout(
            title="Log-Likelihood Surface",
            scene=dict(
                xaxis_title="μ",
                yaxis_title="σ",
                zaxis_title="Log-Likelihood"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"✅ MLE: μ̂ = {mu_mle:.3f}, σ̂ = {sigma_mle:.3f}")

# ===== TAB 4: MLE vs Least Squares =====
with tab4:
    st.markdown("## MLE vs Least Squares for Linear Regression")
    
    st.markdown("""
    For linear regression with Gaussian errors, **MLE is equivalent to OLS**!
    
    Assuming $y_i = \\mathbf{x}_i^T \\theta + \\epsilon_i$ where $\\epsilon_i \\sim N(0, \\sigma^2)$
    """)
    
    if st.button("Demonstrate Equivalence", type="primary"):
        # Generate data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        theta_true = np.array([2.0, -1.0])
        y = X @ theta_true + np.random.randn(100) * 0.5
        
        # OLS solution
        theta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # MLE solution (by maximizing likelihood)
        def neg_log_likelihood(params):
            theta = params[:2]
            sigma = params[2]
            if sigma <= 0:
                return 1e10
            residuals = y - X @ theta
            return -np.sum(stats.norm.logpdf(residuals, 0, sigma))
        
        # Optimize
        result = minimize(neg_log_likelihood, [0, 0, 1], method='L-BFGS-B',
                         bounds=[(-10, 10), (-10, 10), (0.01, 10)])
        theta_mle = result.x[:2]
        sigma_mle = result.x[2]
        
        # Compare
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### OLS Solution")
            st.write("θ₁:", f"{theta_ols[0]:.6f}")
            st.write("θ₂:", f"{theta_ols[1]:.6f}")
        
        with col2:
            st.markdown("### MLE Solution")
            st.write("θ₁:", f"{theta_mle[0]:.6f}")
            st.write("θ₂:", f"{theta_mle[1]:.6f}")
            st.write("σ̂:", f"{sigma_mle:.6f}")
        
        diff = np.linalg.norm(theta_ols - theta_mle)
        st.success(f"✅ **Difference:** {diff:.10f} (essentially identical!)")
        
        st.markdown("""
        **Why are they equivalent?**
        
        Maximizing likelihood:
        $$\\max_\\theta \\prod_{i=1}^n \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{(y_i - \\mathbf{x}_i^T\\theta)^2}{2\\sigma^2}\\right)$$
        
        Is equivalent to minimizing:
        $$\\min_\\theta \\sum_{i=1}^n (y_i - \\mathbf{x}_i^T\\theta)^2$$
        
        Which is exactly the OLS objective!
        """)

with st.sidebar:
    st.markdown("### 📖 About MLE")
    st.info("""
    **Maximum Likelihood Estimation:**
    
    Find parameters θ that maximize:
    
    $$L(\\theta) = \\prod_{i=1}^n f(x_i | \\theta)$$
    
    Usually optimize log-likelihood:
    
    $$\\ell(\\theta) = \\sum_{i=1}^n \\log f(x_i | \\theta)$$
    
    **Properties:**
    - Consistent
    - Asymptotically efficient
    - Invariant to reparameterization
    """)

# ===== TAB 5: Fisher Information & Confidence Intervals =====
with tab5:
    st.markdown("## Fisher Information and Confidence Intervals")
    
    st.markdown("""
    The **Fisher Information** measures how much information the data contains about the parameter.
    Higher Fisher Information → More precise estimates → Narrower confidence intervals
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎲 Setup")
        
        true_mu = st.slider("True μ", -3.0, 3.0, 1.0, 0.5, key='fi_mu')
        true_sigma = st.slider("True σ", 0.5, 3.0, 1.0, 0.5, key='fi_sigma')
        n_samples = st.slider("Sample size", 20, 500, 100, 20, key='fi_n')
        confidence_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
        
        if st.button("Estimate with CI", type="primary", key='fi_btn'):
            np.random.seed(42)
            data = np.random.normal(true_mu, true_sigma, n_samples)
            
            # MLE
            mu_mle = np.mean(data)
            sigma_mle = np.std(data, ddof=0)
            
            # Fisher Information for Normal distribution
            # I(μ) = n/σ²,  I(σ²) = n/(2σ⁴)
            fisher_mu = n_samples / (sigma_mle ** 2)
            fisher_sigma2 = n_samples / (2 * sigma_mle ** 4)
            
            # Standard errors (inverse of square root of Fisher Information)
            se_mu = 1 / np.sqrt(fisher_mu)
            se_sigma = sigma_mle * np.sqrt(2 / n_samples)  # Approximate
            
            # Confidence intervals
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            ci_mu_lower = mu_mle - z_score * se_mu
            ci_mu_upper = mu_mle + z_score * se_mu
            ci_sigma_lower = max(0.01, sigma_mle - z_score * se_sigma)
            ci_sigma_upper = sigma_mle + z_score * se_sigma
            
            st.session_state.fi_data = data
            st.session_state.fi_results = {
                'mu_mle': mu_mle, 'sigma_mle': sigma_mle,
                'true_mu': true_mu, 'true_sigma': true_sigma,
                'fisher_mu': fisher_mu, 'fisher_sigma2': fisher_sigma2,
                'se_mu': se_mu, 'se_sigma': se_sigma,
                'ci_mu': (ci_mu_lower, ci_mu_upper),
                'ci_sigma': (ci_sigma_lower, ci_sigma_upper),
                'conf_level': confidence_level
            }
            
            st.markdown("### 📊 Results")
            st.metric("μ MLE", f"{mu_mle:.4f}")
            st.metric(f"{int(confidence_level*100)}% CI for μ", 
                     f"[{ci_mu_lower:.4f}, {ci_mu_upper:.4f}]")
            st.metric("Fisher Info I(μ)", f"{fisher_mu:.2f}")
            
            st.metric("σ MLE", f"{sigma_mle:.4f}")
            st.metric(f"{int(confidence_level*100)}% CI for σ", 
                     f"[{ci_sigma_lower:.4f}, {ci_sigma_upper:.4f}]")
    
    with col2:
        if 'fi_results' in st.session_state:
            res = st.session_state.fi_results
            
            # Plot confidence interval visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("μ Confidence Interval", "σ Confidence Interval")
            )
            
            # μ plot
            mu_range = np.linspace(res['ci_mu'][0] - 0.5, res['ci_mu'][1] + 0.5, 200)
            likelihood_mu = np.exp(-0.5 * res['fisher_mu'] * (mu_range - res['mu_mle'])**2)
            
            fig.add_trace(go.Scatter(
                x=mu_range, y=likelihood_mu,
                name='Likelihood',
                line=dict(color='blue', width=2),
                fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'
            ), row=1, col=1)
            
            # Add CI region
            ci_mask = (mu_range >= res['ci_mu'][0]) & (mu_range <= res['ci_mu'][1])
            fig.add_trace(go.Scatter(
                x=mu_range[ci_mask], y=likelihood_mu[ci_mask],
                name=f"{int(res['conf_level']*100)}% CI",
                fill='tozeroy', fillcolor='rgba(0,255,0,0.3)',
                line=dict(width=0)
            ), row=1, col=1)
            
            # True value
            fig.add_vline(x=res['true_mu'], line_dash="dash", line_color="red",
                         annotation_text="True μ", row=1, col=1)
            
            # σ plot
            sigma_range = np.linspace(max(0.1, res['ci_sigma'][0] - 0.5), res['ci_sigma'][1] + 0.5, 200)
            # Approximate likelihood for σ
            likelihood_sigma = np.exp(-0.5 * ((sigma_range - res['sigma_mle']) / res['se_sigma'])**2)
            
            fig.add_trace(go.Scatter(
                x=sigma_range, y=likelihood_sigma,
                name='Likelihood',
                line=dict(color='blue', width=2),
                fill='tozeroy', fillcolor='rgba(0,0,255,0.1)',
                showlegend=False
            ), row=1, col=2)
            
            # Add CI region
            ci_mask = (sigma_range >= res['ci_sigma'][0]) & (sigma_range <= res['ci_sigma'][1])
            fig.add_trace(go.Scatter(
                x=sigma_range[ci_mask], y=likelihood_sigma[ci_mask],
                fill='tozeroy', fillcolor='rgba(0,255,0,0.3)',
                line=dict(width=0),
                showlegend=False
            ), row=1, col=2)
            
            # True value
            fig.add_vline(x=res['true_sigma'], line_dash="dash", line_color="red",
                         row=1, col=2)
            
            fig.update_xaxes(title_text="μ", row=1, col=1)
            fig.update_xaxes(title_text="σ", row=1, col=2)
            fig.update_yaxes(title_text="Relative Likelihood", row=1, col=1)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Fisher Information Formulas (Normal Distribution):**
            
            $$I(\\mu) = \\frac{n}{\\sigma^2}, \\quad I(\\sigma^2) = \\frac{n}{2\\sigma^4}$$
            
            **Standard Error:** $SE = \\frac{1}{\\sqrt{I(\\theta)}}$
            
            **Confidence Interval:** $\\hat{\\theta} \\pm z_{\\alpha/2} \\cdot SE(\\hat{\\theta})$
            
            **Key Insight:** More data → Higher Fisher Information → Narrower CIs!
            """)

# ===== TAB 6: Advanced Distributions =====
with tab6:
    st.markdown("## Advanced Distributions")
    
    st.markdown("""
    Explore MLE for distributions beyond the basic ones:
    - **Student-t:** Heavy-tailed, robust to outliers
    - **Laplacian:** L1 loss equivalent (robust regression)
    - **Gamma:** Positive continuous data
    - **Beta:** Data bounded in [0,1]
    """)
    
    dist_choice = st.selectbox(
        "Select distribution",
        ["Student-t", "Laplacian", "Gamma", "Beta"]
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if dist_choice == "Student-t":
            st.markdown("### Student-t Distribution")
            st.markdown("Heavy tails → Robust to outliers")
            
            true_df = st.slider("True df (degrees of freedom)", 2, 30, 5, 1, key='t_df')
            true_loc = st.slider("True location (μ)", -3.0, 3.0, 0.0, 0.5, key='t_loc')
            true_scale = st.slider("True scale (σ)", 0.5, 3.0, 1.0, 0.5, key='t_scale')
            n_samples = st.slider("Sample size", 50, 500, 100, 50, key='t_n')
            
            if st.button("Estimate", type="primary", key='t_btn'):
                np.random.seed(42)
                data = stats.t.rvs(df=true_df, loc=true_loc, scale=true_scale, size=n_samples)
                
                # MLE using scipy
                params = stats.t.fit(data)
                df_mle, loc_mle, scale_mle = params
                
                st.session_state.t_data = data
                st.session_state.t_params = (df_mle, loc_mle, scale_mle)
                st.session_state.t_true = (true_df, true_loc, true_scale)
                
                st.metric("True df", f"{true_df}")
                st.metric("MLE df", f"{df_mle:.2f}")
                st.metric("MLE μ", f"{loc_mle:.4f}")
                st.metric("MLE σ", f"{scale_mle:.4f}")
        
        elif dist_choice == "Laplacian":
            st.markdown("### Laplacian Distribution")
            st.markdown("Equivalent to L1 loss (Least Absolute Deviations)")
            
            true_loc = st.slider("True location", -3.0, 3.0, 0.0, 0.5, key='lap_loc')
            true_scale = st.slider("True scale", 0.5, 3.0, 1.0, 0.5, key='lap_scale')
            n_samples = st.slider("Sample size", 50, 500, 100, 50, key='lap_n')
            
            if st.button("Estimate", type="primary", key='lap_btn'):
                np.random.seed(42)
                data = np.random.laplace(true_loc, true_scale, n_samples)
                
                # MLE: location = median, scale = mean absolute deviation
                loc_mle = np.median(data)
                scale_mle = np.mean(np.abs(data - loc_mle))
                
                st.session_state.lap_data = data
                st.session_state.lap_params = (loc_mle, scale_mle)
                st.session_state.lap_true = (true_loc, true_scale)
                
                st.metric("MLE location", f"{loc_mle:.4f}")
                st.metric("MLE scale", f"{scale_mle:.4f}")
        
        elif dist_choice == "Gamma":
            st.markdown("### Gamma Distribution")
            st.markdown("For positive continuous data (waiting times, etc.)")
            
            true_shape = st.slider("True shape (α)", 1.0, 10.0, 2.0, 0.5, key='gam_shape')
            true_scale = st.slider("True scale (β)", 0.5, 3.0, 1.0, 0.5, key='gam_scale')
            n_samples = st.slider("Sample size", 50, 500, 100, 50, key='gam_n')
            
            if st.button("Estimate", type="primary", key='gam_btn'):
                np.random.seed(42)
                data = np.random.gamma(true_shape, true_scale, n_samples)
                
                # MLE using method of moments as initial guess
                shape_mle, loc_mle, scale_mle = stats.gamma.fit(data, floc=0)
                
                st.session_state.gam_data = data
                st.session_state.gam_params = (shape_mle, scale_mle)
                st.session_state.gam_true = (true_shape, true_scale)
                
                st.metric("MLE shape (α)", f"{shape_mle:.4f}")
                st.metric("MLE scale (β)", f"{scale_mle:.4f}")
        
        elif dist_choice == "Beta":
            st.markdown("### Beta Distribution")
            st.markdown("For data bounded in [0, 1] (proportions, probabilities)")
            
            true_alpha = st.slider("True α", 0.5, 10.0, 2.0, 0.5, key='beta_a')
            true_beta = st.slider("True β", 0.5, 10.0, 5.0, 0.5, key='beta_b')
            n_samples = st.slider("Sample size", 50, 500, 100, 50, key='beta_n')
            
            if st.button("Estimate", type="primary", key='beta_btn'):
                np.random.seed(42)
                data = np.random.beta(true_alpha, true_beta, n_samples)
                
                # MLE
                alpha_mle, beta_mle, loc, scale = stats.beta.fit(data, floc=0, fscale=1)
                
                st.session_state.beta_data = data
                st.session_state.beta_params = (alpha_mle, beta_mle)
                st.session_state.beta_true = (true_alpha, true_beta)
                
                st.metric("MLE α", f"{alpha_mle:.4f}")
                st.metric("MLE β", f"{beta_mle:.4f}")
    
    with col2:
        if dist_choice == "Student-t" and 't_data' in st.session_state:
            data = st.session_state.t_data
            df_mle, loc_mle, scale_mle = st.session_state.t_params
            true_df, true_loc, true_scale = st.session_state.t_true
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data, histnorm='probability density',
                name='Data', marker_color='lightblue', opacity=0.7
            ))
            
            x_range = np.linspace(data.min() - 1, data.max() + 1, 200)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.t.pdf(x_range, df_mle, loc_mle, scale_mle),
                mode='lines', name='MLE Fit',
                line=dict(color='red', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.t.pdf(x_range, true_df, true_loc, true_scale),
                mode='lines', name='True Distribution',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Compare with Normal
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.norm.pdf(x_range, loc_mle, scale_mle),
                mode='lines', name='Normal (same μ, σ)',
                line=dict(color='orange', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title="Student-t vs Normal: Heavy Tails",
                xaxis_title="Value", yaxis_title="Density",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Student-t has heavier tails than Normal → More robust to outliers!**")
        
        elif dist_choice == "Laplacian" and 'lap_data' in st.session_state:
            data = st.session_state.lap_data
            loc_mle, scale_mle = st.session_state.lap_params
            true_loc, true_scale = st.session_state.lap_true
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data, histnorm='probability density',
                name='Data', marker_color='lightblue', opacity=0.7
            ))
            
            x_range = np.linspace(data.min() - 1, data.max() + 1, 200)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.laplace.pdf(x_range, loc_mle, scale_mle),
                mode='lines', name='MLE Fit',
                line=dict(color='red', width=3)
            ))
            
            # Compare with Normal
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.norm.pdf(x_range, np.mean(data), np.std(data)),
                mode='lines', name='Normal (for comparison)',
                line=dict(color='orange', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title="Laplacian Distribution (Double Exponential)",
                xaxis_title="Value", yaxis_title="Density",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Laplacian MLE = Least Absolute Deviations!**
            
            $$\\hat{\\mu} = \\text{median}(x), \\quad \\hat{b} = \\frac{1}{n}\\sum_{i=1}^n |x_i - \\hat{\\mu}|$$
            
            This is equivalent to minimizing $\\sum |y_i - \\hat{y}_i|$ (L1 loss)
            """)
        
        elif dist_choice == "Gamma" and 'gam_data' in st.session_state:
            data = st.session_state.gam_data
            shape_mle, scale_mle = st.session_state.gam_params
            true_shape, true_scale = st.session_state.gam_true
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data, histnorm='probability density',
                name='Data', marker_color='lightblue', opacity=0.7
            ))
            
            x_range = np.linspace(0, data.max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.gamma.pdf(x_range, shape_mle, scale=scale_mle),
                mode='lines', name='MLE Fit',
                line=dict(color='red', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.gamma.pdf(x_range, true_shape, scale=true_scale),
                mode='lines', name='True Distribution',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Gamma Distribution",
                xaxis_title="Value", yaxis_title="Density",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Gamma Distribution:**
            - Used for waiting times, survival analysis
            - Shape α controls skewness
            - Scale β stretches the distribution
            """)
        
        elif dist_choice == "Beta" and 'beta_data' in st.session_state:
            data = st.session_state.beta_data
            alpha_mle, beta_mle = st.session_state.beta_params
            true_alpha, true_beta = st.session_state.beta_true
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data, histnorm='probability density',
                name='Data', marker_color='lightblue', opacity=0.7
            ))
            
            x_range = np.linspace(0, 1, 200)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.beta.pdf(x_range, alpha_mle, beta_mle),
                mode='lines', name='MLE Fit',
                line=dict(color='red', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=stats.beta.pdf(x_range, true_alpha, true_beta),
                mode='lines', name='True Distribution',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Beta Distribution (Bounded [0,1])",
                xaxis_title="Value", yaxis_title="Density",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Beta Distribution:**
            - Perfect for proportions, percentages, probabilities
            - α and β control shape
            - Very flexible (uniform, U-shaped, bell-shaped possible)
            """)

