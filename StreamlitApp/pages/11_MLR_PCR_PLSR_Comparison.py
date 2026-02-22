"""
MLR vs PCR vs PLSR: Norwegian Renewable Energy Analysis
========================================================
Complete comparison of regression methods on real-world renewable energy prediction.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.utils import resample
from scipy import stats
from scipy.stats import f, chi2

st.set_page_config(page_title="MLR-PCR-PLSR Comparison", page_icon="⚡", layout="wide")

st.title("⚡ MLR vs PCR vs PLSR: Norwegian Renewable Energy")
st.markdown("### Understanding Multicollinearity and Dimensionality Reduction")

# Info box with learning objectives
with st.expander("🎯 Learning Objectives", expanded=True):
    st.markdown("""
    By exploring this demonstration, you will learn:
    
    **1. Multicollinearity and Its Consequences**
    - Identify multicollinearity in predictor variables using correlation matrices
    - Understand why highly correlated predictors cause unstable coefficient estimates
    - Demonstrate coefficient instability through bootstrap resampling
    
    **2. Principal Component Analysis (PCA)**
    - Apply PCA for dimensionality reduction and decorrelation
    - Interpret variance explained by principal components
    - Visualize data in reduced PC space using score plots
    
    **3. Principal Component Regression (PCR)**
    - Implement PCR by combining PCA with linear regression
    - Use cross-validation to select optimal number of components
    - Understand PCR's limitation: **PCA maximizes variance in X, not covariance with Y**
    
    **4. Partial Least Squares Regression (PLS)**
    - Understand how PLS differs: **PLS maximizes covariance between X and Y**
    - Interpret PLS components and Variable Importance in Projection (VIP) scores
    - Visualize correlation loadings with response variables
    
    **5. PLS2: Multi-Response Modeling**
    - Extend PLS to simultaneously model multiple response variables
    - Compare efficiency with separate PLS1 models
    
    **6. Model Validation and Comparison**
    - Use K-fold cross-validation for hyperparameter tuning
    - Evaluate final model performance on held-out test data
    - Compare MLR, PCR, PLS1, and PLS2 on the same problem
    """)

# Load data
@st.cache_data
def load_renewable_data():
    """Load Norwegian renewable energy dataset."""
    try:
        data_path = Path(__file__).parent.parent / "data"
        X_raw = pd.read_csv(data_path / "renewable_X.txt", sep=';', index_col=0)
        Y_raw = pd.read_csv(data_path / "renewable_Y.txt", sep=';', index_col=0)
        
        # Remove categorical variables
        X_df = X_raw.drop(columns=['Year', 'Month', 'IsDayBin'], errors='ignore')
        
        feature_names = list(X_df.columns)
        
        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df)
        
        y_wind = Y_raw['WindPower'].values
        y_pv = Y_raw['PVPower'].values
        
        # Train-test split
        X_train, X_test, y_wind_train, y_wind_test = train_test_split(
            X, y_wind, test_size=0.2, random_state=42
        )
        _, _, y_pv_train, y_pv_test = train_test_split(
            X, y_pv, test_size=0.2, random_state=42
        )
        
        return {
            'X': X, 'X_train': X_train, 'X_test': X_test,
            'y_wind': y_wind, 'y_wind_train': y_wind_train, 'y_wind_test': y_wind_test,
            'y_pv': y_pv, 'y_pv_train': y_pv_train, 'y_pv_test': y_pv_test,
            'feature_names': feature_names,
            'X_df': X_df, 'Y_raw': Y_raw
        }
    except Exception as e:
        st.error(f"⚠️ Could not load renewable energy dataset: {e}")
        return None

data = load_renewable_data()

if data is not None:
    # Extract data
    X_train = data['X_train']
    X_test = data['X_test']
    y_wind_train = data['y_wind_train']
    y_wind_test = data['y_wind_test']
    y_pv_train = data['y_pv_train']
    y_pv_test = data['y_pv_test']
    feature_names = data['feature_names']
    X_df = data['X_df']
    Y_raw = data['Y_raw']
    n_features = len(feature_names)
    
    # Color scheme
    BLUE = '#1f77b4'
    ORANGE = '#ff7f0e'
    GREEN = '#2ca02c'
    RED = '#d62728'
    PURPLE = '#9467bd'
    
    # Tabs
    tabs = st.tabs([
        "📊 Data Overview",
        "❌ Multicollinearity",
        "🔵 PCA & Score Plots",
        "📈 PCR Analysis",
        "🎯 PLS Analysis",
        "🎯🎯 PLS2 Multi-Response",
        "⚖️ Final Comparison"
    ])
    
    # ===== TAB 1: Data Overview =====
    with tabs[0]:
        st.markdown("## The Dataset")
        
        st.markdown("""
        Hourly measurements from a Norwegian weather station paired with energy production:
        
        **Predictor Matrix (X):** 11 weather variables
        - **Radiation:** IrrDirect, IrrDiffuse, RadSurface, RadTop
        - **Wind:** WindSpeed
        - **Atmosphere:** Temperature, AirDensity, CloudCover
        - **Precipitation:** Percipitation, SnowFlow, SnowMass
        
        **Response Matrix (Y):** Two target variables
        - **WindPower** — electrical output from wind turbines (MW)
        - **PVPower** — electrical output from solar panels (MW)
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        with col3:
            st.metric("Features", n_features)
        
        # Data preview
        st.markdown("### 📋 Data Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Predictors (First 10 rows)**")
            st.dataframe(X_df.head(10), width='stretch')
        
        with col2:
            st.markdown("**Responses (First 10 rows)**")
            st.dataframe(Y_raw.head(10), width='stretch')
        
        # Summary statistics
        st.markdown("### 📊 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Predictor Statistics**")
            st.dataframe(X_df.describe().round(2), width='stretch')
        
        with col2:
            st.markdown("**Response Statistics**")
            st.dataframe(Y_raw.describe().round(2), width='stretch')
        
        # Distribution plots
        st.markdown("### 📈 Response Distributions")
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['WindPower Distribution', 'PVPower Distribution'])
        
        fig.add_trace(go.Histogram(
            x=y_wind_train, nbinsx=30, name='WindPower',
            marker_color=BLUE, opacity=0.7
        ), row=1, col=1)
        
        fig.add_trace(go.Histogram(
            x=y_pv_train, nbinsx=30, name='PVPower',
            marker_color=ORANGE, opacity=0.7
        ), row=1, col=2)
        
        fig.update_xaxes(title_text='Power (MW)', row=1, col=1)
        fig.update_xaxes(title_text='Power (MW)', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        fig.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig, width='stretch')
    
    # ===== TAB 2: Multicollinearity & MLR =====
    with tabs[1]:
        st.markdown("## Step 1: Multiple Linear Regression (MLR)")
        st.markdown("""
        Weather variables are often highly correlated (e.g., different radiation measurements).
        Let's start with standard MLR and examine the multicollinearity problem.
        """)
        
        # First fit MLR models
        ols_wind = LinearRegression().fit(X_train, y_wind_train)
        ols_pv = LinearRegression().fit(X_train, y_pv_train)
        
        y_wind_pred_mlr = ols_wind.predict(X_train)
        y_pv_pred_mlr = ols_pv.predict(X_train)
        
        r2_wind_mlr = r2_score(y_wind_train, y_wind_pred_mlr)
        r2_pv_mlr = r2_score(y_pv_train, y_pv_pred_mlr)
        
        st.markdown("### 📊 MLR Training Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Wind R² (Training)", f"{r2_wind_mlr:.4f}")
        with col2:
            st.metric("PV R² (Training)", f"{r2_pv_mlr:.4f}")
        
        # Prediction plots
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=[f'WindPower (R²={r2_wind_mlr:.4f})',
                                          f'PVPower (R²={r2_pv_mlr:.4f})'])
        
        fig.add_trace(go.Scatter(
            x=y_wind_train, y=y_wind_pred_mlr,
            mode='markers', marker=dict(size=3, color=BLUE, opacity=0.5),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=y_pv_train, y=y_pv_pred_mlr,
            mode='markers', marker=dict(size=3, color=ORANGE, opacity=0.5),
            showlegend=False
        ), row=1, col=2)
        
        # Perfect prediction lines
        for col in [1, 2]:
            if col == 1:
                vmin, vmax = y_wind_train.min(), y_wind_train.max()
            else:
                vmin, vmax = y_pv_train.min(), y_pv_train.max()
            
            fig.add_trace(go.Scatter(
                x=[vmin, vmax], y=[vmin, vmax],
                mode='lines', line=dict(dash='dash', color='black', width=2),
                showlegend=False
            ), row=1, col=col)
        
        fig.update_xaxes(title_text='Actual (MW)', row=1, col=1)
        fig.update_yaxes(title_text='Predicted (MW)', row=1, col=1)
        fig.update_xaxes(title_text='Actual (MW)', row=1, col=2)
        fig.update_layout(height=500, title_text='MLR Training Predictions')
        
        st.plotly_chart(fig, width='stretch')
        
        st.info("✅ Both models show excellent R² values! But can we trust these coefficients?")
        
        # Store MLR models for later comparison
        st.session_state.mlr_models = {
            'ols_wind': ols_wind,
            'ols_pv': ols_pv,
            'r2_wind': r2_wind_mlr,
            'r2_pv': r2_pv_mlr
        }
        
        # Correlation heatmap
        st.markdown("### 🔥 Correlation Matrix")
        
        corr_matrix = np.corrcoef(X_train.T)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=feature_names,
            y=feature_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=600,
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'}
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Condition number
        cond_num = np.linalg.cond(X_train.T @ X_train)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Condition Number", f"{cond_num:.2e}",
                     help="Measures severity of multicollinearity. >30 is problematic.")
        with col2:
            max_corr = np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            st.metric("Max Correlation", f"{max_corr:.3f}")
        with col3:
            n_high_corr = np.sum(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]) > 0.8)
            st.metric("High Correlations (>0.8)", n_high_corr)
        
        st.warning(f"⚠️ Condition number of {cond_num:.0f} indicates severe multicollinearity!")
        
        # Bootstrap stability analysis
        st.markdown("### 🎲 Coefficient Stability Analysis")
        st.markdown("""
        High R² doesn't guarantee reliable coefficients! When predictors are correlated, 
        small data changes cause large coefficient swings. Let's test with bootstrap resampling:
        """)
        
        if st.button("Run Bootstrap Analysis", type="primary"):
            with st.spinner("Running 200 bootstrap iterations..."):
                n_bootstrap = 200
                n_sub = 2000
                coef_wind_bootstrap = []
                coef_pv_bootstrap = []
                
                for b in range(n_bootstrap):
                    idx = resample(np.arange(len(y_wind_train)), n_samples=n_sub, random_state=b)
                    coef_wind_bootstrap.append(LinearRegression().fit(X_train[idx], y_wind_train[idx]).coef_)
                    coef_pv_bootstrap.append(LinearRegression().fit(X_train[idx], y_pv_train[idx]).coef_)
                
                coef_wind_bootstrap = np.array(coef_wind_bootstrap)
                coef_pv_bootstrap = np.array(coef_pv_bootstrap)
                
                # Plot coefficient distributions
                fig = make_subplots(rows=1, cols=2,
                                   subplot_titles=['MLR Coefficients — WindPower', 
                                                  'MLR Coefficients — PVPower'])
                
                for i, name in enumerate(feature_names):
                    fig.add_trace(go.Box(
                        y=coef_wind_bootstrap[:, i],
                        name=name,
                        marker_color=BLUE,
                        showlegend=False,
                        boxpoints=False
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Box(
                        y=coef_pv_bootstrap[:, i],
                        name=name,
                        marker_color=ORANGE,
                        showlegend=False,
                        boxpoints=False
                    ), row=1, col=2)
                
                fig.add_hline(y=0, line_dash='dash', line_color='black', row=1, col=1)
                fig.add_hline(y=0, line_dash='dash', line_color='black', row=1, col=2)
                fig.update_xaxes(tickangle=-45, row=1, col=1)
                fig.update_xaxes(tickangle=-45, row=1, col=2)
                fig.update_yaxes(title_text='Coefficient Value', row=1, col=1)
                fig.update_layout(height=500, width=1200)
                
                st.plotly_chart(fig, width='stretch')
                
                # Compute coefficient standard deviations
                coef_std_wind = np.std(coef_wind_bootstrap, axis=0)
                coef_mean_wind = np.mean(coef_wind_bootstrap, axis=0)
                
                stability_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Mean Coefficient': coef_mean_wind,
                    'Std Deviation': coef_std_wind,
                    'Relative Std %': (coef_std_wind / (np.abs(coef_mean_wind) + 1e-10)) * 100
                }).sort_values('Std Deviation', ascending=False, key=abs)
                
                st.markdown("**Coefficient Stability Metrics (WindPower):**")
                st.dataframe(stability_df.round(3), width='stretch')
                
                st.error("""
                **❌ Problem Identified:**
                - Coefficients vary dramatically across bootstrap samples
                - Some coefficients even change sign!
                - This makes the model **uninterpretable and unreliable**
                - Solution needed: PCR or PLS to handle multicollinearity
                """)
    
    # ===== TAB 3: PCA & Score Plots =====
    with tabs[2]:
        st.markdown("## Principal Component Analysis")
        st.markdown("""
        PCA transforms correlated predictors into uncorrelated principal components (PCs).
        Each PC is a linear combination of original features that captures maximum variance.
        """)
        
        # Fit PCA
        pca = PCA()
        T = pca.fit_transform(X_train)
        P = pca.components_.T  # Loadings
        explained_var = pca.explained_variance_ratio_
        
        # Scree plot with cumulative variance
        st.markdown("### 📊 Scree Plot: Variance Explained")
        
        cumulative_var = np.cumsum(explained_var)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Scree Plot: Variance per PC', 'Cumulative Variance Explained']
        )
        
        # Individual variance
        fig.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(n_features)],
            y=explained_var * 100,
            marker_color=BLUE,
            name='Variance',
            showlegend=False
        ), row=1, col=1)
        
        # Cumulative variance
        fig.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(n_features)],
            y=cumulative_var * 100,
            mode='lines+markers',
            marker=dict(size=8, color=ORANGE),
            line=dict(width=3, color=ORANGE),
            name='Cumulative',
            showlegend=False
        ), row=1, col=2)
        
        fig.add_hline(y=90, line_dash='dash', line_color='red', 
                      annotation_text='90% threshold', row=1, col=2)
        
        fig.update_yaxes(title_text='Variance Explained (%)', row=1, col=1)
        fig.update_yaxes(title_text='Cumulative Variance (%)', row=1, col=2)
        fig.update_xaxes(tickangle=-45)
        fig.update_layout(height=450, width=1200)
        
        st.plotly_chart(fig, width='stretch')
        
        # Find number of PCs for 90% variance
        n_90 = np.argmax(cumulative_var >= 0.90) + 1
        st.info(f"📌 First **{n_90} PCs** explain 90% of variance | First **3 PCs** explain {cumulative_var[2]*100:.1f}%")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PC1 Variance", f"{explained_var[0]*100:.1f}%")
        with col2:
            st.metric("PC1-3 Cumulative", f"{np.sum(explained_var[:3])*100:.1f}%")
        with col3:
            st.metric("PC1-4 Cumulative", f"{np.sum(explained_var[:4])*100:.1f}%")
        
        # Loadings plot
        st.markdown("### 🎯 PC Loadings (First 3 PCs)")
        st.markdown("Shows how each feature contributes to the principal components")
        
        fig = go.Figure()
        
        colors_pcs = [BLUE, ORANGE, GREEN]
        for pc_idx in range(3):
            fig.add_trace(go.Bar(
                name=f'PC{pc_idx+1} ({explained_var[pc_idx]*100:.1f}%)',
                x=feature_names,
                y=P[:, pc_idx],
                marker_color=colors_pcs[pc_idx],
                opacity=0.8
            ))
        
        fig.update_layout(
            title='PCA Loadings: Which features contribute to each PC?',
            xaxis_title='Features',
            yaxis_title='Loading',
            barmode='group',
            height=500,
            width=1200,
            xaxis_tickangle=-45
        )
        fig.add_hline(y=0, line_dash='dash', line_color='black')
        
        st.plotly_chart(fig, width='stretch')
        
        # Print interpretation
        st.markdown("**🔍 Loading Interpretation:**")
        st.code(f"""
PC1 top features: {', '.join([f'{feature_names[i]}={P[i, 0]:.2f}' for i in np.argsort(np.abs(P[:, 0]))[-5:][::-1]])}
PC2 top features: {', '.join([f'{feature_names[i]}={P[i, 1]:.2f}' for i in np.argsort(np.abs(P[:, 1]))[-5:][::-1]])}
PC3 top features: {', '.join([f'{feature_names[i]}={P[i, 2]:.2f}' for i in np.argsort(np.abs(P[:, 2]))[-5:][::-1]])}
        """.strip())
        
        # Interactive score plot with independent dropdowns
        st.markdown("### 📍 Interactive Score Plot")
        st.markdown("Visualize data in principal component space, colored by response values")
        
        n_components = min(5, T.shape[1])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_pc = st.selectbox("X-axis PC", [f"PC{i+1}" for i in range(n_components)], index=0)
            x_pc_idx = int(x_pc[2:]) - 1
        with col2:
            y_pc = st.selectbox("Y-axis PC", [f"PC{i+1}" for i in range(n_components)], index=1)
            y_pc_idx = int(y_pc[2:]) - 1
        with col3:
            color_var = st.radio("Color by", ["WindPower", "PVPower"])
        
        # Create score plot
        color_values = y_wind_train if color_var == "WindPower" else y_pv_train
        colorscale = 'Viridis' if color_var == "WindPower" else 'Plasma'
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=T[:, x_pc_idx],
            y=T[:, y_pc_idx],
            mode='markers',
            marker=dict(
                size=5,
                color=color_values,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=f'{color_var}<br>(MW)')
            ),
            text=[f'{color_var}: {val:.2f} MW' for val in color_values],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Score Plot: {x_pc} vs {y_pc} (colored by {color_var})',
            xaxis_title=f'{x_pc} ({explained_var[x_pc_idx]*100:.1f}%)',
            yaxis_title=f'{y_pc} ({explained_var[y_pc_idx]*100:.1f}%)',
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.info("💡 The score plot shows how samples are distributed in PC space. Color intensity reveals relationships between PCs and response variables.")
    
    # ===== TAB 4: PCR Analysis =====
    with tabs[3]:
        st.markdown("## Principal Component Regression (PCR)")
        st.markdown("""
        **PCR Strategy:**
        1. Apply PCA to decorrelate predictors
        2. Regress response on selected principal components
        3. Use cross-validation to choose optimal number of components
        
        **Key Limitation:** PCA doesn't consider the response variable Y when creating components.
        """)
        
        # Cross-validation
        st.markdown("### 🔍 Cross-Validation: Selecting Optimal Components")
        
        if st.button("Run PCR Cross-Validation", type="primary"):
            with st.spinner("Running 10-fold cross-validation..."):
                kf = KFold(n_splits=10, shuffle=True, random_state=42)
                max_pcs = 4  # Limit to 4 PCs
                
                r2_wind_cv = []
                r2_pv_cv = []
                
                for k in range(1, max_pcs + 1):
                    scores_wind = []
                    scores_pv = []
                    
                    for train_idx, val_idx in kf.split(X_train):
                        pca_k = PCA(n_components=k)
                        T_train_fold = pca_k.fit_transform(X_train[train_idx])
                        T_val_fold = pca_k.transform(X_train[val_idx])
                        
                        lr_wind = LinearRegression().fit(T_train_fold, y_wind_train[train_idx])
                        scores_wind.append(lr_wind.score(T_val_fold, y_wind_train[val_idx]))
                        
                        lr_pv = LinearRegression().fit(T_train_fold, y_pv_train[train_idx])
                        scores_pv.append(lr_pv.score(T_val_fold, y_pv_train[val_idx]))
                    
                    r2_wind_cv.append(np.mean(scores_wind))
                    r2_pv_cv.append(np.mean(scores_pv))
                
                # Plot results
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, max_pcs + 1)),
                    y=r2_wind_cv,
                    mode='lines+markers',
                    name='WindPower',
                    marker=dict(size=10, color=BLUE),
                    line=dict(width=3, color=BLUE)
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, max_pcs + 1)),
                    y=r2_pv_cv,
                    mode='lines+markers',
                    name='PVPower',
                    marker=dict(size=10, color=ORANGE),
                    line=dict(width=3, color=ORANGE)
                ))
                
                fig.update_layout(
                    title='PCR Cross-Validation: R² vs Number of PCs',
                    xaxis_title='Number of Principal Components',
                    yaxis_title='R² (10-fold CV)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                k_opt_wind = np.argmax(r2_wind_cv) + 1
                k_opt_pv = np.argmax(r2_pv_cv) + 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Optimal PCs (Wind)", k_opt_wind, 
                             delta=f"R² = {max(r2_wind_cv):.4f}")
                with col2:
                    st.metric("Optimal PCs (PV)", k_opt_pv,
                             delta=f"R² = {max(r2_pv_cv):.4f}")
                
                st.session_state.pcr_results = {
                    'k_opt_wind': k_opt_wind,
                    'k_opt_pv': k_opt_pv,
                    'r2_wind_cv': r2_wind_cv,
                    'r2_pv_cv': r2_pv_cv
                }
                
                st.success("✅ PCR cross-validation complete!")
        
        # Train final models if CV done
        if 'pcr_results' in st.session_state:
            st.markdown("### 🎯 Final PCR Models")
            
            k_opt_wind = st.session_state.pcr_results['k_opt_wind']
            k_opt_pv = st.session_state.pcr_results['k_opt_pv']
            
            # Train final models
            pca_wind = PCA(n_components=k_opt_wind)
            T_wind_train = pca_wind.fit_transform(X_train)
            pcr_wind = LinearRegression().fit(T_wind_train, y_wind_train)
            
            pca_pv = PCA(n_components=k_opt_pv)
            T_pv_train = pca_pv.fit_transform(X_train)
            pcr_pv = LinearRegression().fit(T_pv_train, y_pv_train)
            
            # Training predictions
            y_wind_pred_train = pcr_wind.predict(T_wind_train)
            y_pv_pred_train = pcr_pv.predict(T_pv_train)
            
            r2_wind_train = r2_score(y_wind_train, y_wind_pred_train)
            r2_pv_train = r2_score(y_pv_train, y_pv_pred_train)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wind R² (Training)", f"{r2_wind_train:.4f}")
            with col2:
                st.metric("PV R² (Training)", f"{r2_pv_train:.4f}")
            
            # Prediction plots
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=[f'WindPower (R²={r2_wind_train:.4f})',
                                              f'PVPower (R²={r2_pv_train:.4f})'])
            
            fig.add_trace(go.Scatter(
                x=y_wind_train, y=y_wind_pred_train,
                mode='markers', marker=dict(size=3, color=BLUE, opacity=0.5),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=y_pv_train, y=y_pv_pred_train,
                mode='markers', marker=dict(size=3, color=ORANGE, opacity=0.5),
                showlegend=False
            ), row=1, col=2)
            
            # Perfect prediction lines
            for col in [1, 2]:
                if col == 1:
                    vmin, vmax = y_wind_train.min(), y_wind_train.max()
                else:
                    vmin, vmax = y_pv_train.min(), y_pv_train.max()
                
                fig.add_trace(go.Scatter(
                    x=[vmin, vmax], y=[vmin, vmax],
                    mode='lines', line=dict(dash='dash', color='black', width=2),
                    showlegend=False
                ), row=1, col=col)
            
            fig.update_xaxes(title_text='Actual (MW)', row=1, col=1)
            fig.update_yaxes(title_text='Predicted (MW)', row=1, col=1)
            fig.update_xaxes(title_text='Actual (MW)', row=1, col=2)
            fig.update_layout(height=500, title_text='PCR Training Predictions')
            
            st.plotly_chart(fig, width='stretch')
            
            st.session_state.pcr_models = {
                'pca_wind': pca_wind, 'pcr_wind': pcr_wind,
                'pca_pv': pca_pv, 'pcr_pv': pcr_pv,
                'r2_wind_train': r2_wind_train, 'r2_pv_train': r2_pv_train
            }
    
    # ===== TAB 5: PLS Analysis =====
    with tabs[4]:
        st.markdown("## Partial Least Squares Regression (PLS)")
        st.markdown("""
        **Key Difference from PCR:**
        - **PCR:** Finds components that maximize variance in X (unsupervised)
        - **PLS:** Finds components that maximize covariance between X and Y (supervised)
        
        PLS considers the response variable when creating latent components, often leading to better predictions with fewer components.
        """)
        
        # Cross-validation
        st.markdown("### 🔍 Cross-Validation for PLS1")
        
        if st.button("Run PLS Cross-Validation", type="primary"):
            with st.spinner("Running 10-fold cross-validation..."):
                cv = KFold(n_splits=10, shuffle=True, random_state=42)
                max_components = 4
                
                # WindPower PLS
                rmse_values_pls_wind = []
                for n_comp in range(1, max_components + 1):
                    cv_scores = []
                    for train_idx, val_idx in cv.split(X_train):
                        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        y_train_fold, y_val_fold = y_wind_train[train_idx], y_wind_train[val_idx]
                        
                        pls = PLSRegression(n_components=n_comp, scale=False)
                        pls.fit(X_train_fold, y_train_fold)
                        y_pred = pls.predict(X_val_fold).ravel()
                        cv_scores.append(mean_squared_error(y_val_fold, y_pred))
                    
                    rmse_values_pls_wind.append(np.sqrt(np.mean(cv_scores)))
                
                # PVPower PLS
                rmse_values_pls_pv = []
                for n_comp in range(1, max_components + 1):
                    cv_scores = []
                    for train_idx, val_idx in cv.split(X_train):
                        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        y_train_fold, y_val_fold = y_pv_train[train_idx], y_pv_train[val_idx]
                        
                        pls = PLSRegression(n_components=n_comp, scale=False)
                        pls.fit(X_train_fold, y_train_fold)
                        y_pred = pls.predict(X_val_fold).ravel()
                        cv_scores.append(mean_squared_error(y_val_fold, y_pred))
                    
                    rmse_values_pls_pv.append(np.sqrt(np.mean(cv_scores)))
                
                # Plot results
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, max_components + 1)),
                    y=rmse_values_pls_wind,
                    mode='lines+markers',
                    name='WindPower',
                    marker=dict(size=10, color=BLUE),
                    line=dict(width=3, color=BLUE)
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, max_components + 1)),
                    y=rmse_values_pls_pv,
                    mode='lines+markers',
                    name='PVPower',
                    marker=dict(size=10, color=ORANGE),
                    line=dict(width=3, color=ORANGE)
                ))
                
                # Mark optimal points
                optimal_k_wind = np.argmin(rmse_values_pls_wind) + 1
                optimal_k_pv = np.argmin(rmse_values_pls_pv) + 1
                
                fig.add_trace(go.Scatter(
                    x=[optimal_k_wind],
                    y=[rmse_values_pls_wind[optimal_k_wind-1]],
                    mode='markers',
                    marker=dict(size=15, color=BLUE, symbol='star'),
                    name=f'Optimal Wind (k={optimal_k_wind})',
                    showlegend=True
                ))
                
                fig.add_trace(go.Scatter(
                    x=[optimal_k_pv],
                    y=[rmse_values_pls_pv[optimal_k_pv-1]],
                    mode='markers',
                    marker=dict(size=15, color=ORANGE, symbol='star'),
                    name=f'Optimal PV (k={optimal_k_pv})',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title='PLS Cross-Validation: RMSE vs Number of Components',
                    xaxis_title='Number of Components',
                    yaxis_title='RMSE (10-fold CV)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Optimal Components (Wind)", optimal_k_wind,
                             delta=f"RMSE = {rmse_values_pls_wind[optimal_k_wind-1]:.4f}")
                with col2:
                    st.metric("Optimal Components (PV)", optimal_k_pv,
                             delta=f"RMSE = {rmse_values_pls_pv[optimal_k_pv-1]:.4f}")
                
                st.session_state.pls_results = {
                    'optimal_k_wind': optimal_k_wind,
                    'optimal_k_pv': optimal_k_pv,
                    'rmse_wind': rmse_values_pls_wind,
                    'rmse_pv': rmse_values_pls_pv
                }
                
                st.success("✅ PLS cross-validation complete!")
        
        # Train final PLS models
        if 'pls_results' in st.session_state:
            st.markdown("### 🎯 Final PLS1 Models")
            
            optimal_k_wind = st.session_state.pls_results['optimal_k_wind']
            optimal_k_pv = st.session_state.pls_results['optimal_k_pv']
            
            # Train models
            pls_wind = PLSRegression(n_components=optimal_k_wind, scale=False)
            pls_wind.fit(X_train, y_wind_train)
            y_wind_pred_train_pls = pls_wind.predict(X_train).ravel()
            
            pls_pv = PLSRegression(n_components=optimal_k_pv, scale=False)
            pls_pv.fit(X_train, y_pv_train)
            y_pv_pred_train_pls = pls_pv.predict(X_train).ravel()
            
            r2_wind_train_pls = r2_score(y_wind_train, y_wind_pred_train_pls)
            r2_pv_train_pls = r2_score(y_pv_train, y_pv_pred_train_pls)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wind R² (Training)", f"{r2_wind_train_pls:.4f}")
            with col2:
                st.metric("PV R² (Training)", f"{r2_pv_train_pls:.4f}")
            
            # Prediction plots
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=[f'WindPower (R²={r2_wind_train_pls:.4f})',
                                              f'PVPower (R²={r2_pv_train_pls:.4f})'])
            
            fig.add_trace(go.Scatter(
                x=y_wind_train, y=y_wind_pred_train_pls,
                mode='markers', marker=dict(size=3, color=BLUE, opacity=0.5),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=y_pv_train, y=y_pv_pred_train_pls,
                mode='markers', marker=dict(size=3, color=ORANGE, opacity=0.5),
                showlegend=False
            ), row=1, col=2)
            
            # Perfect prediction lines
            for col in [1, 2]:
                if col == 1:
                    vmin, vmax = y_wind_train.min(), y_wind_train.max()
                else:
                    vmin, vmax = y_pv_train.min(), y_pv_train.max()
                
                fig.add_trace(go.Scatter(
                    x=[vmin, vmax], y=[vmin, vmax],
                    mode='lines', line=dict(dash='dash', color='black', width=2),
                    showlegend=False
                ), row=1, col=col)
            
            fig.update_xaxes(title_text='Actual (MW)', row=1, col=1)
            fig.update_yaxes(title_text='Predicted (MW)', row=1, col=1)
            fig.update_xaxes(title_text='Actual (MW)', row=1, col=2)
            fig.update_layout(height=500, title_text='PLS1 Training Predictions')
            
            st.plotly_chart(fig, width='stretch')
            
            # VIP scores
            st.markdown("### 📊 Variable Importance in Projection (VIP)")
            st.markdown("VIP scores identify which features are most important for prediction (VIP > 1 = important)")
            
            # Calculate VIP for WindPower
            T = pls_wind.x_scores_
            W = pls_wind.x_weights_
            Q = pls_wind.y_loadings_
            
            p, h = W.shape
            vip_scores = np.zeros(p)
            
            s = np.sum(T**2, axis=0) * (Q.T**2).ravel()
            total_s = np.sum(s)
            
            for i in range(p):
                weight = np.sum(s * (W[i, :] ** 2))
                vip_scores[i] = np.sqrt(p * weight / total_s)
            
            # Plot VIP scores
            fig = go.Figure()
            
            colors = [GREEN if vip > 1 else BLUE for vip in vip_scores]
            
            fig.add_trace(go.Bar(
                x=feature_names,
                y=vip_scores,
                marker_color=colors,
                text=np.round(vip_scores, 2),
                textposition='outside'
            ))
            
            fig.add_hline(y=1, line_dash='dash', line_color='red',
                         annotation_text='VIP = 1 threshold')
            
            fig.update_layout(
                title='VIP Scores for WindPower Prediction',
                xaxis_title='Feature',
                yaxis_title='VIP Score',
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, width='stretch')
            
            st.info("💡 Features with VIP > 1 are considered important for the model. Green bars indicate significant features.")
            
            # PLS Score plots
            st.markdown("### 📍 PLS Score Plots")
            st.markdown("Visualizing data in the PLS latent space (colored by response values)")
            
            T_wind = pls_wind.x_scores_
            T_pv = pls_pv.x_scores_
            
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=['PLS Scores — WindPower', 'PLS Scores — PVPower'])
            
            # WindPower: T1 vs T2
            fig.add_trace(go.Scatter(
                x=T_wind[:, 0], y=T_wind[:, 1],
                mode='markers',
                marker=dict(size=4, color=y_wind_train, colorscale='Viridis',
                           showscale=True, colorbar=dict(title='WindPower<br>(MW)', x=0.46)),
                showlegend=False
            ), row=1, col=1)
            
            # PVPower: T1 vs T2
            fig.add_trace(go.Scatter(
                x=T_pv[:, 0], y=T_pv[:, 1],
                mode='markers',
                marker=dict(size=4, color=y_pv_train, colorscale='Plasma',
                           showscale=True, colorbar=dict(title='PVPower<br>(MW)', x=1.02)),
                showlegend=False
            ), row=1, col=2)
            
            fig.update_xaxes(title_text='T1 (1st PLS component)', row=1, col=1)
            fig.update_yaxes(title_text='T2 (2nd PLS component)', row=1, col=1)
            fig.update_xaxes(title_text='T1 (1st PLS component)', row=1, col=2)
            fig.update_yaxes(title_text='T2 (2nd PLS component)', row=1, col=2)
            
            fig.update_layout(height=500, width=1200)
            
            st.plotly_chart(fig, width='stretch')
            
            st.info("✅ PLS scores show data projected onto latent components. Colors reveal clear correlation between components and responses!")
            
            # Correlation loading plots
            st.markdown("### 🎯 Correlation Loading Plots (Correlation Circles)")
            st.markdown("Shows how features AND responses correlate with PLS components")
            
            # Compute correlation loadings
            def correlation_loadings(X_data, T):
                """Compute correlation between original variables and scores"""
                n_vars = X_data.shape[1]
                n_comps = T.shape[1]
                corr = np.zeros((n_vars, n_comps))
                for i in range(n_comps):
                    for j in range(n_vars):
                        corr[j, i] = np.corrcoef(X_data[:, j], T[:, i])[0, 1]
                return corr
            
            corr_load_wind = correlation_loadings(X_train, T_wind)
            corr_load_pv = correlation_loadings(X_train, T_pv)
            
            # Response correlations
            y_wind_corr_T1 = np.corrcoef(y_wind_train, T_wind[:, 0])[0, 1]
            y_wind_corr_T2 = np.corrcoef(y_wind_train, T_wind[:, 1])[0, 1] if T_wind.shape[1] > 1 else 0
            y_pv_corr_T1 = np.corrcoef(y_pv_train, T_pv[:, 0])[0, 1]
            y_pv_corr_T2 = np.corrcoef(y_pv_train, T_pv[:, 1])[0, 1] if T_pv.shape[1] > 1 else 0
            
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=['Correlation Loadings — WindPower', 
                                              'Correlation Loadings — PVPower'])
            
            # Add correlation circles (r=1 and r=0.7)
            theta = np.linspace(0, 2*np.pi, 100)
            for col in [1, 2]:
                fig.add_trace(go.Scatter(
                    x=np.cos(theta), y=np.sin(theta),
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=1),
                    showlegend=False
                ), row=1, col=col)
                
                fig.add_trace(go.Scatter(
                    x=0.7*np.cos(theta), y=0.7*np.sin(theta),
                    mode='lines',
                    line=dict(color='lightgray', dash='dot', width=1),
                    showlegend=False
                ), row=1, col=col)
            
            # WindPower loadings
            for i, feat in enumerate(feature_names):
                fig.add_trace(go.Scatter(
                    x=[0, corr_load_wind[i, 0]],
                    y=[0, corr_load_wind[i, 1] if corr_load_wind.shape[1] > 1 else 0],
                    mode='lines+markers+text',
                    line=dict(color=BLUE, width=2),
                    marker=dict(size=8, color=BLUE),
                    text=['', feat],
                    textposition='top center',
                    textfont=dict(size=9),
                    showlegend=False
                ), row=1, col=1)
            
            # PVPower loadings
            for i, feat in enumerate(feature_names):
                fig.add_trace(go.Scatter(
                    x=[0, corr_load_pv[i, 0]],
                    y=[0, corr_load_pv[i, 1] if corr_load_pv.shape[1] > 1 else 0],
                    mode='lines+markers+text',
                    line=dict(color=ORANGE, width=2),
                    marker=dict(size=8, color=ORANGE),
                    text=['', feat],
                    textposition='top center',
                    textfont=dict(size=9),
                    showlegend=False
                ), row=1, col=2)
            
            # Add response variables as red stars
            fig.add_trace(go.Scatter(
                x=[0, y_wind_corr_T1],
                y=[0, y_wind_corr_T2],
                mode='lines+markers+text',
                line=dict(color=RED, width=3),
                marker=dict(size=12, color=RED, symbol='star'),
                text=['', 'Y: WindPower'],
                textposition='top center',
                textfont=dict(size=11, color=RED),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[0, y_pv_corr_T1],
                y=[0, y_pv_corr_T2],
                mode='lines+markers+text',
                line=dict(color=RED, width=3),
                marker=dict(size=12, color=RED, symbol='star'),
                text=['', 'Y: PVPower'],
                textposition='top center',
                textfont=dict(size=11, color=RED),
                showlegend=False
            ), row=1, col=2)
            
            fig.update_xaxes(title_text='Correlation with T1', range=[-1.1, 1.1], row=1, col=1)
            fig.update_yaxes(title_text='Correlation with T2', range=[-1.1, 1.1], row=1, col=1)
            fig.update_xaxes(title_text='Correlation with T1', range=[-1.1, 1.1], row=1, col=2)
            fig.update_yaxes(title_text='Correlation with T2', range=[-1.1, 1.1], row=1, col=2)
            
            fig.update_layout(height=600, width=1200)
            
            st.plotly_chart(fig, width='stretch')
            
            st.markdown(f"""
            **🔍 Interpretation:**
            - **Blue/Orange arrows:** Predictor variables (X)
            - **Red stars:** Response variables (Y)
            - **Arrow length:** Correlation strength (closer to outer circle = stronger)
            - **Arrow angle:** Which component captures that feature/response
            
            **Response correlations:**
            - WindPower: r(Y,T1)={y_wind_corr_T1:.3f}, r(Y,T2)={y_wind_corr_T2:.3f}
            - PVPower: r(Y,T1)={y_pv_corr_T1:.3f}, r(Y,T2)={y_pv_corr_T2:.3f}
            """)
            
            st.session_state.pls_models = {
                'pls_wind': pls_wind, 'pls_pv': pls_pv,
                'r2_wind_train': r2_wind_train_pls,
                'r2_pv_train': r2_pv_train_pls
            }
    
    # ===== TAB 6: PLS2 Multi-Response =====
    with tabs[5]:
        st.markdown("## PLS2: Simultaneous Multi-Response Modeling")
        st.markdown("""
        **PLS2 Advantage:**
        - Models both WindPower and PVPower **simultaneously**
        - Finds common latent structure predicting both responses
        - More efficient than separate PLS1 models
        - Reveals shared patterns between responses
        """)
        
        if st.button("Run PLS2 Analysis", type="primary"):
            with st.spinner("Running PLS2 cross-validation..."):
                Y_both_train = np.column_stack([y_wind_train, y_pv_train])
                cv = KFold(n_splits=10, shuffle=True, random_state=42)
                max_components = 4
                
                pls2_results = {}
                for n_comp in range(1, max_components + 1):
                    cv_rmse_wind = []
                    cv_rmse_pv = []
                    
                    for train_idx, val_idx in cv.split(X_train):
                        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                        Y_train_fold, Y_val_fold = Y_both_train[train_idx], Y_both_train[val_idx]
                        
                        pls2 = PLSRegression(n_components=n_comp, scale=False)
                        pls2.fit(X_train_fold, Y_train_fold)
                        Y_pred = pls2.predict(X_val_fold)
                        
                        cv_rmse_wind.append(np.sqrt(mean_squared_error(Y_val_fold[:, 0], Y_pred[:, 0])))
                        cv_rmse_pv.append(np.sqrt(mean_squared_error(Y_val_fold[:, 1], Y_pred[:, 1])))
                    
                    pls2_results[n_comp] = {
                        'rmse_wind': np.mean(cv_rmse_wind),
                        'rmse_pv': np.mean(cv_rmse_pv)
                    }
                
                rmse_wind_pls2 = [pls2_results[k]['rmse_wind'] for k in range(1, max_components+1)]
                rmse_pv_pls2 = [pls2_results[k]['rmse_pv'] for k in range(1, max_components+1)]
                
                # Find optimal k
                total_rmse = [rmse_wind_pls2[i] + rmse_pv_pls2[i] for i in range(max_components)]
                k_opt_pls2 = np.argmin(total_rmse) + 1
                
                # Plot comparison with PLS1
                fig = go.Figure()
                
                if 'pls_results' in st.session_state:
                    rmse_wind_pls1 = st.session_state.pls_results['rmse_wind']
                    rmse_pv_pls1 = st.session_state.pls_results['rmse_pv']
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(1, max_components + 1)),
                        y=rmse_wind_pls1,
                        mode='lines',
                        name='WindPower (PLS1)',
                        line=dict(width=2, color=BLUE, dash='dash'),
                        opacity=0.5
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(1, max_components + 1)),
                        y=rmse_pv_pls1,
                        mode='lines',
                        name='PVPower (PLS1)',
                        line=dict(width=2, color=ORANGE, dash='dash'),
                        opacity=0.5
                    ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, max_components + 1)),
                    y=rmse_wind_pls2,
                    mode='lines+markers',
                    name='WindPower (PLS2)',
                    marker=dict(size=8, color=BLUE),
                    line=dict(width=3, color=BLUE, dash='dot')
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, max_components + 1)),
                    y=rmse_pv_pls2,
                    mode='lines+markers',
                    name='PVPower (PLS2)',
                    marker=dict(size=8, color=ORANGE),
                    line=dict(width=3, color=ORANGE, dash='dot')
                ))
                
                fig.update_layout(
                    title='PLS2 vs PLS1: Cross-Validation Comparison',
                    xaxis_title='Number of Components',
                    yaxis_title='CV RMSE',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Optimal Components", k_opt_pls2)
                with col2:
                    st.metric("Wind RMSE", f"{rmse_wind_pls2[k_opt_pls2-1]:.4f}")
                with col3:
                    st.metric("PV RMSE", f"{rmse_pv_pls2[k_opt_pls2-1]:.4f}")
                
                # Train final PLS2 model
                pls2_final = PLSRegression(n_components=k_opt_pls2, scale=False)
                pls2_final.fit(X_train, Y_both_train)
                Y_pred_train = pls2_final.predict(X_train)
                
                r2_wind_pls2 = r2_score(y_wind_train, Y_pred_train[:, 0])
                r2_pv_pls2 = r2_score(y_pv_train, Y_pred_train[:, 1])
                
                # Prediction plots
                fig = make_subplots(rows=1, cols=2,
                                   subplot_titles=[f'WindPower (R²={r2_wind_pls2:.4f})',
                                                  f'PVPower (R²={r2_pv_pls2:.4f})'])
                
                fig.add_trace(go.Scatter(
                    x=y_wind_train, y=Y_pred_train[:, 0],
                    mode='markers', marker=dict(size=3, color=GREEN, opacity=0.5),
                    showlegend=False
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=y_pv_train, y=Y_pred_train[:, 1],
                    mode='markers', marker=dict(size=3, color=PURPLE, opacity=0.5),
                    showlegend=False
                ), row=1, col=2)
                
                # Perfect prediction lines
                for col in [1, 2]:
                    if col == 1:
                        vmin, vmax = y_wind_train.min(), y_wind_train.max()
                    else:
                        vmin, vmax = y_pv_train.min(), y_pv_train.max()
                    
                    fig.add_trace(go.Scatter(
                        x=[vmin, vmax], y=[vmin, vmax],
                        mode='lines', line=dict(dash='dash', color='black', width=2),
                        showlegend=False
                    ), row=1, col=col)
                
                fig.update_xaxes(title_text='Actual (MW)', row=1, col=1)
                fig.update_yaxes(title_text='Predicted (MW)', row=1, col=1)
                fig.update_xaxes(title_text='Actual (MW)', row=1, col=2)
                fig.update_layout(height=500, title_text='PLS2 Training Predictions')
                
                st.plotly_chart(fig, width='stretch')
                
                st.session_state.pls2_model = {
                    'pls2': pls2_final,
                    'k_opt': k_opt_pls2,
                    'r2_wind': r2_wind_pls2,
                    'r2_pv': r2_pv_pls2
                }
                
                st.success("✅ PLS2 model trained successfully!")
    
    # ===== TAB 7: Final Comparison =====
    with tabs[6]:
        st.markdown("## Final Model Comparison on Test Set")
        st.markdown("""
        **Test Set Evaluation:**
        - Test data was held out during all model development
        - Provides unbiased estimate of generalization performance
        - Shows which method handles multicollinearity best
        """)
        
        if st.button("Evaluate All Models on Test Set", type="primary"):
            # Train MLR baseline
            ols_wind = LinearRegression().fit(X_train, y_wind_train)
            ols_pv = LinearRegression().fit(X_train, y_pv_train)
            
            y_wind_test_mlr = ols_wind.predict(X_test)
            y_pv_test_mlr = ols_pv.predict(X_test)
            
            r2_wind_test_mlr = r2_score(y_wind_test, y_wind_test_mlr)
            r2_pv_test_mlr = r2_score(y_pv_test, y_pv_test_mlr)
            
            results_data = []
            
            # MLR results
            results_data.append({
                'Method': 'MLR',
                'Components': 'All (11)',
                'Response': 'WindPower',
                'Train R²': r2_score(y_wind_train, ols_wind.predict(X_train)),
                'Test R²': r2_wind_test_mlr
            })
            results_data.append({
                'Method': 'MLR',
                'Components': 'All (11)',
                'Response': 'PVPower',
                'Train R²': r2_score(y_pv_train, ols_pv.predict(X_train)),
                'Test R²': r2_pv_test_mlr
            })
            
            # PCR results
            if 'pcr_models' in st.session_state:
                pca_wind = st.session_state.pcr_models['pca_wind']
                pcr_wind = st.session_state.pcr_models['pcr_wind']
                pca_pv = st.session_state.pcr_models['pca_pv']
                pcr_pv = st.session_state.pcr_models['pcr_pv']
                
                T_wind_test = pca_wind.transform(X_test)
                T_pv_test = pca_pv.transform(X_test)
                
                y_wind_test_pcr = pcr_wind.predict(T_wind_test)
                y_pv_test_pcr = pcr_pv.predict(T_pv_test)
                
                r2_wind_test_pcr = r2_score(y_wind_test, y_wind_test_pcr)
                r2_pv_test_pcr = r2_score(y_pv_test, y_pv_test_pcr)
                
                results_data.append({
                    'Method': 'PCR',
                    'Components': str(pca_wind.n_components),
                    'Response': 'WindPower',
                    'Train R²': st.session_state.pcr_models['r2_wind_train'],
                    'Test R²': r2_wind_test_pcr
                })
                results_data.append({
                    'Method': 'PCR',
                    'Components': str(pca_pv.n_components),
                    'Response': 'PVPower',
                    'Train R²': st.session_state.pcr_models['r2_pv_train'],
                    'Test R²': r2_pv_test_pcr
                })
            
            # PLS1 results
            if 'pls_models' in st.session_state:
                pls_wind = st.session_state.pls_models['pls_wind']
                pls_pv = st.session_state.pls_models['pls_pv']
                
                y_wind_test_pls1 = pls_wind.predict(X_test).ravel()
                y_pv_test_pls1 = pls_pv.predict(X_test).ravel()
                
                r2_wind_test_pls1 = r2_score(y_wind_test, y_wind_test_pls1)
                r2_pv_test_pls1 = r2_score(y_pv_test, y_pv_test_pls1)
                
                results_data.append({
                    'Method': 'PLS1',
                    'Components': str(pls_wind.n_components),
                    'Response': 'WindPower',
                    'Train R²': st.session_state.pls_models['r2_wind_train'],
                    'Test R²': r2_wind_test_pls1
                })
                results_data.append({
                    'Method': 'PLS1',
                    'Components': str(pls_pv.n_components),
                    'Response': 'PVPower',
                    'Train R²': st.session_state.pls_models['r2_pv_train'],
                    'Test R²': r2_pv_test_pls1
                })
            
            # PLS2 results
            if 'pls2_model' in st.session_state:
                pls2 = st.session_state.pls2_model['pls2']
                Y_test_pred = pls2.predict(X_test)
                
                r2_wind_test_pls2 = r2_score(y_wind_test, Y_test_pred[:, 0])
                r2_pv_test_pls2 = r2_score(y_pv_test, Y_test_pred[:, 1])
                
                results_data.append({
                    'Method': 'PLS2',
                    'Components': str(pls2.n_components),
                    'Response': 'WindPower',
                    'Train R²': st.session_state.pls2_model['r2_wind'],
                    'Test R²': r2_wind_test_pls2
                })
                results_data.append({
                    'Method': 'PLS2',
                    'Components': str(pls2.n_components),
                    'Response': 'PVPower',
                    'Train R²': st.session_state.pls2_model['r2_pv'],
                    'Test R²': r2_pv_test_pls2
                })
            
            # Display results table
            results_df = pd.DataFrame(results_data)
            
            st.markdown("### 📊 Comprehensive Results")
            st.dataframe(results_df.round(4), width='stretch', hide_index=True)
            
            # Visualization
            st.markdown("### 📈 Model Performance Comparison")
            
            fig = go.Figure()
            
            methods = results_df['Method'].unique()
            colors_map = {'MLR': RED, 'PCR': BLUE, 'PLS1': GREEN, 'PLS2': PURPLE}
            
            for method in methods:
                method_data = results_df[results_df['Method'] == method]
                
                fig.add_trace(go.Bar(
                    name=f'{method} - Wind',
                    x=[f"{method}\nWind"],
                    y=method_data[method_data['Response'] == 'WindPower']['Test R²'].values,
                    marker_color=colors_map.get(method, BLUE),
                    showlegend=True
                ))
                
                fig.add_trace(go.Bar(
                    name=f'{method} - PV',
                    x=[f"{method}\nPV"],
                    y=method_data[method_data['Response'] == 'PVPower']['Test R²'].values,
                    marker_color=colors_map.get(method, ORANGE),
                    opacity=0.7,
                    showlegend=True
                ))
            
            fig.update_layout(
                title='Test Set Performance: R² Comparison',
                xaxis_title='Method and Response',
                yaxis_title='R² Score',
                height=500,
                barmode='group'
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Key findings
            st.markdown("### 🎯 Key Findings")
            
            best_method_wind = results_df[results_df['Response'] == 'WindPower'].sort_values('Test R²', ascending=False).iloc[0]
            best_method_pv = results_df[results_df['Response'] == 'PVPower'].sort_values('Test R²', ascending=False).iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Best for WindPower:** {best_method_wind['Method']} (R² = {best_method_wind['Test R²']:.4f})")
            with col2:
                st.success(f"**Best for PVPower:** {best_method_pv['Method']} (R² = {best_method_pv['Test R²']:.4f})")
            
            st.info("""
            **Summary:**
            - **MLR suffers** from multicollinearity despite using all features
            - **PCR reduces dimensionality** but doesn't consider response variable
            - **PLS methods** achieve best performance by maximizing covariance with Y
            - **PLS2 efficiently** models both responses simultaneously
            - Using just **4 components** instead of 11 features demonstrates the power of dimensionality reduction!
            """)

else:
    st.error("⚠️ Could not load data. Please ensure renewable_X.txt and renewable_Y.txt are in the data folder.")
