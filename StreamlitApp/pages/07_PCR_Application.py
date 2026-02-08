"""
PCR Application - Norwegian Energy Data
========================================
Step-by-step investigation of Principal Component Regression
on real-world renewable energy forecasting data.

Based on PCR_Application.ipynb
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample

st.set_page_config(page_title="PCR Application", page_icon="⚡", layout="wide")

# ===== STYLING =====
BLUE   = '#4169E1'
RED    = '#DC143C'
GREEN  = '#228B22'
ORANGE = '#FF8C00'
TEAL   = '#4ECDC4'
PURPLE = '#800080'
GRAY   = '#888888'

# ===== LOAD DATA =====
@st.cache_data
def load_energy_data():
    """Load Norwegian renewable energy data"""
    data_path = Path(__file__).parent.parent / "data"
    
    X_raw = pd.read_csv(data_path / "X.txt", sep=';', index_col=0)
    Y_raw = pd.read_csv(data_path / "Y.txt", sep=';', index_col=0)
    
    # Encode categorical variable
    X_df = X_raw.copy()
    X_df['IsDayBin'] = (X_df['IsDayBin'] == 'Day').astype(int)
    
    feature_names = list(X_df.columns)
    n_features = len(feature_names)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    
    y_wind = Y_raw['WindPower'].values
    y_pv   = Y_raw['PVPower'].values
    
    return X, X_df, y_wind, y_pv, feature_names, n_features, Y_raw

X, X_df, y_wind, y_pv, feature_names, n_features, Y_raw = load_energy_data()

# ===== TITLE =====
st.title("⚡ Understanding PCR — Norwegian Renewable Energy")
st.markdown("### A Step-by-Step Investigation")

st.markdown("""
**The Problem:** Norway produces renewable energy from both **wind farms** and **photovoltaic (PV) solar panels**. 
Accurate forecasting of power output is essential for grid balancing and energy trading.

**The Dataset:** Hourly measurements from a Norwegian weather station with:
- **14 weather variables** (predictors)
- **2 responses:** WindPower and PVPower (MW)

**The Central Question:** Can we reduce dimensionality using PCA and still make accurate predictions?
""")

# ===== TABS =====
tabs = st.tabs([
    "📊 Data Overview",
    "1️⃣ MLR Baseline", 
    "2️⃣ PCA Structure",
    "3️⃣ PCR Performance",
    "4️⃣ PCA Diagnostics",
    "5️⃣ Feature Analysis",
    "6️⃣ PC Interpretation",
    "7️⃣ PC-Response Correlation",
    "8️⃣ R² vs k"
])

# ===== TAB 0: Data Overview =====
with tabs[0]:
    st.markdown("## Dataset Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Shape:** {X.shape[0]:,} samples × {X.shape[1]} features")
        st.markdown("**Features:**")
        st.write(feature_names)
        
    with col2:
        st.markdown("**Responses:** WindPower, PVPower")
        st.dataframe(Y_raw.describe().round(2), use_container_width=True)
    
    st.markdown("### Predictor Matrix (first 10 rows)")
    st.dataframe(X_df.head(10), use_container_width=True)
    
    st.markdown("### Correlation Matrix")
    corr_matrix = X_df.corr()
    
    fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, aspect='equal',
                    title='Correlation Matrix of Predictor Variables')
    fig.update_layout(width=800, height=750, coloraxis_colorbar_title='Pearson r',
                      title_font_size=16)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**Notice:** Strong correlation block among radiation variables (IrrDirect, IrrDiffuse, RadSurface, RadTop, IsDayBin, Temperature) — source of multicollinearity.")

# ===== TAB 1: MLR Baseline =====
with tabs[1]:
    st.markdown("## Step 1 — Multiple Linear Regression Baseline")
    st.markdown("""
    Start with ordinary least squares (OLS) regression. How well does MLR fit? 
    Are the coefficients trustworthy?
    """)
    
    # Fit OLS
    ols_wind = LinearRegression().fit(X, y_wind)
    ols_pv   = LinearRegression().fit(X, y_pv)
    
    r2_wind_ols = r2_score(y_wind, ols_wind.predict(X))
    r2_pv_ols   = r2_score(y_pv,   ols_pv.predict(X))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("WindPower R²", f"{r2_wind_ols:.4f}")
    with col2:
        st.metric("PVPower R²", f"{r2_pv_ols:.4f}")
    
    st.success("Both look excellent! But can we trust these coefficients?")
    
    st.markdown("### Bootstrap Stability Test")
    st.markdown("Draw 200 random subsets and examine coefficient variation.")
    
    with st.spinner("Running bootstrap..."):
        n_boot = 200
        n_sub  = 2000
        
        coefs_wind = np.zeros((n_boot, n_features))
        coefs_pv   = np.zeros((n_boot, n_features))
        
        np.random.seed(42)
        for b in range(n_boot):
            idx = resample(np.arange(len(y_wind)), n_samples=n_sub, random_state=b)
            coefs_wind[b] = LinearRegression().fit(X[idx], y_wind[idx]).coef_
            coefs_pv[b]   = LinearRegression().fit(X[idx], y_pv[idx]).coef_
    
    # Plot
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=['MLR Coefficients — WindPower', 'MLR Coefficients — PVPower'])
    
    for i, feat in enumerate(feature_names):
        fig.add_trace(go.Box(y=coefs_wind[:, i], name=feat, marker_color=BLUE,
                             showlegend=False, boxpoints=False), row=1, col=1)
        fig.add_trace(go.Box(y=coefs_pv[:, i], name=feat, marker_color=ORANGE,
                             showlegend=False, boxpoints=False), row=1, col=2)
    
    fig.add_hline(y=0, line_dash='dot', line_color='black', line_width=0.5)
    fig.update_layout(height=500, width=1200)
    st.plotly_chart(fig, use_container_width=True)
    
    cond = np.linalg.cond(X.T @ X)
    st.warning(f"**Condition number of X'X:** {cond:.1f} — High condition number indicates severe multicollinearity!")
    st.markdown("The boxes are wide → coefficients are **wildly unstable** across bootstrap samples.")

# ===== TAB 2: PCA Structure =====
with tabs[2]:
    st.markdown("## Step 2 — PCA: Understanding the Variance Structure")
    
    # Run PCA
    pca = PCA().fit(X)
    T = pca.transform(X)
    P = pca.components_
    
    expl_var = pca.explained_variance_ratio_
    cum_var  = np.cumsum(expl_var)
    
    # Scree and cumulative plot
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Scree Plot', 'Cumulative Variance'])
    
    fig.add_trace(go.Bar(
        x=list(range(1, n_features+1)), y=expl_var * 100,
        marker_color=BLUE,
        text=[f'{v:.1f}%' for v in expl_var[:5]*100] + [''] * (n_features - 5),
        textposition='outside', showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=list(range(1, n_features+1)), y=cum_var * 100,
        mode='lines+markers', marker=dict(color=BLUE, size=8),
        showlegend=False
    ), row=1, col=2)
    
    # Reference lines
    fig.add_hline(y=60, line_dash='dash', line_color=RED, line_width=1,
                  row=1, col=2, annotation_text='60%')
    fig.add_hline(y=90, line_dash='dash', line_color=ORANGE, line_width=1,
                  row=1, col=2, annotation_text='90%')
    
    # Highlight k=3
    fig.add_trace(go.Scatter(
        x=[3], y=[cum_var[2]*100], mode='markers',
        marker=dict(color=RED, size=14, symbol='circle-open', line=dict(width=3)),
        showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='Principal Component', dtick=1, row=1, col=1)
    fig.update_xaxes(title_text='Number of PCs', dtick=1, row=1, col=2)
    fig.update_yaxes(title_text='Variance Explained (%)', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Variance Explained (%)', row=1, col=2)
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Explained Variance Per PC")
    var_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n_features)],
        'Variance (%)': expl_var * 100,
        'Cumulative (%)': cum_var * 100
    })
    st.dataframe(var_df.style.format({'Variance (%)': '{:.1f}', 'Cumulative (%)': '{:.1f}'}), 
                 use_container_width=True)
    
    st.info("First 3 PCs capture ~61% of X-variance. Let's use k=3 as our starting point for PCR.")

# ===== TAB 3: PCR Performance =====
with tabs[3]:
    st.markdown("## Step 3 — Build PCR with k=3")
    st.markdown("Use the **same 3 PCs** to predict both responses.")
    
    k = st.slider("Number of components (k)", 1, n_features, 3, 1, key='pcr_k')
    
    # Recompute PCA if not done
    if 'pca' not in locals():
        pca = PCA().fit(X)
        T = pca.transform(X)
        expl_var = pca.explained_variance_ratio_
    
    T_k = T[:, :k]
    
    pcr_wind = LinearRegression().fit(T_k, y_wind)
    pcr_pv   = LinearRegression().fit(T_k, y_pv)
    
    y_pred_wind_pcr = pcr_wind.predict(T_k)
    y_pred_pv_pcr   = pcr_pv.predict(T_k)
    
    r2_wind_pcr = r2_score(y_wind, y_pred_wind_pcr)
    r2_pv_pcr   = r2_score(y_pv,   y_pred_pv_pcr)
    
    # MLR vs PCR comparison
    st.markdown("### MLR vs PCR Comparison")
    methods = ['MLR', f'PCR (k={k})']
    r2_wind_list = [r2_wind_ols, r2_wind_pcr]
    r2_pv_list   = [r2_pv_ols,   r2_pv_pcr]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['WindPower', 'PVPower'])
    
    fig.add_trace(go.Bar(x=methods, y=r2_wind_list, marker_color=[GREEN, BLUE],
        text=[f'{v:.4f}' for v in r2_wind_list], textposition='outside',
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=methods, y=r2_pv_list, marker_color=[GREEN, ORANGE],
        text=[f'{v:.4f}' for v in r2_pv_list], textposition='outside',
        showlegend=False), row=1, col=2)
    
    fig.update_yaxes(range=[0, 1.15], title_text='R²')
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Predicted vs Actual
    st.markdown("### Predicted vs Actual")
    n_plot = 5000
    np.random.seed(42)
    idx_plot = np.random.choice(len(y_wind), size=n_plot, replace=False)
    
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f'PCR (k={k}) — WindPower  |  R² = {r2_wind_pcr:.4f}',
                        f'PCR (k={k}) — PVPower  |  R² = {r2_pv_pcr:.4f}'])
    
    fig.add_trace(go.Scattergl(x=y_wind[idx_plot], y=y_pred_wind_pcr[idx_plot],
        mode='markers', marker=dict(color=BLUE, size=3, opacity=0.2),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, y_wind.max()], y=[0, y_wind.max()],
        mode='lines', line=dict(color='black', dash='dash'), showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scattergl(x=y_pv[idx_plot], y=y_pred_pv_pcr[idx_plot],
        mode='markers', marker=dict(color=ORANGE, size=3, opacity=0.2),
        showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, y_pv.max()], y=[0, y_pv.max()],
        mode='lines', line=dict(color='black', dash='dash'), showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='Actual WindPower', row=1, col=1)
    fig.update_yaxes(title_text='Predicted WindPower', row=1, col=1)
    fig.update_xaxes(title_text='Actual PVPower', row=1, col=2)
    fig.update_yaxes(title_text='Predicted PVPower', row=1, col=2)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    if k == 3:
        st.error(f"""
        🔴 **Something is seriously wrong!**
        
        - **PVPower:** R² = {r2_pv_pcr:.4f} ✓ Decent!
        - **WindPower:** R² = {r2_wind_pcr:.4f} ✗ Terrible!
        
        The same PCR model works for one response but fails for the other. Why?
        """)
    elif r2_wind_pcr < 0.5:
        st.warning(f"WindPower R² is still low at {r2_wind_pcr:.4f}. Try increasing k!")
    else:
        st.success(f"Both responses achieve good R² scores!")

# ===== TAB 4: PCA Diagnostics =====
with tabs[4]:
    st.markdown("## Step 4 — PCA Diagnostics: Score Plots")
    
    # Compute PCA fresh
    pca = PCA().fit(X)
    T = pca.transform(X)
    expl_var = pca.explained_variance_ratio_
    
    st.markdown("### Interactive Score Plot")
    st.markdown("**Instructions:** Use dropdowns to select which PCs to display. Try PC5 vs PC6 to see the wind signal!")
    
    # Subsample for performance
    n_plot = 5000
    np.random.seed(42)
    idx_plot = np.random.choice(len(y_wind), size=n_plot, replace=False)
    T_sub = T[idx_plot]
    
    col1, col2 = st.columns(2)
    with col1:
        pc_x = st.selectbox("X-axis PC", [f"PC{i+1}" for i in range(n_features)], index=0, key='score_x')
    with col2:
        pc_y = st.selectbox("Y-axis PC", [f"PC{i+1}" for i in range(n_features)], index=1, key='score_y')
    
    pc_x_idx = int(pc_x[2:]) - 1
    pc_y_idx = int(pc_y[2:]) - 1
    
    # Debug
    st.caption(f"Debug: Score plot using {pc_x} (index {pc_x_idx}) vs {pc_y} (index {pc_y_idx})")
    
    # Create three figures side by side
    # Figure 1: Score plot colored by WindPower
    fig_wind = go.Figure()
    fig_wind.add_trace(go.Scattergl(
        x=T_sub[:, pc_x_idx], y=T_sub[:, pc_y_idx], mode='markers',
        marker=dict(size=4, color=y_wind[idx_plot], colorscale='Viridis', opacity=0.4,
                    colorbar=dict(title='WindPower')),
        showlegend=False
    ))
    fig_wind.update_xaxes(title_text=f'{pc_x} ({expl_var[pc_x_idx]*100:.1f}%)')
    fig_wind.update_yaxes(title_text=f'{pc_y} ({expl_var[pc_y_idx]*100:.1f}%)')
    fig_wind.update_layout(
        title='Scores by WindPower',
        height=600,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    # Compute correlations for SELECTED PCs (not just PC1/PC2)
    corr_loadings = np.zeros((n_features, 2))
    for j in range(n_features):
        corr_loadings[j, 0] = np.corrcoef(X[:, j], T[:, pc_x_idx])[0, 1]
        corr_loadings[j, 1] = np.corrcoef(X[:, j], T[:, pc_y_idx])[0, 1]
    
    corr_wind_x = np.corrcoef(y_wind, T[:, pc_x_idx])[0, 1]
    corr_wind_y = np.corrcoef(y_wind, T[:, pc_y_idx])[0, 1]
    corr_pv_x   = np.corrcoef(y_pv,   T[:, pc_x_idx])[0, 1]
    corr_pv_y   = np.corrcoef(y_pv,   T[:, pc_y_idx])[0, 1]
    
    # Debug
    st.caption(f"Debug: WindSpeed corr = ({corr_loadings[feature_names.index('WindSpeed'), 0]:.3f}, {corr_loadings[feature_names.index('WindSpeed'), 1]:.3f})")
    
    radiation_vars = {'IrrDirect', 'IrrDiffuse', 'RadSurface', 'RadTop', 'IsDayBin'}
    wind_vars = {'WindSpeed'}
    
    # Figure 2: Correlation loading plot
    fig_corr = go.Figure()
    
    # Unit circles
    theta = np.linspace(0, 2*np.pi, 100)
    fig_corr.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color=GRAY, dash='dash', width=1),
        showlegend=False, hoverinfo='skip'))
    fig_corr.add_trace(go.Scatter(x=0.5*np.cos(theta), y=0.5*np.sin(theta),
        mode='lines', line=dict(color=GRAY, dash='dot', width=0.5),
        showlegend=False, hoverinfo='skip'))
    
    # Variables
    for j, name in enumerate(feature_names):
        cx, cy = float(corr_loadings[j, 0]), float(corr_loadings[j, 1])
        if name in radiation_vars:
            color = ORANGE
        elif name in wind_vars:
            color = BLUE
        else:
            color = GRAY
        
        fig_corr.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers+text', text=[name],
            textposition='top center', textfont=dict(color=color, size=9),
            marker=dict(color=color, size=7), showlegend=False
        ))
        fig_corr.add_annotation(x=cx, y=cy, ax=0, ay=0, xref='x', yref='y',
                           axref='x', ayref='y', showarrow=True,
                           arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=color)
    
    # Responses
    fig_corr.add_trace(go.Scatter(
        x=[float(corr_pv_x)], y=[float(corr_pv_y)], mode='markers+text',
        text=['PVPower'], textposition='top right', textfont=dict(color=ORANGE, size=11),
        marker=dict(color=ORANGE, size=15, symbol='star', line=dict(color='black', width=1)),
        showlegend=False
    ))
    fig_corr.add_trace(go.Scatter(
        x=[float(corr_wind_x)], y=[float(corr_wind_y)], mode='markers+text',
        text=['WindPower'], textposition='top right', textfont=dict(color=BLUE, size=11),
        marker=dict(color=BLUE, size=15, symbol='star', line=dict(color='black', width=1)),
        showlegend=False
    ))
    
    fig_corr.update_layout(
        title=f'Correlation Loading',
        xaxis=dict(title=f'Corr with {pc_x}',
                   range=[-1.15, 1.15], constrain='domain', zeroline=True),
        yaxis=dict(title=f'Corr with {pc_y}',
                   range=[-1.15, 1.15], scaleanchor='x', scaleratio=1, zeroline=True),
        height=600,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    # Figure 3: Score plot colored by PVPower
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scattergl(
        x=T_sub[:, pc_x_idx], y=T_sub[:, pc_y_idx], mode='markers',
        marker=dict(size=4, color=y_pv[idx_plot], colorscale='Magma', opacity=0.4,
                    colorbar=dict(title='PVPower')),
        showlegend=False
    ))
    fig_pv.update_xaxes(title_text=f'{pc_x} ({expl_var[pc_x_idx]*100:.1f}%)')
    fig_pv.update_yaxes(title_text=f'{pc_y} ({expl_var[pc_y_idx]*100:.1f}%)')
    fig_pv.update_layout(
        title='Scores by PVPower',
        height=600,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    # Display all three side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_wind, use_container_width=True)
    with col2:
        st.plotly_chart(fig_corr, use_container_width=True)
    with col3:
        st.plotly_chart(fig_pv, use_container_width=True)
    
    st.success("""
    **Key Insight:** 
    - PVPower (orange ★) sits right next to radiation cluster → PC1 captures PV information
    - WindPower (blue ★) is near the center → PC1 and PC2 do NOT capture wind information
    """)

# ===== TAB 5: Feature Analysis =====
with tabs[5]:
    st.markdown("## Step 5 — Feature-Response Relationships")
    
    wind_features = ['WindSpeed', 'AirDensity', 'Percipitation', 'SnowFlow']
    pv_features   = ['IrrDirect', 'IrrDiffuse', 'RadSurface', 'Temperature']
    
    wind_corrs = [np.corrcoef(X_df[f].values, y_wind)[0, 1] for f in wind_features]
    pv_corrs   = [np.corrcoef(X_df[f].values, y_pv)[0, 1] for f in pv_features]
    
    n_plot = 3000
    np.random.seed(42)
    idx_plot = np.random.choice(len(y_wind), size=n_plot, replace=False)
    
    fig = make_subplots(rows=2, cols=4,
        subplot_titles=[f'{f}  (r = {r:.3f})' for f, r in zip(wind_features, wind_corrs)] +
                       [f'{f}  (r = {r:.3f})' for f, r in zip(pv_features, pv_corrs)],
        row_titles=['WindPower', 'PVPower'])
    
    for i, feat in enumerate(wind_features):
        fig.add_trace(go.Scattergl(
            x=X_df[feat].values[idx_plot], y=y_wind[idx_plot],
            mode='markers', marker=dict(color=BLUE, size=3, opacity=0.15),
            showlegend=False
        ), row=1, col=i+1)
        fig.update_xaxes(title_text=feat, row=1, col=i+1)
    
    for i, feat in enumerate(pv_features):
        fig.add_trace(go.Scattergl(
            x=X_df[feat].values[idx_plot], y=y_pv[idx_plot],
            mode='markers', marker=dict(color=ORANGE, size=3, opacity=0.15),
            showlegend=False
        ), row=2, col=i+1)
        fig.update_xaxes(title_text=feat, row=2, col=i+1)
    
    fig.update_yaxes(title_text='WindPower', row=1, col=1)
    fig.update_yaxes(title_text='PVPower', row=2, col=1)
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **Conclusion:**
    - WindPower is almost entirely determined by **WindSpeed** (r = 0.98)
    - PVPower is driven by **radiation variables** (IrrDirect r = 0.98)
    - These are two completely different groups of predictors!
    """)

# ===== TAB 6: PC Interpretation =====
with tabs[6]:
    st.markdown("## Step 6 — What Do PC1 and PC2 Actually Capture?")
    
    if 'P' not in locals():
        pca = PCA().fit(X)
        P = pca.components_
        expl_var = pca.explained_variance_ratio_
    
    radiation_vars = {'IrrDirect', 'IrrDiffuse', 'RadSurface', 'RadTop', 'IsDayBin'}
    wind_vars = {'WindSpeed'}
    
    # Loading bar charts
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f'PC1 Loadings ({expl_var[0]*100:.1f}% variance)',
                        f'PC2 Loadings ({expl_var[1]*100:.1f}% variance)'])
    
    for col, pc_idx in enumerate([0, 1]):
        loadings_pc = P[pc_idx]
        sort_idx = np.argsort(np.abs(loadings_pc))[::-1]
        sorted_names = [feature_names[i] for i in sort_idx]
        sorted_vals  = loadings_pc[sort_idx]
        colors = [ORANGE if n in radiation_vars else BLUE if n in wind_vars else GRAY
                  for n in sorted_names]
        
        fig.add_trace(go.Bar(
            y=sorted_names, x=sorted_vals, orientation='h',
            marker_color=colors, showlegend=False
        ), row=1, col=col+1)
        fig.update_yaxes(autorange='reversed', row=1, col=col+1)
        fig.update_xaxes(title_text='Loading value', row=1, col=col+1)
    
    fig.add_vline(x=0, line_color='black', line_width=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **PC1:** Dominated by RADIATION block (orange) → Excellent for PVPower, useless for WindPower
    
    **PC2:** Captures seasonal/snow patterns → Neither response strongly aligned
    """)
    
    st.markdown("### Full Loading Matrix")
    loading_df = pd.DataFrame(
        P,
        index=[f'PC{i+1}  ({expl_var[i]*100:.1f}%)' for i in range(n_features)],
        columns=feature_names
    )
    
    fig = px.imshow(loading_df, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-0.6, zmax=0.6, aspect='auto',
                    title='PCA Loading Matrix')
    fig.update_layout(height=650, coloraxis_colorbar_title='Loading',
                      xaxis_title='Original Feature', yaxis_title='Principal Component')
    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 7: PC-Response Correlation =====
with tabs[7]:
    st.markdown("## Step 7 — Where Does the Wind Signal Live?")
    st.markdown("Systematically check correlation of every PC with both responses.")
    
    if 'T' not in locals():
        pca = PCA().fit(X)
        T = pca.transform(X)
        expl_var = pca.explained_variance_ratio_
    
    pc_corr_wind = [np.corrcoef(T[:, j], y_wind)[0, 1] for j in range(n_features)]
    pc_corr_pv   = [np.corrcoef(T[:, j], y_pv  )[0, 1] for j in range(n_features)]
    
    pc_labels = [f'PC{i+1}' for i in range(n_features)]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(
        x=pc_labels, y=pc_corr_wind, name='WindPower',
        marker_color=BLUE, offsetgroup=0, width=0.35
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=pc_labels, y=pc_corr_pv, name='PVPower',
        marker_color=ORANGE, offsetgroup=1, width=0.35
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=pc_labels, y=expl_var * 100, name='X-variance (%)',
        mode='lines+markers', marker=dict(color=GRAY, size=8, symbol='square'),
        line=dict(color=GRAY, dash='dash'), opacity=0.6
    ), secondary_y=True)
    
    fig.add_hline(y=0, line_color='black', line_width=0.5, secondary_y=False)
    fig.update_layout(
        title='Correlation of Each PC with Responses vs X-Variance',
        barmode='group', height=500
    )
    fig.update_yaxes(title_text='Correlation with response', secondary_y=False)
    fig.update_yaxes(title_text='X-variance explained (%)', secondary_y=True)
    fig.update_xaxes(title_text='Principal Component')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Detailed Correlation Table")
    corr_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n_features)],
        'X-variance (%)': expl_var * 100,
        'r(Wind)': pc_corr_wind,
        'r(PV)': pc_corr_pv
    })
    
    def highlight_strong(val, threshold=0.3):
        if abs(val) > threshold:
            return 'background-color: yellow'
        return ''
    
    styled_df = corr_df.style.format({
        'X-variance (%)': '{:.1f}',
        'r(Wind)': '{:+.3f}',
        'r(PV)': '{:+.3f}'
    }).applymap(highlight_strong, subset=['r(Wind)', 'r(PV)'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.error("""
    🔑 **The Key Finding:**
    
    | PC | X-variance | Corr with WindPower | Corr with PVPower |
    |----|-----------|---------------------|-------------------|
    | PC1 | **34.7%** (highest!) | −0.14 (weak) | **+0.87** (strong ✓) |
    | PC5 | **7.2%** (low!) | **+0.58** (strong ✗) | +0.19 |
    | PC6 | **6.2%** (low!) | **+0.56** (strong ✗) | −0.15 |
    
    **PCA ranks by X-variance, NOT by predictive relevance for y!**
    """)

# ===== TAB 8: R² vs k =====
with tabs[8]:
    st.markdown("## Step 8 — PCR Accuracy vs Number of Components")
    st.markdown("Verify: PCR should improve for WindPower once we include PC5 and PC6.")
    
    if 'T' not in locals():
        pca = PCA().fit(X)
        T = pca.transform(X)
        expl_var = pca.explained_variance_ratio_
    
    ks = range(1, n_features + 1)
    r2_vs_k_wind = []
    r2_vs_k_pv   = []
    
    with st.spinner("Computing R² for all k values..."):
        for k_val in ks:
            T_k = T[:, :k_val]
            reg_w = LinearRegression().fit(T_k, y_wind)
            reg_p = LinearRegression().fit(T_k, y_pv)
            r2_vs_k_wind.append(r2_score(y_wind, reg_w.predict(T_k)))
            r2_vs_k_pv.append(r2_score(y_pv, reg_p.predict(T_k)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(ks), y=r2_vs_k_wind, mode='lines+markers', name='WindPower',
        marker=dict(color=BLUE, size=10), line=dict(color=BLUE)
    ))
    fig.add_trace(go.Scatter(
        x=list(ks), y=r2_vs_k_pv, mode='lines+markers', name='PVPower',
        marker=dict(color=ORANGE, size=10, symbol='square'), line=dict(color=ORANGE)
    ))
    
    # MLR reference
    fig.add_hline(y=r2_wind_ols, line_dash='dot', line_color=BLUE, opacity=0.5,
        annotation_text=f'MLR Wind R²={r2_wind_ols:.3f}', annotation_position='bottom right')
    fig.add_hline(y=r2_pv_ols, line_dash='dot', line_color=ORANGE, opacity=0.5,
        annotation_text=f'MLR PV R²={r2_pv_ols:.3f}', annotation_position='top right')
    
    # k=3 line
    fig.add_vline(x=3, line_dash='dash', line_color=RED, line_width=1.5, opacity=0.7)
    fig.add_annotation(x=3.3, y=0.15, text='k=3<br>(initial choice)',
                       showarrow=False, font=dict(color=RED, size=11))
    
    # PC5-6 annotation
    fig.add_annotation(x=6, y=r2_vs_k_wind[5], text='PC5–PC6 added →<br>Wind captured!',
                       ax=80, ay=-60, arrowhead=2, arrowcolor=BLUE,
                       font=dict(color=BLUE, size=11))
    
    fig.update_layout(
        title='PCR Accuracy vs Number of Components',
        xaxis=dict(title='Number of PCs (k)', dtick=1),
        yaxis=dict(title='R²', range=[-0.02, 1.05]),
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"""
    **The Story:**
    - PVPower reaches R² > 0.75 at k=1 — PC1 alone is almost enough!
    - WindPower stays below R² = 0.20 until k=5
    - At k=6, WindPower jumps to R² = {r2_vs_k_wind[5]:.3f} — PC5 & PC6 carry the wind information
    """)
    
    st.markdown("---")
    st.markdown("## 🎯 Summary: The Fundamental Lesson")
    
    st.markdown(r"""
    ### PCA maximizes X-variance, NOT covariance with y
    
    $$\text{PCA maximizes: } \text{Var}(\mathbf{t}_a) = \text{Var}(\mathbf{Xp}_a)$$
    
    $$\text{Prediction requires: } \text{Cov}(\mathbf{t}_a, \mathbf{y})$$
    
    **These are not the same thing!**
    
    - When predictive signal lies in **high-variance direction** (PVPower) → PCR works
    - When predictive signal lies in **low-variance direction** (WindPower) → PCR fails
    
    **This is the motivation for Partial Least Squares (PLS),** which directly maximizes covariance with y.
    """)

st.markdown("---")
st.markdown("*Based on PCR_Application.ipynb — Norwegian Renewable Energy Dataset*")
