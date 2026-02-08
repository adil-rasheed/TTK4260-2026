"""
Principal Component Analysis Explorer
======================================
Interactive PCA demonstrations and visualizations with real datasets.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_generator import generate_rotated_data, load_citytemp_data
from utils.visualizations import (plot_pca_2d, plot_correlation_heatmap, 
                                   plot_loading_heatmap, plot_correlation_loading_plot,
                                   plot_communalities, nipals_algorithm, 
                                   plot_nipals_convergence, plot_nipals_step)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="PCA Explorer", page_icon="📐", layout="wide")

st.title("📐 Principal Component Analysis")
st.markdown("### Dimensionality Reduction and Data Visualization")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 2D PCA Demo",
    "📊 Variance Explained",
    "🍔 McDonald's Dataset",
    "🌡️ CityTemp Dataset",
    "📉 Loadings & Biplots",
    "🔍 Outlier Detection",
    "🔬 Advanced Visualizations"
])

# ===== TAB 1: 2D PCA Demo =====
with tab1:
    st.markdown("## Interactive 2D PCA Demonstration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Data Generation")
        
        angle = st.slider("Rotation angle (degrees)", 0, 180, 45, 5)
        var_parallel = st.slider("Variance along main axis", 0.5, 10.0, 4.0, 0.5)
        var_perp = st.slider("Variance perpendicular", 0.1, 5.0, 0.5, 0.1)
        n_samples = st.slider("Number of samples", 50, 500, 200, 50)
        
        # Generate data
        data = generate_rotated_data(angle, var_parallel, var_perp, n_samples, random_state=42)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(data)
        
        st.markdown("### 📈 Results")
        st.metric("Variance PC1", f"{pca.explained_variance_[0]:.2f}")
        st.metric("Variance PC2", f"{pca.explained_variance_[1]:.2f}")
        st.metric("Total Var Explained", f"{pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    with col2:
        # Visualize
        fig = plot_pca_2d(data, pca.components_.T)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 PCA Components"):
            st.write("**PC1 (Red):**", pca.components_[0])
            st.write("**PC2 (Green):**", pca.components_[1])
            st.write("**Explained Variance Ratio:**", pca.explained_variance_ratio_)

# ===== TAB 2: Variance Explained =====
with tab2:
    st.markdown("## Scree Plot and Cumulative Variance")
    
    n_features = st.slider("Number of features", 3, 20, 10, 1, key='var_features')
    correlation = st.slider("Feature correlation", 0.0, 0.9, 0.5, 0.1, key='var_corr')
    
    # Generate high-dimensional data
    from utils.data_generator import generate_correlated_data
    X, _, _ = generate_correlated_data(n=200, p=n_features, correlation=correlation, random_state=42)
    
    # Fit PCA
    pca = PCA()
    pca.fit(X)
    
    # Create scree plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Scree Plot", "Cumulative Variance Explained")
    )
    
    # Scree plot
    fig.add_trace(go.Bar(
        x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
        y=pca.explained_variance_ratio_,
        name='Variance Ratio',
        marker_color='steelblue'
    ), row=1, col=1)
    
    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumsum) + 1)),
        y=cumsum,
        mode='lines+markers',
        name='Cumulative',
        line=dict(color='red', width=3)
    ), row=1, col=2)
    
    fig.add_hline(y=0.95, line_dash="dash", line_color="green", row=1, col=2)
    
    fig.update_xaxes(title_text="Component", row=1, col=1)
    fig.update_yaxes(title_text="Variance Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Component", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Variance", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Find components for 95% variance
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    st.info(f"💡 **{n_components_95}** components needed to explain 95% of variance")

# ===== TAB 3: McDonald's Dataset =====
with tab3:
    st.markdown("## 🍔 McDonald's Menu Nutritional Analysis")
    
    st.info("📊 Analyzing nutritional content of McDonald's menu items using PCA")
    
    # Load McDonald's data
    try:
        df_mcdonalds = pd.read_csv('data/macdonald.csv')
        
        st.markdown("### 📋 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Items", df_mcdonalds.shape[0])
        col2.metric("Features", df_mcdonalds.shape[1])
        col3.metric("Categories", df_mcdonalds['Menu Category'].nunique())
        
        # Select numerical features for PCA
        numerical_cols = ['Energy (kCal)', 'Protein (g)', 'Total fat (g)', 
                         'Sat Fat (g)', 'Trans fat (g)', 'Cholesterols (mg)',
                         'Total carbohydrate (g)', 'Total Sugars (g)', 
                         'Added Sugars (g)', 'Sodium (mg)']
        
        X = df_mcdonalds[numerical_cols].dropna()
        categories = df_mcdonalds.loc[X.index, 'Menu Category']
        item_names = df_mcdonalds.loc[X.index, 'Menu Items']
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA()
        scores = pca.fit_transform(X_scaled)
        
        # Show options
        show_labels = st.checkbox("Show item labels", value=False)
        color_by = st.selectbox("Color by:", ["Menu Category", "Energy (kCal)", "Protein (g)", "Total fat (g)"])
        
        # Create tabs for different analyses
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "📊 PC1 vs PC2", 
            "📈 Variance Explained",
            "🔍 Feature Correlations",
            "📉 Loadings Plot"
        ])
        
        with subtab1:
            # PC1 vs PC2 scatter plot
            fig = go.Figure()
            
            if color_by == "Menu Category":
                for cat in categories.unique():
                    mask = categories == cat
                    fig.add_trace(go.Scatter(
                        x=scores[mask, 0],
                        y=scores[mask, 1],
                        mode='markers+text' if show_labels else 'markers',
                        text=item_names[mask].values if show_labels else None,
                        textposition='top center',
                        name=cat,
                        marker=dict(size=8)
                    ))
            else:
                color_values = df_mcdonalds.loc[X.index, color_by]
                fig.add_trace(go.Scatter(
                    x=scores[:, 0],
                    y=scores[:, 1],
                    mode='markers+text' if show_labels else 'markers',
                    text=item_names.values if show_labels else None,
                    textposition='top center',
                    marker=dict(
                        size=8,
                        color=color_values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=color_by)
                    ),
                    showlegend=False
                ))
            
            fig.update_layout(
                title=f"McDonald's Menu Items in PC Space",
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("""
            **Interpretation:**
            - Items closer together have similar nutritional profiles
            - PC1 captures the main variation in nutritional content
            - PC2 captures secondary patterns
            """)
        
        with subtab2:
            # Scree plot and cumulative variance
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Scree Plot", "Cumulative Variance")
            )
            
            # Scree plot
            fig.add_trace(go.Bar(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=pca.explained_variance_ratio_,
                name='Variance',
                marker_color='steelblue'
            ), row=1, col=1)
            
            # Cumulative variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumsum) + 1)),
                y=cumsum,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='red', width=3)
            ), row=1, col=2)
            
            fig.add_hline(y=0.95, line_dash="dash", line_color="green", row=1, col=2)
            
            fig.update_xaxes(title_text="Component", row=1, col=1)
            fig.update_yaxes(title_text="Variance Ratio", row=1, col=1)
            fig.update_xaxes(title_text="Component", row=1, col=2)
            fig.update_yaxes(title_text="Cumulative Variance", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            n_components_95 = np.argmax(cumsum >= 0.95) + 1
            st.success(f"✅ **{n_components_95}** components explain 95% of variance")
            
            # Show variance table
            var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(min(5, len(pca.explained_variance_ratio_)))],
                'Variance Explained': pca.explained_variance_ratio_[:5],
                'Cumulative Variance': cumsum[:5]
            })
            st.dataframe(var_df, use_container_width=True)
        
        with subtab3:
            # Correlation heatmap
            st.markdown("### Feature Correlation Matrix")
            
            corr_matrix = X.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 8}
            ))
            
            fig.update_layout(
                title="Correlation Between Nutritional Features",
                height=600,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # High correlations
            st.markdown("### 🔗 Highly Correlated Features")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
            else:
                st.info("No feature pairs with correlation > 0.7")
        
        with subtab4:
            # Loadings plot (contribution of features to PCs)
            st.markdown("### Feature Loadings on Principal Components")
            
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            fig = go.Figure()
            
            for i, feature in enumerate(numerical_cols):
                fig.add_trace(go.Scatter(
                    x=[0, loadings[i, 0]],
                    y=[0, loadings[i, 1]],
                    mode='lines+text',
                    name=feature,
                    text=['', feature],
                    textposition='top center',
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Feature Contributions to PC1 and PC2",
                xaxis_title=f"PC1 Loadings ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                yaxis_title=f"PC2 Loadings ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                height=600,
                showlegend=False
            )
            
            # Add circle
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Longer arrows indicate stronger contribution to PCs
            - Direction shows which PC the feature contributes to
            - Features pointing in similar directions are correlated
            """)
            
            # Loadings table
            st.markdown("### Loadings Values")
            loadings_df = pd.DataFrame(
                pca.components_[:3, :].T,
                columns=['PC1', 'PC2', 'PC3'],
                index=numerical_cols
            )
            st.dataframe(loadings_df.style.background_gradient(cmap='RdBu', axis=None), 
                        use_container_width=True)
        
        # Raw data preview
        with st.expander("📄 View Raw Data"):
            st.dataframe(df_mcdonalds[['Menu Items', 'Menu Category'] + numerical_cols].head(20),
                        use_container_width=True)
    
    except FileNotFoundError:
        st.error("❌ McDonald's dataset not found. Please ensure 'data/macdonald.csv' exists.")

# ===== TAB 4: CityTemp Dataset =====
with tab4:
    st.markdown("## 🌡️ City Temperature Analysis")
    
    st.info("📂 Analyzing temperature patterns across different cities using PCA")
    
    # Create sample data if real data not available
    np.random.seed(42)
    cities = ['New York', 'London', 'Tokyo', 'Sydney', 'Mumbai', 'Cairo', 
              'Moscow', 'Rio', 'Berlin', 'Singapore']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Generate realistic temperature data with seasonal patterns
    base_temps = [0, 5, 15, 22, 25, 30, -5, 25, 10, 27]  # Average temps
    data_dict = {}
    for i, city in enumerate(cities):
        # Southern hemisphere cities have inverted seasons
        if city in ['Sydney', 'Rio']:
            seasonal = base_temps[i] + 10 * np.sin(np.linspace(np.pi, 3*np.pi, 12))
        else:
            seasonal = base_temps[i] + 10 * np.sin(np.linspace(0, 2*np.pi, 12))
        noise = np.random.randn(12) * 2
        data_dict[city] = seasonal + noise
    
    df = pd.DataFrame(data_dict, index=months)
    
    subtab1, subtab2, subtab3 = st.tabs([
        "📊 Temperature Data",
        "🗺️ PCA Projection",
        "📈 Time Series"
    ])
    
    with subtab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Temperature Data (°C)")
            st.dataframe(df.round(1), use_container_width=True)
            
            # Statistics
            st.markdown("### 📈 City Statistics")
            stats_df = pd.DataFrame({
                'City': cities,
                'Mean Temp': df.mean().values,
                'Std Dev': df.std().values,
                'Min': df.min().values,
                'Max': df.max().values
            })
            st.dataframe(stats_df.round(2), use_container_width=True)
        
        with col2:
            # Correlation heatmap
            corr = df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=cities,
                y=cities,
                colorscale='RdBu',
                zmid=0,
                text=corr.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 8}
            ))
            fig.update_layout(
                title="City Temperature Correlations",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with subtab2:
        # Perform PCA
        pca = PCA(n_components=2)
        scores = pca.fit_transform(df.T)
        
        # Plot
        fig = go.Figure()
        
        # Add cities
        fig.add_trace(go.Scatter(
            x=scores[:, 0],
            y=scores[:, 1],
            mode='markers+text',
            text=cities,
            textposition='top center',
            marker=dict(size=12, color=df.mean().values, 
                       colorscale='RdYlBu_r', showscale=True,
                       colorbar=dict(title="Mean Temp"))
        ))
        
        fig.update_layout(
            title="Cities in Principal Component Space",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔍 PC1 Interpretation")
            loadings_pc1 = pca.components_[0]
            st.write("**Top contributing months:**")
            top_months = pd.Series(loadings_pc1, index=months).abs().nlargest(3)
            for month, loading in top_months.items():
                st.write(f"- {month}: {loading:.3f}")
        
        with col2:
            st.markdown("### 🔍 PC2 Interpretation")
            loadings_pc2 = pca.components_[1]
            st.write("**Top contributing months:**")
            top_months = pd.Series(loadings_pc2, index=months).abs().nlargest(3)
            for month, loading in top_months.items():
                st.write(f"- {month}: {loading:.3f}")
    
    with subtab3:
        # Time series visualization
        st.markdown("### 🌡️ Temperature Time Series")
        
        selected_cities = st.multiselect(
            "Select cities to display:",
            cities,
            default=cities[:4]
        )
        
        if selected_cities:
            fig = go.Figure()
            
            for city in selected_cities:
                fig.add_trace(go.Scatter(
                    x=months,
                    y=df[city],
                    mode='lines+markers',
                    name=city,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Temperature Patterns Across Months",
                xaxis_title="Month",
                yaxis_title="Temperature (°C)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ===== TAB 5: Loadings & Biplots =====
with tab5:
    st.markdown("## 📉 Advanced PCA Visualizations: Loadings and Biplots")
    
    st.markdown("""
    **Biplots** combine scores (observations) and loadings (variables) in a single plot,
    showing both the data structure and the contribution of features.
    """)
    
    # Generate sample data
    n_samples = st.slider("Number of samples", 30, 200, 100, 10, key='biplot_n')
    n_features = st.slider("Number of features", 3, 10, 5, 1, key='biplot_p')
    
    if st.button("Generate Biplot", type="primary"):
        from utils.data_generator import generate_correlated_data
        X, y, beta = generate_correlated_data(n=n_samples, p=n_features, 
                                              correlation=0.6, random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=min(n_features, 10))
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        st.session_state.biplot_scores = scores
        st.session_state.biplot_loadings = loadings
        st.session_state.biplot_pca = pca
        st.session_state.biplot_n_features = n_features
    
    if 'biplot_scores' in st.session_state:
        scores = st.session_state.biplot_scores
        loadings = st.session_state.biplot_loadings
        pca = st.session_state.biplot_pca
        n_features = st.session_state.biplot_n_features
        
        subtab1, subtab2, subtab3 = st.tabs([
            "🎯 Biplot",
            "📊 Loadings Heatmap",
            "📈 Component Analysis"
        ])
        
        with subtab1:
            # Create biplot
            fig = go.Figure()
            
            # Add observations (scores)
            fig.add_trace(go.Scatter(
                x=scores[:, 0],
                y=scores[:, 1],
                mode='markers',
                name='Observations',
                marker=dict(size=6, color='lightblue', opacity=0.6)
            ))
            
            # Add variable arrows (loadings)
            scale_factor = np.max(np.abs(scores[:, :2])) / np.max(np.abs(loadings[:, :2])) * 0.8
            for i in range(n_features):
                fig.add_trace(go.Scatter(
                    x=[0, loadings[i, 0] * scale_factor],
                    y=[0, loadings[i, 1] * scale_factor],
                    mode='lines+text',
                    name=f'Feature {i+1}',
                    text=['', f'F{i+1}'],
                    textposition='top center',
                    line=dict(color='red', width=2),
                    marker=dict(size=8, color='red')
                ))
            
            fig.update_layout(
                title=f"PCA Biplot - Observations and Variables",
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Biplot Interpretation:**
            - **Blue dots**: Individual observations
            - **Red arrows**: Original features
            - **Arrow length**: Importance of feature
            - **Arrow direction**: Correlation with PCs
            - **Angle between arrows**: Correlation between features
            """)
        
        with subtab2:
            # Loadings heatmap
            st.markdown("### Feature Loadings on All Components")
            
            n_components_display = min(5, pca.n_components_)
            loadings_matrix = pca.components_[:n_components_display, :]
            
            fig = go.Figure(data=go.Heatmap(
                z=loadings_matrix,
                x=[f'Feature {i+1}' for i in range(n_features)],
                y=[f'PC{i+1}' for i in range(n_components_display)],
                colorscale='RdBu',
                zmid=0,
                text=loadings_matrix,
                texttemplate='%{text:.3f}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Loadings Matrix - Feature Contributions to PCs",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Reading the Heatmap:**
            - **Red**: Positive contribution
            - **Blue**: Negative contribution
            - **White**: Minimal contribution
            - Each row shows how features contribute to that PC
            """)
        
        with subtab3:
            # Component analysis
            st.markdown("### Individual Component Analysis")
            
            selected_pc = st.selectbox(
                "Select Principal Component:",
                [f"PC{i+1}" for i in range(min(5, pca.n_components_))],
                key='select_pc'
            )
            
            pc_idx = int(selected_pc[2:]) - 1
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Bar plot of loadings
                loadings_pc = pca.components_[pc_idx]
                
                fig = go.Figure(data=go.Bar(
                    x=[f'F{i+1}' for i in range(n_features)],
                    y=loadings_pc,
                    marker_color=['red' if x > 0 else 'blue' for x in loadings_pc]
                ))
                
                fig.update_layout(
                    title=f"{selected_pc} Feature Contributions",
                    xaxis_title="Feature",
                    yaxis_title="Loading",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Score distribution
                fig = go.Figure(data=go.Histogram(
                    x=scores[:, pc_idx],
                    nbinsx=30,
                    marker_color='steelblue'
                ))
                
                fig.update_layout(
                    title=f"{selected_pc} Score Distribution",
                    xaxis_title="Score Value",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown(f"### {selected_pc} Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Variance Explained", f"{pca.explained_variance_ratio_[pc_idx]*100:.2f}%")
            col2.metric("Eigenvalue", f"{pca.explained_variance_[pc_idx]:.3f}")
            col3.metric("Score Mean", f"{np.mean(scores[:, pc_idx]):.3f}")
            col4.metric("Score Std", f"{np.std(scores[:, pc_idx]):.3f}")

# ===== TAB 6: Outlier Detection =====
with tab6:
    st.markdown("## 🔍 Outlier Detection Using PCA")
    
    st.markdown("""
    PCA can help identify outliers as points that are:
    1. Far from the main cluster in PC space
    2. Have high reconstruction error
    3. Have extreme scores on principal components
    """)
    
    # Generate data with outliers
    n_normal = st.slider("Number of normal samples", 50, 300, 180, 10, key='outlier_n')
    n_outliers = st.slider("Number of outliers", 5, 50, 20, 5, key='outlier_count')
    
    if st.button("Generate Data with Outliers", type="primary"):
        np.random.seed(42)
        normal_data = generate_rotated_data(45, 4.0, 0.5, n_normal, random_state=42)
        outliers = np.random.uniform(-15, 15, (n_outliers, 2))
        data_with_outliers = np.vstack([normal_data, outliers])
        labels = np.array(['Normal']*n_normal + ['Outlier']*n_outliers)
        
        st.session_state.outlier_data = data_with_outliers
        st.session_state.outlier_labels = labels
    
    if 'outlier_data' in st.session_state:
        data_with_outliers = st.session_state.outlier_data
        labels = st.session_state.outlier_labels
        
        # PCA
        pca = PCA(n_components=2)
        scores = pca.fit_transform(data_with_outliers)
        
        # Reconstruction error
        reconstructed = pca.inverse_transform(scores)
        reconstruction_error = np.sqrt(np.sum((data_with_outliers - reconstructed)**2, axis=1))
        
        subtab1, subtab2, subtab3 = st.tabs([
            "📊 Score-based Detection",
            "🔧 Reconstruction Error",
            "📈 Combined Methods"
        ])
        
        with subtab1:
            st.markdown("### Distance-based Outlier Detection in PC Space")
            
            # Calculate distances
            distances = np.sqrt(np.sum(scores**2, axis=1))
            threshold = st.slider("Outlier threshold (Mahalanobis distance)", 
                                 1.0, 15.0, 6.0, 0.5, key='distance_threshold')
            
            detected_outliers = distances > threshold
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            true_outliers = labels == 'Outlier'
            tp = np.sum(detected_outliers & true_outliers)
            fp = np.sum(detected_outliers & ~true_outliers)
            fn = np.sum(~detected_outliers & true_outliers)
            tn = np.sum(~detected_outliers & ~true_outliers)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            col1.metric("Detected", detected_outliers.sum())
            col2.metric("True Outliers", true_outliers.sum())
            col3.metric("Precision", f"{precision:.2f}")
            col4.metric("Recall", f"{recall:.2f}")
            
            # Plot
            fig = go.Figure()
            
            # Normal points
            fig.add_trace(go.Scatter(
                x=scores[~detected_outliers, 0],
                y=scores[~detected_outliers, 1],
                mode='markers',
                name='Normal',
                marker=dict(size=6, color='blue', opacity=0.6)
            ))
            
            # Detected outliers
            fig.add_trace(go.Scatter(
                x=scores[detected_outliers, 0],
                y=scores[detected_outliers, 1],
                mode='markers',
                name='Detected Outliers',
                marker=dict(size=10, color='red', symbol='x')
            ))
            
            # Threshold circle
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=threshold * np.cos(theta),
                y=threshold * np.sin(theta),
                mode='lines',
                name='Threshold',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Outlier Detection in PC Space",
                xaxis_title="PC1",
                yaxis_title="PC2",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with subtab2:
            st.markdown("### Reconstruction Error Method")
            
            st.markdown("""
            Points with high reconstruction error don't fit the main pattern
            captured by principal components.
            """)
            
            error_threshold = st.slider("Error threshold", 
                                       0.1, 10.0, 2.0, 0.1, key='error_threshold')
            
            error_outliers = reconstruction_error > error_threshold
            
            # Histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=reconstruction_error[labels == 'Normal'],
                name='Normal',
                opacity=0.7,
                marker_color='blue',
                nbinsx=30
            ))
            
            fig.add_trace(go.Histogram(
                x=reconstruction_error[labels == 'Outlier'],
                name='True Outliers',
                opacity=0.7,
                marker_color='red',
                nbinsx=30
            ))
            
            fig.add_vline(x=error_threshold, line_dash="dash", 
                         annotation_text="Threshold")
            
            fig.update_layout(
                title="Distribution of Reconstruction Errors",
                xaxis_title="Reconstruction Error",
                yaxis_title="Frequency",
                height=400,
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            tp_error = np.sum(error_outliers & true_outliers)
            fp_error = np.sum(error_outliers & ~true_outliers)
            precision_error = tp_error / (tp_error + fp_error) if (tp_error + fp_error) > 0 else 0
            recall_error = tp_error / true_outliers.sum() if true_outliers.sum() > 0 else 0
            
            col1, col2 = st.columns(2)
            col1.metric("Precision (Error)", f"{precision_error:.2f}")
            col2.metric("Recall (Error)", f"{recall_error:.2f}")
        
        with subtab3:
            st.markdown("### Combined Detection Strategy")
            
            st.markdown("Combine both distance and reconstruction error for robust detection")
            
            # Combined detection
            combined_outliers = detected_outliers | error_outliers
            
            # Confusion matrix
            cm = np.array([
                [np.sum(~combined_outliers & ~true_outliers), np.sum(combined_outliers & ~true_outliers)],
                [np.sum(~combined_outliers & true_outliers), np.sum(combined_outliers & true_outliers)]
            ])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Normal', 'Predicted Outlier'],
                y=['True Normal', 'True Outlier'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title="Confusion Matrix - Combined Method",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final metrics
            tp_combined = cm[1, 1]
            fp_combined = cm[0, 1]
            fn_combined = cm[1, 0]
            tn_combined = cm[0, 0]
            
            precision_combined = tp_combined / (tp_combined + fp_combined) if (tp_combined + fp_combined) > 0 else 0
            recall_combined = tp_combined / (tp_combined + fn_combined) if (tp_combined + fn_combined) > 0 else 0
            accuracy_combined = (tp_combined + tn_combined) / cm.sum()
            f1_combined = 2 * precision_combined * recall_combined / (precision_combined + recall_combined) if (precision_combined + recall_combined) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_combined:.3f}")
            col2.metric("Precision", f"{precision_combined:.3f}")
            col3.metric("Recall", f"{recall_combined:.3f}")
            col4.metric("F1 Score", f"{f1_combined:.3f}")

# ===== TAB 7: Advanced Visualizations =====
with tab7:
    st.markdown("## 🔬 Advanced PCA Visualizations")
    
    st.info("""
    This section provides advanced visualizations from the PCA notebooks:
    - **Loading Heatmap**: See all feature contributions across components
    - **Correlation Loading Plot**: Visualize variable correlations in PC space
    - **Communalities**: Check how well each variable is represented
    - **NIPALS Algorithm**: Watch the iterative PCA computation step-by-step
    """)
    
    adv_subtab1, adv_subtab2, adv_subtab3, adv_subtab4 = st.tabs([
        "📊 Loading Heatmap",
        "⭕ Correlation Plot",
        "📈 Communalities",
        "🔄 NIPALS Algorithm"
    ])
    
    with adv_subtab1:
        st.markdown("### Loading Heatmap - Feature Contributions")
        
        st.markdown("""
        The **loading heatmap** shows how each original feature contributes to each principal component.
        - **Red**: Positive contribution
        - **Blue**: Negative contribution  
        - **White**: Minimal contribution
        """)
        
        # Use McDonald's data for demonstration
        try:
            df_mcdonalds = pd.read_csv(Path(__file__).parent.parent / 'data' / 'macdonald.csv')
            
            numerical_cols = ['Energy (kCal)', 'Protein (g)', 'Total fat (g)', 
                            'Sat Fat (g)', 'Total carbohydrate (g)', 'Total Sugars (g)', 'Sodium (mg)']
            X = df_mcdonalds[numerical_cols].dropna()
            
            # Standardize and fit PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            n_comp = st.slider("Number of components to display", 2, min(10, X_scaled.shape[1]), 5, 
                              key='heatmap_ncomp')
            
            pca_heatmap = PCA(n_components=n_comp)
            pca_heatmap.fit(X_scaled)
            
            # Plot loading heatmap
            fig = plot_loading_heatmap(pca_heatmap, feature_names=numerical_cols, n_components=n_comp)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation guide
            st.markdown("### 📖 How to Interpret")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **High Positive Loading (Red)**
                - Feature increases when PC increases
                - Strong positive association
                
                **High Negative Loading (Blue)**
                - Feature decreases when PC increases
                - Strong negative association
                """)
            
            with col2:
                st.markdown("""
                **Near-Zero Loading (White)**
                - Feature has minimal contribution
                - Not important for this PC
                
                **Pattern Recognition**
                - Features with similar colors are correlated
                - Each PC captures different patterns
                """)
                
        except FileNotFoundError:
            st.error("❌ McDonald's dataset not found. Using synthetic data.")
            
            # Fallback to synthetic data
            from utils.data_generator import generate_correlated_data
            X, _, _ = generate_correlated_data(n=100, p=6, correlation=0.6, random_state=42)
            X_scaled = StandardScaler().fit_transform(X)
            
            pca_heatmap = PCA(n_components=5)
            pca_heatmap.fit(X_scaled)
            
            fig = plot_loading_heatmap(pca_heatmap, n_components=5)
            st.plotly_chart(fig, use_container_width=True)
    
    with adv_subtab2:
        st.markdown("### Correlation Loading Plot")
        
        st.markdown("""
        The **correlation loading plot** shows variables in the principal component space.
        - **Red circle (r=1.0)**: Perfect correlation boundary
        - **Orange circle (r=0.7)**: Strong correlation threshold
        - **Arrow length**: Strength of relationship with PCs
        - **Arrow direction**: Type of relationship
        - **Angle between arrows**: Correlation between variables
        """)
        
        # Use the same data from heatmap
        try:
            df_mcdonalds = pd.read_csv(Path(__file__).parent.parent / 'data' / 'macdonald.csv')
            numerical_cols = ['Energy (kCal)', 'Protein (g)', 'Total fat (g)', 
                            'Sat Fat (g)', 'Total carbohydrate (g)', 'Total Sugars (g)', 'Sodium (mg)']
            X = df_mcdonalds[numerical_cols].dropna()
            X_scaled = StandardScaler().fit_transform(X)
            
            pca_corr = PCA()
            pca_corr.fit(X_scaled)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Settings")
                pc1_idx = st.selectbox("X-axis (PC)", [f"PC{i+1}" for i in range(min(5, pca_corr.n_components_))], 
                                      index=0, key='corr_pc1')
                pc2_idx = st.selectbox("Y-axis (PC)", [f"PC{i+1}" for i in range(min(5, pca_corr.n_components_))], 
                                      index=1, key='corr_pc2')
                
                pc1 = int(pc1_idx[2:]) - 1
                pc2 = int(pc2_idx[2:]) - 1
            
            with col2:
                fig = plot_correlation_loading_plot(pca_corr, feature_names=numerical_cols, 
                                                    pc1=pc1, pc2=pc2)
                st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("### 🔍 Interpretation Tips")
            st.markdown("""
            - **Variables close to r=1 circle**: Well represented by these 2 PCs
            - **Variables far from origin**: Strongly associated with at least one PC
            - **Variables near each other**: Positively correlated
            - **Variables opposite each other**: Negatively correlated
            - **Variables at 90°**: Uncorrelated
            """)
            
        except FileNotFoundError:
            st.warning("Using synthetic data for demonstration")
            from utils.data_generator import generate_correlated_data
            X, _, _ = generate_correlated_data(n=100, p=6, correlation=0.6, random_state=42)
            X_scaled = StandardScaler().fit_transform(X)
            pca_corr = PCA()
            pca_corr.fit(X_scaled)
            fig = plot_correlation_loading_plot(pca_corr)
            st.plotly_chart(fig, use_container_width=True)
    
    with adv_subtab3:
        st.markdown("### Communalities - Variable Representation Quality")
        
        st.markdown("""
        **Communalities** measure how well each variable is represented by the selected principal components.
        - **Communality = Sum of squared loadings** for each variable
        - **Value near 1.0**: Variable is well explained by PCs
        - **Value near 0.0**: Variable is poorly explained
        
        **Color coding:**
        - 🟢 **Green (>0.7)**: Good representation
        - 🟠 **Orange (0.5-0.7)**: Acceptable representation
        - 🔴 **Red (<0.5)**: Poor representation - may need more components
        """)
        
        try:
            df_mcdonalds = pd.read_csv(Path(__file__).parent.parent / 'data' / 'macdonald.csv')
            numerical_cols = ['Energy (kCal)', 'Protein (g)', 'Total fat (g)', 
                            'Sat Fat (g)', 'Total carbohydrate (g)', 'Total Sugars (g)', 'Sodium (mg)']
            X = df_mcdonalds[numerical_cols].dropna()
            X_scaled = StandardScaler().fit_transform(X)
            
            n_comp_comm = st.slider("Number of components to include", 2, min(6, X_scaled.shape[1]), 3,
                                   key='comm_ncomp')
            
            pca_comm = PCA()
            pca_comm.fit(X_scaled)
            
            fig = plot_communalities(pca_comm, feature_names=numerical_cols, n_components=n_comp_comm)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show actual values in table
            loadings = pca_comm.components_.T
            communalities = np.sum(loadings[:, :n_comp_comm]**2, axis=1)
            
            comm_df = pd.DataFrame({
                'Feature': numerical_cols,
                'Communality': communalities,
                'Quality': ['Good' if c > 0.7 else 'Acceptable' if c > 0.5 else 'Poor' for c in communalities]
            })
            
            st.markdown("### 📊 Communality Values")
            st.dataframe(comm_df.style.background_gradient(subset=['Communality'], cmap='RdYlGn', vmin=0, vmax=1),
                        use_container_width=True)
            
            avg_comm = communalities.mean()
            if avg_comm > 0.7:
                st.success(f"✅ Average communality: {avg_comm:.3f} - Excellent representation!")
            elif avg_comm > 0.5:
                st.info(f"ℹ️ Average communality: {avg_comm:.3f} - Good representation")
            else:
                st.warning(f"⚠️ Average communality: {avg_comm:.3f} - Consider adding more components")
                
        except FileNotFoundError:
            st.warning("Using synthetic data for demonstration")
            from utils.data_generator import generate_correlated_data
            X, _, _ = generate_correlated_data(n=100, p=6, correlation=0.6, random_state=42)
            X_scaled = StandardScaler().fit_transform(X)
            pca_comm = PCA()
            pca_comm.fit(X_scaled)
            fig = plot_communalities(pca_comm, n_components=3)
            st.plotly_chart(fig, use_container_width=True)
    
    with adv_subtab4:
        st.markdown("### NIPALS Algorithm - Iterative PCA")
        
        st.markdown("""
        **NIPALS** (Non-linear Iterative Partial Least Squares) is an iterative algorithm for computing PCA:
        
        **How it works:**
        1. Initialize score vector **t** (random or data column)
        2. Compute loading **p** by regressing X onto t
        3. Normalize p
        4. Compute new scores **t** by regressing X onto p
        5. Repeat until convergence
        
        **Advantages:**
        - Can handle missing data
        - Memory efficient for large datasets
        - Computes one PC at a time
        - Educational: shows how PCA "finds" directions of maximum variance
        """)
        
        st.markdown("### 🎮 Interactive Demo")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Data Generation")
            angle_nipals = st.slider("Rotation angle (degrees)", 0, 90, 35, 5, key='nipals_angle')
            var_x_nipals = st.slider("Variance X", 1.0, 10.0, 5.0, 0.5, key='nipals_varx')
            var_y_nipals = st.slider("Variance Y", 0.5, 5.0, 1.0, 0.5, key='nipals_vary')
            n_samples_nipals = st.slider("Number of samples", 50, 200, 100, 10, key='nipals_n')
            
            if st.button("Run NIPALS", type="primary", key='run_nipals'):
                # Generate 2D data
                np.random.seed(42)
                angle_rad = np.radians(angle_nipals)
                
                # Create rotation matrix
                R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                             [np.sin(angle_rad), np.cos(angle_rad)]])
                
                # Generate data
                data_orig = np.random.randn(n_samples_nipals, 2) * np.array([np.sqrt(var_x_nipals), np.sqrt(var_y_nipals)])
                X_nipals = data_orig @ R.T
                
                # Run NIPALS
                history, final_t, final_p = nipals_algorithm(X_nipals, max_iter=50, tol=1e-6)
                
                st.session_state.nipals_history = history
                st.session_state.nipals_X = X_nipals
                st.session_state.nipals_converged = True
                
                st.success(f"✅ Converged in {len(history)} iterations!")
                st.metric("Final Angle", f"{history[-1]['angle']:.1f}°")
                st.metric("Final Change", f"{history[-1]['change']:.2e}")
        
        with col2:
            if 'nipals_converged' in st.session_state and st.session_state.nipals_converged:
                # Show convergence plot
                fig_conv = plot_nipals_convergence(st.session_state.nipals_history)
                st.plotly_chart(fig_conv, use_container_width=True)
        
        # Interactive iteration viewer
        if 'nipals_converged' in st.session_state and st.session_state.nipals_converged:
            st.markdown("### 📹 Step-by-Step Visualization")
            
            iteration = st.slider(
                "Select Iteration",
                1,
                len(st.session_state.nipals_history),
                1,
                key='nipals_iter_slider'
            )
            
            fig_step = plot_nipals_step(
                st.session_state.nipals_X,
                st.session_state.nipals_history,
                iteration - 1
            )
            st.plotly_chart(fig_step, use_container_width=True)
            
            # Show numerical details
            state = st.session_state.nipals_history[iteration - 1]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Iteration", state['iteration'])
            col2.metric("Change", f"{state['change']:.2e}")
            col3.metric("Angle", f"{state['angle']:.2f}°")
            
            st.markdown(f"""
            **Loading vector p:** [{state['p'][0]:.6f}, {state['p'][1]:.6f}]
            
            - **Red arrow**: Current PC1 direction estimate
            - **Gray dashed lines**: Projections of data onto PC1
            - **Red dots**: Projected points on PC1
            """)

with st.sidebar:
    st.markdown("### 📖 About PCA")
    st.info("""
    **Principal Component Analysis** is a dimensionality reduction technique that:
    
    **Algorithm:**
    1. **Standardize** data (zero mean, unit variance)
    2. **Compute** covariance matrix
    3. **Find** eigenvectors and eigenvalues
    4. **Sort** by eigenvalues (descending)
    5. **Project** data onto top k eigenvectors
    
    **Key Concepts:**
    - **PC1**: Direction of maximum variance
    - **PC2**: Direction of max variance orthogonal to PC1
    - **Variance Explained**: How much information each PC captures
    - **Loadings**: Feature contributions to PCs
    - **Scores**: Data points in PC space
    
    **Use Cases:**
    - Dimensionality reduction
    - Data visualization (2D/3D)
    - Noise filtering
    - Feature extraction
    - Outlier detection
    - Multicollinearity handling
    
    **Interpretation:**
    - PCs are **linear combinations** of original features
    - Features with high loadings strongly influence that PC
    - Similar items cluster together in PC space
    """)
    
    st.markdown("### 🎯 Quick Tips")
    st.success("""
    - Use **Scree Plot** to choose number of components
    - Aim for **95% cumulative variance** explained
    - **Standardize** data before PCA
    - Check **loadings** to interpret PCs
    - Use **biplots** to see features and observations together
    """)
