"""
TTK4260-2026 Interactive Machine Learning Course App
=====================================================
A comprehensive Streamlit application for exploring ML algorithms and statistical methods.

Author: TTK4260 Course Team
Date: February 2026
"""

import streamlit as st
import numpy as np
import pandas as pd

# Configure page
st.set_page_config(
    page_title="TTK4260 ML Course",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🎓 TTK4260 Machine Learning Course</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Interactive Exploration of Statistical Learning Methods</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=TTK4260", use_container_width=True)
    st.markdown("## 🎯 Navigation")
    st.markdown("Select a topic from the pages above to explore different algorithms and concepts.")
    
    st.markdown("---")
    st.markdown("## 📚 Course Topics")
    st.markdown("""
    1. **Least Squares** - OLS & Optimization
    2. **Maximum Likelihood** - Parameter Estimation
    3. **MLR & Regularization** - Ridge, Lasso, ElasticNet
    4. **PCA Explorer** - Interactive PCA Concepts
    5. **PCA Application** - McDonald's Menu Analysis
    6. **PCR Interactive** - Multicollinearity Demo
    7. **PCR Application** - Norwegian Energy Data
    8. **Performance Metrics** - Model Evaluation
    9. **ICA Pulse Detection** - Heart Rate from Video
    """)
    
    st.markdown("---")
    st.markdown("## ⚙️ Global Settings")
    
    # Random seed
    random_seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999, help="Set random seed for reproducibility")
    np.random.seed(random_seed)
    
    # Store in session state
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = random_seed
    
    st.markdown("---")
    st.markdown("### 👨‍🏫 About")
    st.info("This app is designed for TTK4260 students to interactively explore machine learning concepts through visualization and experimentation.")

# Main content area
st.markdown("## 🚀 Welcome to the Interactive ML Lab!")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Data Exploration")
    st.markdown("""
    - Generate synthetic datasets
    - Upload your own data
    - Control noise and correlation
    - Visualize data distributions
    """)

with col2:
    st.markdown("### 🤖 Model Training")
    st.markdown("""
    - Implement algorithms from scratch
    - Compare multiple methods
    - Tune hyperparameters
    - Cross-validation
    """)

with col3:
    st.markdown("### 📈 Visualization")
    st.markdown("""
    - Interactive 3D plots
    - Real-time parameter adjustment
    - Side-by-side comparisons
    - Performance metrics
    """)

st.markdown("---")

# Quick start guide
st.markdown("## 🎯 Quick Start Guide")

with st.expander("📖 How to use this app", expanded=True):
    st.markdown("""
    ### Getting Started
    
    1. **Select a topic** from the sidebar or the pages above
    2. **Adjust parameters** using sliders and input controls
    3. **Generate or upload data** as needed
    4. **Explore visualizations** and compare different methods
    5. **Experiment freely** - all changes are temporary!
    
    ### Features
    
    - **🎮 Interactive Controls**: Real-time parameter adjustment
    - **📊 Rich Visualizations**: 2D, 3D, and animated plots
    - **🔄 Model Comparison**: Side-by-side algorithm evaluation
    - **💾 Data Options**: Synthetic generation or file upload
    - **📥 Export Results**: Download plots and metrics
    
    ### Tips
    
    - Use the **sidebar** for global settings and navigation
    - **Hover** over info icons (ℹ️) for explanations
    - Try **different datasets** to see how algorithms behave
    - **Compare methods** to understand trade-offs
    """)

# Dataset preview section
st.markdown("## 📦 Dataset Preview")

dataset_option = st.selectbox(
    "Choose a sample dataset or generate synthetic data:",
    ["None", "Linear Regression Sample", "Correlated Features", "High Dimensional", "Real Dataset (CityTemp)"]
)

if dataset_option != "None":
    if dataset_option == "Linear Regression Sample":
        n_samples = st.slider("Number of samples", 20, 200, 50)
        noise = st.slider("Noise level", 0.0, 5.0, 1.0)
        
        X = np.linspace(0, 10, n_samples)
        y = 2.5 * X + 1.0 + noise * np.random.randn(n_samples)
        
        df = pd.DataFrame({'X': X, 'y': y})
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 of {n_samples} samples")
        
        with col2:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data'))
            fig.update_layout(title="Linear Regression Sample", xaxis_title="X", yaxis_title="y", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    elif dataset_option == "Correlated Features":
        st.info("Navigate to the **Multiple Linear Regression** page to explore correlated features in detail!")
    
    elif dataset_option == "High Dimensional":
        st.info("Navigate to the **Principal Component Analysis** page to explore dimensionality reduction!")
    
    elif dataset_option == "Real Dataset (CityTemp)":
        st.info("Navigate to the **Principal Component Analysis** page to analyze real temperature data!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>TTK4260-2026 | Interactive Machine Learning Course | Norwegian University of Science and Technology</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
