"""
PCA Application - McDonald's Menu Nutrition
============================================
Comprehensive PCA analysis on real-world food nutrition data.

Based on PCA_McDonald_DA.ipynb
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
from scipy import stats

st.set_page_config(page_title="PCA Application", page_icon="🍔", layout="wide")

# ===== STYLING =====
COLORS = {
    'mcd_red': '#DA291C',
    'mcd_yellow': '#FFC72C',
    'blue': '#4169E1',
    'orange': '#FF8C00',
    'green': '#228B22',
    'purple': '#800080',
    'teal': '#4ECDC4',
    'gray': '#888888'
}

CATEGORY_COLORS = {
    'Beverages': '#FF6B6B',
    'Breakfast': '#FFA500',
    'Chicken & Fish': '#FFD700',
    'Beef & Pork': '#8B4513',
    'Beverages Smoothies McCafe': '#9370DB',
    'Desserts': '#FF69B4',
    'Salads': '#90EE90',
    'Snacks & Sides': '#20B2AA'
}

# ===== LOAD DATA =====
@st.cache_data
def load_mcdonalds_data():
    """Load McDonald's menu nutrition data"""
    data_path = Path(__file__).parent.parent / "data"
    df = pd.read_csv(data_path / "macdonald.csv")
    
    # Define columns
    nutritional_cols = ['Energy (kCal)', 'Protein (g)', 'Total fat (g)', 'Sat Fat (g)',
                        'Trans fat (g)', 'Cholesterols (mg)', 'Total carbohydrate (g)',
                        'Total Sugars (g)', 'Added Sugars (g)', 'Sodium (mg)']
    
    short_names = ['Energy', 'Protein', 'TotalFat', 'SatFat', 'TransFat', 
                   'Cholest', 'Carbs', 'Sugars', 'AddSugar', 'Sodium']
    
    # Clean data
    df_clean = df.dropna(subset=nutritional_cols)
    X_raw = df_clean[nutritional_cols].values
    item_names = df_clean['Menu Items'].values
    categories = df_clean['Menu Category'].values
    
    return df_clean, X_raw, item_names, categories, nutritional_cols, short_names

df, X_raw, item_names, categories, nutritional_cols, short_names = load_mcdonalds_data()

# ===== TITLE =====
st.title("🍔 PCA Application — McDonald's Menu Nutrition")
st.markdown("### A Complete Tutorial on Principal Component Analysis")

st.markdown("""
**The Dataset:** McDonald's India Menu with 141 items across 7 categories

**10 Nutritional Variables:** Energy, Protein, Fats (Total, Saturated, Trans), 
Cholesterol, Carbohydrates, Sugars (Total, Added), Sodium

**The Goal:** Use PCA to understand the nutritional structure of the menu and 
discover patterns in how items cluster.
""")

# ===== TABS =====
tabs = st.tabs([
    "📊 Data Overview",
    "🔬 Standardization",
    "📈 PCA Results",
    "🎯 Score & Corr Loading",
    "🔍 Biplot",
    "📍 Loading Analysis",
    "🚨 Outlier Detection",
    "🔧 Contribution Plots",
    "💡 Interpretation"
])

# ===== TAB 0: Data Overview =====
with tabs[0]:
    st.markdown("## Dataset Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Shape:** {df.shape[0]} menu items × {len(nutritional_cols)} nutritional variables")
        
        # Category distribution
        cat_counts = df['Menu Category'].value_counts()
        fig = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            title='Menu Category Distribution',
            color=cat_counts.index,
            color_discrete_map=CATEGORY_COLORS,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Categories:**")
        for cat, count in cat_counts.items():
            st.write(f"• {cat}: {count} items ({100*count/len(df):.1f}%)")
    
    st.markdown("### Sample Data (First 10 Items)")
    display_cols = ['Menu Items', 'Menu Category'] + nutritional_cols[:5]
    st.dataframe(df[display_cols].head(10), use_container_width=True)
    
    st.markdown("### Statistical Summary")
    st.dataframe(df[nutritional_cols].describe().round(2), use_container_width=True)

# ===== TAB 1: Standardization =====
with tabs[1]:
    st.markdown("## Why Standardization is Essential")
    
    st.markdown("""
    PCA finds directions of maximum **variance**. Without standardization:
    - Variables with larger scales dominate (e.g., Sodium in mg vs Trans fat in g)
    - The analysis becomes meaningless
    
    After standardization:
    - Each variable has mean = 0 and standard deviation = 1
    - All variables contribute equally to the analysis
    """)
    
    # Show before/after comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Before Standardization")
        scale_df = pd.DataFrame({
            'Variable': short_names,
            'Mean': X_raw.mean(axis=0),
            'Std': X_raw.std(axis=0)
        })
        st.dataframe(scale_df.style.format({'Mean': '{:.2f}', 'Std': '{:.2f}'}), 
                     use_container_width=True)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    with col2:
        st.markdown("### After Standardization")
        scale_df_after = pd.DataFrame({
            'Variable': short_names,
            'Mean': X.mean(axis=0),
            'Std': X.std(axis=0)
        })
        st.dataframe(scale_df_after.style.format({'Mean': '{:.10f}', 'Std': '{:.2f}'}), 
                     use_container_width=True)
    
    st.markdown("### Correlation Matrix")
    st.markdown("Shows relationships between nutritional variables (red = positive, blue = negative)")
    
    corr_matrix = np.corrcoef(X.T)
    fig = px.imshow(
        corr_matrix,
        x=short_names,
        y=short_names,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Correlation Matrix',
        text_auto='.2f'
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Key Patterns:**
    - **Energy cluster:** Energy, TotalFat, SatFat, Cholesterol strongly correlated (red)
    - **Sugar cluster:** Sugars and AddSugar nearly perfectly correlated
    - **Sodium is semi-independent:** Moderate correlation with fats, weak with sugars
    """)

# ===== TAB 2: PCA Results =====
with tabs[2]:
    st.markdown("## PCA: Extracting Principal Components")
    
    # Perform PCA
    pca = PCA()
    T = pca.fit_transform(X)
    P = pca.components_.T  # loadings as columns
    explained_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    
    st.markdown("### Scree Plot & Cumulative Variance")
    
    cum_var = np.cumsum(explained_var)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Scree Plot (Variance per PC)', 'Cumulative Variance Explained']
    )
    
    # Scree plot
    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(explained_var))],
        y=explained_var * 100,
        marker_color=COLORS['blue'],
        name='Variance',
        text=[f'{v:.1f}%' for v in explained_var[:5]*100] + ['']*(len(explained_var)-5),
        textposition='outside'
    ), row=1, col=1)
    
    # Cumulative plot
    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(explained_var))],
        y=cum_var * 100,
        mode='lines+markers',
        marker=dict(color=COLORS['orange'], size=10),
        line=dict(color=COLORS['orange']),
        name='Cumulative'
    ), row=1, col=2)
    
    # Reference lines
    fig.add_hline(y=80, line_dash='dash', line_color=COLORS['mcd_red'], 
                  row=1, col=2, annotation_text='80%')
    fig.add_hline(y=90, line_dash='dash', line_color=COLORS['green'], 
                  row=1, col=2, annotation_text='90%')
    
    fig.update_yaxes(title_text='Variance Explained (%)', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Variance (%)', row=1, col=2)
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Variance table
    st.markdown("### Explained Variance by Component")
    var_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained_var))],
        'Eigenvalue': eigenvalues,
        'Variance (%)': explained_var * 100,
        'Cumulative (%)': cum_var * 100
    })
    st.dataframe(var_df.style.format({
        'Eigenvalue': '{:.3f}',
        'Variance (%)': '{:.1f}',
        'Cumulative (%)': '{:.1f}'
    }), use_container_width=True)
    
    st.success(f"""
    **Key Results:**
    - PC1 captures {explained_var[0]*100:.1f}% — Overall caloric/nutritional density
    - PC2 captures {explained_var[1]*100:.1f}% — Sweet vs Savory distinction
    - First 2 PCs explain {cum_var[1]*100:.1f}% of variance
    - First 3 PCs explain {cum_var[2]*100:.1f}% of variance
    """)

# ===== TAB 3: Score & Correlation Loading (Side by Side) =====
with tabs[3]:
    st.markdown("## Score Plot & Correlation Loading Plot")
    st.markdown("**Side-by-side view for easy comparison**")
    
    # Compute PCA (fresh computation to ensure consistency)
    pca = PCA()
    T = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    P = pca.components_.T
    
    # Compute correlation loadings
    corr_loadings = P * np.sqrt(eigenvalues)
    
    # Interactive PC selection (shared for both plots)
    col1, col2 = st.columns(2)
    with col1:
        pc_x = st.selectbox("X-axis", [f"PC{i+1}" for i in range(min(6, len(explained_var)))], 
                           index=0, key='score_corr_x')
    with col2:
        pc_y = st.selectbox("Y-axis", [f"PC{i+1}" for i in range(min(6, len(explained_var)))], 
                           index=1, key='score_corr_y')
    
    pc_x_idx = int(pc_x[2:]) - 1
    pc_y_idx = int(pc_y[2:]) - 1
    
    # Debug info
    st.caption(f"Debug: Plotting {pc_x} (index {pc_x_idx}) vs {pc_y} (index {pc_y_idx})")
    st.caption(f"Debug: First variable ({short_names[0]}) corr loadings = ({corr_loadings[0, pc_x_idx]:.3f}, {corr_loadings[0, pc_y_idx]:.3f})")
    
    # Create side-by-side plots
    st.markdown("### Score Plot & Correlation Loading Plot")
    
    # Create both figures first
    # Score plot
    fig_score = go.Figure()
    
    for cat in df['Menu Category'].unique():
        mask = categories == cat
        fig_score.add_trace(go.Scatter(
            x=T[mask, pc_x_idx],
            y=T[mask, pc_y_idx],
            mode='markers',
            marker=dict(size=8, color=CATEGORY_COLORS.get(cat, COLORS['gray']), 
                       line=dict(width=0.5, color='white')),
            name=cat,
            text=item_names[mask],
            hovertemplate='%{text}<br>'+f'{pc_x}: %{{x:.2f}}<br>{pc_y}: %{{y:.2f}}<extra></extra>'
        ))
    
    fig_score.add_hline(y=0, line_dash='dot', line_color='gray', line_width=0.5)
    fig_score.add_vline(x=0, line_dash='dot', line_color='gray', line_width=0.5)
    
    fig_score.update_layout(
        title=f'Score Plot: {pc_x} vs {pc_y}',
        xaxis_title=f'{pc_x} ({explained_var[pc_x_idx]*100:.1f}%)',
        yaxis_title=f'{pc_y} ({explained_var[pc_y_idx]*100:.1f}%)',
        height=600,
        hovermode='closest',
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50)
    )
    
    # Correlation loading plot
    fig_corr = go.Figure()
    
    # Unit circles
    theta = np.linspace(0, 2*np.pi, 100)
    fig_corr.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color='gray', dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    fig_corr.add_trace(go.Scatter(
        x=0.5*np.cos(theta), y=0.5*np.sin(theta),
        mode='lines', line=dict(color='lightgray', dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Variable arrows
    for i, name in enumerate(short_names):
        cx = corr_loadings[i, pc_x_idx]
        cy = corr_loadings[i, pc_y_idx]
        
        fig_corr.add_annotation(
            x=cx, y=cy, ax=0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=2, arrowcolor=COLORS['mcd_red']
        )
        fig_corr.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode='markers+text',
            marker=dict(size=10, color=COLORS['mcd_red']),
            text=[name],
            textposition='top center',
            textfont=dict(size=10),
            showlegend=False,
            hovertemplate=f'{name}<br>{pc_x}: {cx:.3f}<br>{pc_y}: {cy:.3f}<extra></extra>'
        ))
    
    fig_corr.add_hline(y=0, line_color='black', line_width=0.5)
    fig_corr.add_vline(x=0, line_color='black', line_width=0.5)
    
    fig_corr.update_layout(
        title=f'Correlation Loading: {pc_x} vs {pc_y}',
        xaxis=dict(title=f'Correlation with {pc_x}', 
                   range=[-1.1, 1.1], constrain='domain', zeroline=True),
        yaxis=dict(title=f'Correlation with {pc_y}', 
                   range=[-1.1, 1.1], scaleanchor='x', scaleratio=1, zeroline=True),
        height=600,
        margin=dict(l=50, r=20, t=40, b=50)
    )
    
    # Display side by side
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_score, use_container_width=True)
    with col_right:
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Interpretation
    if pc_x_idx == 0 and pc_y_idx == 1:
        st.success("""
        **Joint Interpretation — PC1 vs PC2:**
        
        **Left panel (Score Plot):**
        - **Horizontal:** Light items (left) → Hearty meals (right)
        - **Vertical:** Savory (bottom) → Sweet (top)
        - **Desserts** cluster at top (pink) — high sugar
        - **Burgers/Chicken** on right (brown/yellow) — high calories
        
        **Right panel (Correlation Loading):**
        - **Right arrows:** Energy, Fats, Protein → explain rightward items
        - **Top arrows:** Sugars → explain upward items
        - **Arrow alignment:** Energy/TotalFat/Protein point same direction (correlated)
        
        **Combined insight:** Items positioned in the direction of an arrow have high values for that variable
        """)

# ===== TAB 4: Biplot =====
with tabs[4]:
    st.markdown("## Biplot: Scores + Loadings Combined")
    st.markdown("Shows both observations (menu items) and variables (nutritional factors) in the same plot")
    
    # Compute PCA
    pca = PCA()
    T = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    P = pca.components_.T
    
    # PC selection
    col1, col2 = st.columns(2)
    with col1:
        bi_pc_x = st.selectbox("X-axis", [f"PC{i+1}" for i in range(min(6, len(explained_var)))], 
                              index=0, key='biplot_x')
    with col2:
        bi_pc_y = st.selectbox("Y-axis", [f"PC{i+1}" for i in range(min(6, len(explained_var)))], 
                              index=1, key='biplot_y')
    
    bi_x_idx = int(bi_pc_x[2:]) - 1
    bi_y_idx = int(bi_pc_y[2:]) - 1
    
    # Create biplot
    fig = go.Figure()
    
    # Add scores (observations) by category
    for cat in df['Menu Category'].unique():
        mask = categories == cat
        fig.add_trace(go.Scatter(
            x=T[mask, bi_x_idx],
            y=T[mask, bi_y_idx],
            mode='markers',
            marker=dict(size=8, color=CATEGORY_COLORS.get(cat, COLORS['gray']), 
                       opacity=0.6, line=dict(width=0.5, color='white')),
            name=cat,
            text=item_names[mask],
            hovertemplate='%{text}<br>'+f'{bi_pc_x}: %{{x:.2f}}<br>{bi_pc_y}: %{{y:.2f}}<extra></extra>'
        ))
    
    # Add origin lines
    fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=0.5)
    fig.add_vline(x=0, line_dash='dot', line_color='gray', line_width=0.5)
    
    # Add loading arrows (scaled for visibility)
    scale = 4
    for i, name in enumerate(short_names):
        px_val = P[i, bi_x_idx] * scale
        py_val = P[i, bi_y_idx] * scale
        
        # Arrow
        fig.add_annotation(
            x=px_val, y=py_val, ax=0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=3, arrowcolor='black'
        )
        # Label
        fig.add_trace(go.Scatter(
            x=[px_val * 1.1], y=[py_val * 1.1],
            mode='text',
            text=[name],
            textfont=dict(size=11, color='black', family='Arial Black'),
            showlegend=False,
            hovertemplate=f'{name}<br>Loading {bi_pc_x}: {P[i, bi_x_idx]:.3f}<br>Loading {bi_pc_y}: {P[i, bi_y_idx]:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Biplot: {bi_pc_x} vs {bi_pc_y} (Scores + Loadings)',
        xaxis_title=f'{bi_pc_x} ({explained_var[bi_x_idx]*100:.1f}%)',
        yaxis_title=f'{bi_pc_y} ({explained_var[bi_y_idx]*100:.1f}%)',
        height=700,
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **💡 How to Read a Biplot:**
    
    1. **Points (colored dots):** Menu items positioned by their PC scores
    2. **Arrows (black):** Variables (nutritional factors) showing their contribution
    3. **Arrow direction:** Items in that direction have HIGH values for that variable
    4. **Arrow length:** Longer = more important for these PCs
    5. **Arrows pointing same direction:** Positively correlated variables
    6. **Arrows pointing opposite:** Negatively correlated variables
    
    **Example:** If "Chicken Maharaja Mac" is positioned toward the Energy/TotalFat arrows, 
    it means this burger is high in both calories and fat.
    """)
    
    if bi_x_idx == 0 and bi_y_idx == 1:
        st.success("""
        **PC1 vs PC2 Biplot Insights:**
        
        - **Energy, TotalFat, Protein arrows** point right → High-calorie items on the right
        - **Sugars, AddSugar arrows** point up → Sweet items at the top
        - **Desserts** (pink dots) positioned near Sugar arrows → Confirms high sugar content
        - **Large burgers** positioned near Energy/Fat arrows → Confirms high calorie/fat
        - **Salads** (green dots) opposite to arrows → Low in everything (healthy options)
        """)

# ===== TAB 5: Loading Analysis =====
with tabs[5]:
    st.markdown("## Loading Analysis: Variable Contributions")
    st.markdown("Shows which nutritional variables contribute to each PC")
    
    # Compute PCA
    pca = PCA()
    pca.fit(X)
    P = pca.components_.T
    explained_var = pca.explained_variance_ratio_
    
    # Loading bar charts for first 4 PCs
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'PC{i+1} ({explained_var[i]*100:.1f}%)' for i in range(4)]
    )
    
    colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]
    
    for i in range(4):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(
            go.Bar(x=short_names, y=P[:, i], marker_color=colors_list[i], 
                   showlegend=False, hovertemplate='%{x}: %{y:.3f}<extra></extra>'),
            row=row, col=col
        )
        fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=0.5, 
                      row=row, col=col)
    
    fig.update_layout(height=600, title_text='Loading Profiles by Principal Component')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Loading Matrix Heatmap")
    loading_df = pd.DataFrame(
        P[:, :6],
        index=short_names,
        columns=[f'PC{i+1}' for i in range(6)]
    )
    
    fig = px.imshow(
        loading_df.T,
        color_continuous_scale='RdBu_r',
        zmin=-0.6, zmax=0.6,
        text_auto='.2f',
        title='Loading Matrix (First 6 PCs)',
        aspect='auto'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **Component Interpretation:**
    
    - **PC1 (Hearty/Filling):** High loadings on Energy, TotalFat, Protein, Sodium
      → Separates hearty meals from light items
    
    - **PC2 (Sweet vs Savory):** High Sugars/AddSugar, low Protein/Fat
      → Separates desserts/drinks from main courses
    
    - **PC3 (Trans Fat):** Dominated by TransFat
      → Flags items with partially hydrogenated oils
    
    - **PC4 (Cholesterol):** High Cholesterol, negative SatFat
      → Identifies egg-based breakfast items
    """)

# ===== TAB 6: Outlier Detection =====
with tabs[6]:
    st.markdown("## Outlier Detection: Influence Plot")
    st.markdown("Uses Hotelling's T² and Q-residual (SPE) to identify unusual menu items")
    
    # Compute PCA
    pca = PCA(n_components=min(6, X.shape[1]))
    T = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    P = pca.components_.T
    
    # Number of components for outlier detection
    k = st.slider("Number of components for model", 2, min(6, X.shape[1]), 3, key='outlier_k')
    
    # Reconstruct and compute residuals
    T_k = T[:, :k]
    P_k = P[:, :k]
    X_reconstructed = T_k @ P_k.T
    E = X - X_reconstructed
    
    # Compute T² (Hotelling's statistic)
    T2 = np.sum((T_k ** 2) / eigenvalues[:k], axis=1)
    
    # Compute Q-residual (SPE)
    Q = np.sum(E ** 2, axis=1)
    
    # Control limits (using F-distribution approximation)
    n, p = X.shape
    alpha = 0.05
    T2_limit = stats.f.ppf(1-alpha, k, n-k) * k * (n-1) / (n-k)
    
    # Q limit (using chi-squared approximation)
    theta1 = np.sum(eigenvalues[k:])
    theta2 = np.sum(eigenvalues[k:] ** 2)
    theta3 = np.sum(eigenvalues[k:] ** 3)
    h0 = 1 - (2*theta1*theta3)/(3*theta2**2)
    ca = stats.norm.ppf(1-alpha)
    Q_limit = theta1 * (1 + (ca*np.sqrt(2*theta2)*h0)/theta1 + (theta2*h0*(h0-1))/(theta1**2)) ** (1/h0)
    
    # Classify points
    colors_list = []
    for t2, q in zip(T2, Q):
        if t2 > T2_limit and q > Q_limit:
            colors_list.append('red')       # Both high: extreme outlier
        elif t2 > T2_limit:
            colors_list.append('orange')    # High T²: extreme in model space
        elif q > Q_limit:
            colors_list.append('purple')    # High Q: doesn't fit model
        else:
            colors_list.append(COLORS['blue'])  # Normal
    
    # Create T² vs Q plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=T2, y=Q,
        mode='markers',
        marker=dict(size=8, color=colors_list, opacity=0.7,
                    line=dict(width=1, color='black')),
        text=item_names,
        hovertemplate='<b>%{text}</b><br>T²: %{x:.2f}<br>Q: %{y:.2f}<extra></extra>'
    ))
    
    # Add control limits
    fig.add_vline(x=T2_limit, line_dash='dash', line_color='red', line_width=2,
                  annotation_text=f'T² limit ({int(T2_limit)})')
    fig.add_hline(y=Q_limit, line_dash='dash', line_color='red', line_width=2,
                  annotation_text=f'Q limit ({Q_limit:.1f})')
    
    # Add quadrant labels
    fig.add_annotation(x=T2_limit*0.4, y=Q_limit*0.4, text='NORMAL',
                       showarrow=False, font=dict(size=16, color='green', family='Arial Black'))
    fig.add_annotation(x=T2.max()*0.7, y=Q_limit*0.4, text='High Leverage<br>(Extreme but typical)',
                       showarrow=False, font=dict(size=11, color='orange'))
    fig.add_annotation(x=T2_limit*0.4, y=Q.max()*0.7, text='Unusual Pattern<br>(Doesn\'t fit model)',
                       showarrow=False, font=dict(size=11, color='purple'))
    fig.add_annotation(x=T2.max()*0.7, y=Q.max()*0.7, text='OUTLIER<br>(Both)',
                       showarrow=False, font=dict(size=13, color='red', family='Arial Black'))
    
    fig.update_layout(
        title=f'Influence Plot: T² vs Q (with k={k} components)',
        xaxis_title="Hotelling's T² (variation IN model space)",
        yaxis_title='Q Residual / SPE (variation OUTSIDE model)',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # List outliers
    col1, col2, col3 = st.columns(3)
    
    outliers_both = [item_names[i] for i in range(len(T2)) if T2[i] > T2_limit and Q[i] > Q_limit]
    outliers_t2 = [item_names[i] for i in range(len(T2)) if T2[i] > T2_limit and Q[i] <= Q_limit]
    outliers_q = [item_names[i] for i in range(len(T2)) if T2[i] <= T2_limit and Q[i] > Q_limit]
    
    with col1:
        st.markdown(f"### 🔴 Extreme Outliers ({len(outliers_both)})")
        st.markdown("*High T² AND Q — investigate immediately*")
        for item in outliers_both[:10]:
            st.markdown(f"- {item}")
    
    with col2:
        st.markdown(f"### 🟠 High Leverage ({len(outliers_t2)})")
        st.markdown("*Extreme but follows pattern*")
        for item in outliers_t2[:10]:
            st.markdown(f"- {item}")
    
    with col3:
        st.markdown(f"### 🟣 Unusual Pattern ({len(outliers_q)})")
        st.markdown("*Doesn't fit model well*")
        for item in outliers_q[:10]:
            st.markdown(f"- {item}")
    
    st.info("""
    **Understanding the Quadrants:**
    
    - **Normal (Blue):** Typical menu items that fit the model well
    - **High T² (Orange):** Extreme items but with expected proportions (e.g., "supersized" burgers)
    - **High Q (Purple):** Items with unusual nutritional ratios that don't match other items
    - **Both High (Red):** Extreme AND unusual — worth investigating for data errors or truly unique items
    """)

# ===== TAB 7: Contribution Plots =====
with tabs[7]:
    st.markdown("## Contribution Plots: Fault Diagnosis")
    st.markdown("When an item is an outlier, which variables are responsible?")
    
    # Compute PCA
    pca = PCA(n_components=min(6, X.shape[1]))
    T = pca.fit_transform(X)
    P = pca.components_.T
    eigenvalues = pca.explained_variance_
    
    # Select item to analyze
    selected_item = st.selectbox("Select menu item to analyze:", item_names, 
                                 index=0, key='contrib_item')
    item_idx = np.where(item_names == selected_item)[0][0]
    
    # Number of components
    k_contrib = st.slider("Number of components", 2, min(6, X.shape[1]), 3, key='contrib_k')
    
    # Reconstruct
    T_k = T[:, :k_contrib]
    P_k = P[:, :k_contrib]
    X_reconstructed = T_k @ P_k.T
    E = X - X_reconstructed
    
    # Compute contributions
    # T² contribution
    T2_contrib = np.zeros(len(short_names))
    for j in range(k_contrib):
        T2_contrib += (T_k[item_idx, j] * P_k[:, j]) ** 2 / eigenvalues[j]
    
    # Q contribution (squared residuals)
    Q_contrib = E[item_idx] ** 2
    
    # Total T² and Q for this item
    T2_total = np.sum((T_k[item_idx] ** 2) / eigenvalues[:k_contrib])
    Q_total = np.sum(E[item_idx] ** 2)
    
    # Create contribution plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'T² Contribution (Total T²: {T2_total:.2f})',
                        f'Q Contribution (Total Q: {Q_total:.2f})']
    )
    
    fig.add_trace(
        go.Bar(x=short_names, y=T2_contrib, marker_color='orange', 
               hovertemplate='%{x}<br>Contribution: %{y:.3f}<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=short_names, y=Q_contrib, marker_color='purple',
               hovertemplate='%{x}<br>Contribution: %{y:.3f}<extra></extra>'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=450, 
        showlegend=False,
        title_text=f'Contribution Analysis for: {selected_item}'
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show actual values
    st.markdown(f"### Nutritional Values for {selected_item}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Standardized Values:**")
        for i, var in enumerate(short_names):
            z_score = X[item_idx, i]
            st.markdown(f"- **{var}:** {z_score:+.2f} σ")
    
    with col2:
        st.markdown("**Raw Values:**")
        for i, var in enumerate(nutritional_cols):
            raw_val = X_raw[item_idx, i]
            st.markdown(f"- **{short_names[i]}:** {raw_val:.1f}")
    
    st.info("""
    **Interpreting Contributions:**
    
    **T² Contribution (Left):**
    - Shows which variables make this item extreme within the model
    - Large bars = variables with unusually high/low values compared to average
    - Example: High Energy bar means item is much higher/lower in calories than typical
    
    **Q Contribution (Right):**
    - Shows which variables don't fit the expected pattern
    - Large bars = variables that behave unexpectedly given other variables
    - Example: High Protein bar might mean protein is higher than expected given the fat content
    
    **Diagnosis Priority:**
    1. **Large Q contribution:** Investigate first — indicates unusual behavior
    2. **Large T² contribution:** Secondary — just confirms item is extreme
    """)

# ===== TAB 8: Interpretation =====
with tabs[8]:
    st.markdown("## Correlation Loading Plot")
    st.markdown("""
    Shows correlation between each variable and the principal components.
    - **Variables near outer circle:** Well-modeled by the selected PCs
    - **Arrow direction:** Shows which quadrant items high in that variable will fall
    - **Angle between arrows:** Small = positively correlated, 180° = negatively correlated
    """)
    
    if 'pca' not in locals():
        pca = PCA()
        pca.fit(X)
        P = pca.components_.T
        explained_var = pca.explained_variance_ratio_
        eigenvalues = pca.explained_variance_
    
    # Compute correlation loadings
    corr_loadings = P * np.sqrt(eigenvalues)
    
    # PC selection
    col1, col2 = st.columns(2)
    with col1:
        corr_pc_x = st.selectbox("X-axis", [f"PC{i+1}" for i in range(min(6, len(explained_var)))], 
                                 index=0, key='corr_x')
    with col2:
        corr_pc_y = st.selectbox("Y-axis", [f"PC{i+1}" for i in range(min(6, len(explained_var)))], 
                                 index=1, key='corr_y')
    
    corr_x_idx = int(corr_pc_x[2:]) - 1
    corr_y_idx = int(corr_pc_y[2:]) - 1
    
    fig = go.Figure()
    
    # Unit circles
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color='gray', dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=0.5*np.cos(theta), y=0.5*np.sin(theta),
        mode='lines', line=dict(color='lightgray', dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Variable arrows
    for i, name in enumerate(short_names):
        cx = corr_loadings[i, corr_x_idx]
        cy = corr_loadings[i, corr_y_idx]
        
        fig.add_annotation(
            x=cx, y=cy, ax=0, ay=0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=2, arrowcolor=COLORS['mcd_red']
        )
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode='markers+text',
            marker=dict(size=10, color=COLORS['mcd_red']),
            text=[name],
            textposition='top center',
            showlegend=False,
            hovertemplate=f'{name}<br>{corr_pc_x}: {cx:.3f}<br>{corr_pc_y}: {cy:.3f}<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_color='black', line_width=0.5)
    fig.add_vline(x=0, line_color='black', line_width=0.5)
    
    fig.update_layout(
        title=f'Correlation Loading Plot: {corr_pc_x} vs {corr_pc_y}',
        xaxis=dict(title=f'Correlation with {corr_pc_x}', 
                   range=[-1.1, 1.1], constrain='domain', zeroline=True),
        yaxis=dict(title=f'Correlation with {corr_pc_y}', 
                   range=[-1.1, 1.1], scaleanchor='x', scaleratio=1, zeroline=True),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if corr_x_idx == 0 and corr_y_idx == 1:
        st.info("""
        **PC1 vs PC2 Correlation Pattern:**
        - **Right quadrant:** Energy, Fats, Protein, Sodium → high-calorie savory items
        - **Top quadrant:** Sugars, AddSugar, Carbs → sweet items
        - **Cholesterol is separate:** Distinct from other fats → egg items
        - **TransFat points down:** Negative PC2 → fried savory items
        """)

# ===== TAB 8: Interpretation =====
with tabs[8]:
    st.markdown("## 💡 What Have We Learned?")
    
    st.markdown("### The Power of PCA")
    st.markdown("""
    Starting with **10 nutritional variables**, PCA reduced the data to just **2-3 meaningful dimensions** 
    that capture **~75-85% of the variance**, while revealing clear patterns:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### PC1: Overall Nutritional Density
        - **High PC1:** Burgers, fried chicken, large meals
        - **Low PC1:** Salads, plain coffee, diet beverages
        - **Meaning:** How "filling" or calorie-dense is the item?
        
        #### PC2: Sweet vs Savory
        - **High PC2:** Desserts, milkshakes, sugary drinks
        - **Low PC2:** Grilled items, eggs, plain burgers
        - **Meaning:** Is it a treat or a meal?
        """)
    
    with col2:
        st.markdown("""
        #### Menu Insights
        1. **Beverages span the full range:** From diet drinks (low PC1, low PC2) 
           to milkshakes (high PC1, high PC2)
        
        2. **Desserts cluster together:** High sugar, moderate calories
        
        3. **Breakfast items are unique:** High cholesterol (eggs) but moderate overall
        
        4. **Trans fats are independent:** PC3 specifically captures this health concern
        """)
    
    st.markdown("### Comprehensive PCA Workflow")
    st.markdown("""
    This application covered the complete PCA analysis pipeline:
    
    1. **📊 Data Overview** → Understand the dataset and variable relationships
    2. **🔬 Standardization** → Ensure variables are on comparable scales
    3. **📈 PCA Results** → Determine optimal number of components
    4. **🎯 Score & Correlation Loadings** → See both observations and variable relationships side-by-side
    5. **🔍 Biplot** → Combine scores and loadings for integrated view
    6. **📍 Loading Analysis** → Deep dive into variable contributions
    7. **🚨 Outlier Detection** → Identify unusual items using T² and Q statistics
    8. **🔧 Contribution Plots** → Diagnose why specific items are outliers
    """)
    
    st.markdown("### Practical Applications")
    st.markdown("""
    This PCA analysis enables:
    
    - **Menu optimization:** Identify redundant items or gaps in the menu space
    - **Health scoring:** Create composite health indices from PC scores
    - **Recommendation systems:** "If you like X (high PC1, low PC2), try Y"
    - **Nutritional labeling:** Simplify communication with 2-3 scores instead of 10 numbers
    - **Quality control:** Detect outliers or items that don't fit expected patterns
    - **New product development:** See where proposed items fall in nutritional space
    """)
    
    st.markdown("### The Bigger Picture")
    st.success("""
    **PCA transforms complexity into simplicity** without throwing away information.
    
    Instead of tracking 10 correlated nutritional values:
    - PC1 tells you: "Is it light or hearty?"
    - PC2 tells you: "Is it sweet or savory?"
    - PC3-PC4 capture specific concerns (trans fats, cholesterol)
    
    **This is the essence of multivariate data analysis:** Find the hidden structure 
    that explains why things vary, and express it in a handful of interpretable dimensions.
    """)
    
    st.markdown("---")
    st.markdown("### Compare with PCR Application")
    st.markdown("""
    **Key Difference:** 
    - **PCA Application (this page):** Unsupervised — finds patterns in X without considering any response y
    - **PCR Application:** Supervised — uses PCA to predict responses (WindPower, PVPower)
    
    **Common Thread:** Both show that **variance structure matters**. 
    - Here, we care about variance to understand menu diversity
    - In PCR, we learned that high X-variance doesn't guarantee predictive power
    
    **Tools Covered:**
    - Score plots, Loading plots, Biplots → Understanding structure
    - Correlation loading plots → Variable relationships
    - Outlier detection (T² vs Q) → Quality control
    - Contribution plots → Root cause analysis
    """)

st.markdown("---")
st.markdown("*Based on PCA_McDonald_DA.ipynb — McDonald's India Menu Dataset*")