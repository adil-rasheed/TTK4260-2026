"""
Visualization Utilities
=======================
Functions for creating interactive and static visualizations.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


def plot_3d_cost_surface(X, y, theta_range=(-5, 5), resolution=50):
    """
    Create interactive 3D cost surface for linear regression.
    
    Parameters:
    -----------
    X : ndarray
        Predictor values
    y : ndarray
        Response values
    theta_range : tuple
        (min, max) for theta grid
    resolution : int
        Grid resolution
    
    Returns:
    --------
    fig : plotly Figure
    """
    theta0 = np.linspace(theta_range[0], theta_range[1], resolution)
    theta1 = np.linspace(theta_range[0], theta_range[1], resolution)
    Theta0, Theta1 = np.meshgrid(theta0, theta1)
    
    # Calculate cost for each theta combination
    Cost = np.zeros_like(Theta0)
    for i in range(resolution):
        for j in range(resolution):
            y_pred = Theta0[i,j] + Theta1[i,j] * X
            Cost[i,j] = np.mean((y - y_pred) ** 2)
    
    # Create 3D surface
    fig = go.Figure(data=[
        go.Surface(x=Theta0, y=Theta1, z=Cost, colorscale='Viridis', name='Cost Surface')
    ])
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(Cost), Cost.shape)
    opt_theta0 = Theta0[min_idx]
    opt_theta1 = Theta1[min_idx]
    opt_cost = Cost[min_idx]
    
    # Add optimal point
    fig.add_trace(go.Scatter3d(
        x=[opt_theta0], y=[opt_theta1], z=[opt_cost],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Optimal θ'
    ))
    
    fig.update_layout(
        title='Cost Function Surface',
        scene=dict(
            xaxis_title='θ₀ (intercept)',
            yaxis_title='θ₁ (slope)',
            zaxis_title='Cost (MSE)'
        ),
        height=600
    )
    
    return fig


def plot_contour_overlay(X, y, theta0_range, theta1_range, resolution=100):
    """
    Create contour plot of cost function with data overlay.
    
    Parameters:
    -----------
    X, y : ndarray
        Data
    theta0_range, theta1_range : tuple
        Parameter ranges
    resolution : int
        Grid resolution
    
    Returns:
    --------
    fig : plotly Figure
    """
    theta0 = np.linspace(theta0_range[0], theta0_range[1], resolution)
    theta1 = np.linspace(theta1_range[0], theta1_range[1], resolution)
    Theta0, Theta1 = np.meshgrid(theta0, theta1)
    
    Cost = np.zeros_like(Theta0)
    for i in range(resolution):
        for j in range(resolution):
            y_pred = Theta0[i,j] + Theta1[i,j] * X
            Cost[i,j] = np.mean((y - y_pred) ** 2)
    
    fig = go.Figure()
    
    # Contour plot
    fig.add_trace(go.Contour(
        x=theta0, y=theta1, z=Cost,
        colorscale='Viridis',
        contours=dict(showlabels=True),
        name='Cost'
    ))
    
    # Optimal point
    min_idx = np.unravel_index(np.argmin(Cost), Cost.shape)
    opt_theta0 = Theta0[min_idx]
    opt_theta1 = Theta1[min_idx]
    
    fig.add_trace(go.Scatter(
        x=[opt_theta0], y=[opt_theta1],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Optimal θ'
    ))
    
    fig.update_layout(
        title='Cost Function Contours',
        xaxis_title='θ₀',
        yaxis_title='θ₁',
        height=500
    )
    
    return fig


def plot_ols_vs_pcr_contours(theta_ols, theta_pcr, beta_scores, theta_ols_perturbed=None, beta_scores_perturbed=None):
    """
    Side-by-side contour comparison of OLS vs PCR optimization landscapes.
    Shows how PCR transforms the unstable elongated valleys of OLS into stable circular contours.
    
    Parameters:
    -----------
    theta_ols : ndarray
        OLS coefficients (2,)
    theta_pcr : ndarray
        PCR coefficients (2,)
    beta_scores : ndarray
        PCR score coefficients (2,)
    theta_ols_perturbed : ndarray, optional
        Perturbed OLS solutions (n_perturbations, 2)
    beta_scores_perturbed : ndarray, optional
        Perturbed PCR solutions (n_perturbations, 2)
    
    Returns:
    --------
    fig : plotly Figure
    """
    # Calculate statistics for annotations
    if theta_ols_perturbed is not None:
        ols_std = theta_ols_perturbed.std(axis=0)
        ols_spread = f"std(θ₁)={ols_std[0]:.3f}, std(θ₂)={ols_std[1]:.3f}"
    else:
        ols_spread = "Single solution"
    
    if beta_scores_perturbed is not None:
        pcr_std = beta_scores_perturbed.std(axis=0)
        pcr_spread = f"std(β₁)={pcr_std[0]:.3f}, std(β₂)={pcr_std[1]:.3f}"
        
        # Stability ratio
        if theta_ols_perturbed is not None and pcr_std[0] > 0:
            stability_ratio = np.mean(ols_std) / np.mean(pcr_std)
        else:
            stability_ratio = None
    else:
        pcr_spread = "Single solution"
        stability_ratio = None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'<b>OLS: Original Space</b><br><span style="color:red;font-size:11px;">UNSTABLE - Elongated valleys</span><br>{ols_spread}',
            f'<b>PCR: Score Space</b><br><span style="color:green;font-size:11px;">STABLE - Circular contours</span><br>{pcr_spread}'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
        horizontal_spacing=0.12
    )
    
    # Grid for contours
    theta_range = np.linspace(-3, 3, 100)
    Theta1, Theta2 = np.meshgrid(theta_range, theta_range)
    
    # OLS loss (elongated - high condition number)
    loss_ols = (Theta1 - 1.0)**2 + (Theta2 - 1.0)**2 + 5 * (Theta1 - 1.0) * (Theta2 - 1.0)
    
    # PCR loss (circular - low condition number)
    loss_pcr = (Theta1 - 1.0)**2 + (Theta2 - 1.0)**2
    
    # LEFT: Add OLS contours
    fig.add_trace(go.Contour(
        x=theta_range, y=theta_range, z=loss_ols,
        colorscale='Reds',
        showscale=False,
        contours=dict(showlabels=False, start=0, end=20, size=2),
        opacity=0.7,
        name='Cost Surface'
    ), row=1, col=1)
    
    # Add OLS optimal point
    fig.add_trace(go.Scatter(
        x=[theta_ols[0]], y=[theta_ols[1]],
        mode='markers+text',
        marker=dict(size=14, color='darkred', symbol='star', line=dict(color='black', width=2)),
        text=['θ̂_OLS'],
        textposition='top center',
        textfont=dict(size=12, color='darkred'),
        name='OLS Solution',
        showlegend=True
    ), row=1, col=1)
    
    # Add perturbed OLS points if provided
    if theta_ols_perturbed is not None:
        fig.add_trace(go.Scatter(
            x=theta_ols_perturbed[:, 0],
            y=theta_ols_perturbed[:, 1],
            mode='markers',
            marker=dict(size=6, color='orange', opacity=0.6, symbol='circle'),
            name='Perturbed Solutions',
            showlegend=True
        ), row=1, col=1)
        
        # Add annotation showing instability
        fig.add_annotation(
            text='<b>WIDE SPREAD</b><br>High variance',
            xref="x", yref="y",
            x=theta_ols[0], y=theta_ols[1] - 0.8,
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            font=dict(size=11, color='red'),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            row=1, col=1
        )
    
    # RIGHT: Add PCR contours
    fig.add_trace(go.Contour(
        x=theta_range, y=theta_range, z=loss_pcr,
        colorscale='Blues',
        showscale=False,
        contours=dict(showlabels=False, start=0, end=20, size=2),
        opacity=0.7,
        name='Cost Surface'
    ), row=1, col=2)
    
    # Add PCR optimal point
    fig.add_trace(go.Scatter(
        x=[beta_scores[0]], y=[beta_scores[1]],
        mode='markers+text',
        marker=dict(size=14, color='darkblue', symbol='star', line=dict(color='black', width=2)),
        text=['β̂_PCR'],
        textposition='top center',
        textfont=dict(size=12, color='darkblue'),
        name='PCR Solution',
        showlegend=True
    ), row=1, col=2)
    
    # Add perturbed PCR points if provided
    if beta_scores_perturbed is not None:
        fig.add_trace(go.Scatter(
            x=beta_scores_perturbed[:, 0],
            y=beta_scores_perturbed[:, 1],
            mode='markers',
            marker=dict(size=6, color='cyan', opacity=0.6, symbol='circle'),
            name='Perturbed Solutions',
            showlegend=False  # Already shown in left plot
        ), row=1, col=2)
        
        # Add annotation showing stability
        fig.add_annotation(
            text='<b>TIGHT CLUSTER</b><br>Low variance',
            xref="x2", yref="y2",
            x=beta_scores[0], y=beta_scores[1] - 0.8,
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            font=dict(size=11, color='green'),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="green",
            borderwidth=1
        )
    
    # Update axes
    fig.update_xaxes(title_text='θ₁', row=1, col=1)
    fig.update_yaxes(title_text='θ₂', row=1, col=1)
    fig.update_xaxes(title_text='β₁ (PC1 coefficient)', row=1, col=2)
    fig.update_yaxes(title_text='β₂ (PC2 coefficient)', row=1, col=2)
    
    # Add overall stability message
    if stability_ratio is not None:
        fig.add_annotation(
            text=f'<b>⚡ Stability Improvement</b><br>PCR is <b>{stability_ratio:.1f}×</b> more stable than OLS',
            xref="paper", yref="paper",
            x=0.5, y=1.12,
            showarrow=False,
            font=dict(size=14, color='green'),
            bgcolor="lightgreen",
            bordercolor="darkgreen",
            borderwidth=2
        )
    
    fig.update_layout(
        height=550,
        showlegend=True,
        legend=dict(x=1.05, y=1),
        title_text="<b>Loss Surface Comparison: Why PCR is More Stable</b><br><sub>Perturbations create wide spread in OLS space but tight clusters in PCR score space</sub>",
        title_x=0.5
    )
    
    return fig


def plot_bootstrap_comparison(bootstrap_ols, bootstrap_pcr, beta_true):
    """
    Create 2x2 bootstrap comparison plot matching notebook layout.
    
    Layout (matching notebook):
    Top row: OLS (UNSTABLE)
    Bottom row: PCR (STABLE)
    Left column: Coefficient histograms
    Right column: Joint scatter plots
    
    Parameters:
    -----------
    bootstrap_ols : ndarray
        OLS bootstrap coefficients (n_bootstrap, 2)
    bootstrap_pcr : ndarray
        PCR bootstrap coefficients (n_bootstrap, 2)
    beta_true : ndarray
        True coefficients (2,)
    
    Returns:
    --------
    fig : plotly Figure
    """
    # Calculate statistics
    ols_std = bootstrap_ols.std(axis=0)
    pcr_std = bootstrap_pcr.std(axis=0)
    ols_mean = bootstrap_ols.mean(axis=0)
    pcr_mean = bootstrap_pcr.mean(axis=0)
    
    # Stability ratio
    stability_ratio_1 = ols_std[0] / pcr_std[0] if pcr_std[0] > 0 else 0
    stability_ratio_2 = ols_std[1] / pcr_std[1] if pcr_std[1] > 0 else 0
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'<b>OLS</b>: Coefficient Distributions<br><span style="color:red;font-size:12px;">UNSTABLE - WIDE SPREAD</span><br>std(β₁)={ols_std[0]:.3f}, std(β₂)={ols_std[1]:.3f}',
            f'<b>OLS</b>: Joint Distribution<br><span style="color:red;font-size:12px;">WIDE SPREAD</span>',
            f'<b>PCR</b>: Coefficient Distributions<br><span style="color:green;font-size:12px;">STABLE - TIGHT CLUSTER</span><br>std(β₁)={pcr_std[0]:.3f}, std(β₂)={pcr_std[1]:.3f}',
            f'<b>PCR</b>: Joint Distribution<br><span style="color:green;font-size:12px;">TIGHT CLUSTER</span>'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # TOP LEFT: OLS histograms
    fig.add_trace(go.Histogram(
        x=bootstrap_ols[:, 0], name='β₁ (OLS)',
        marker_color='red', opacity=0.6, nbinsx=25,
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=bootstrap_ols[:, 1], name='β₂ (OLS)',
        marker_color='pink', opacity=0.6, nbinsx=25,
        showlegend=True
    ), row=1, col=1)
    
    # TOP RIGHT: OLS scatter
    fig.add_trace(go.Scatter(
        x=bootstrap_ols[:, 0], y=bootstrap_ols[:, 1],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.4),
        name='OLS samples',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[beta_true[0]], y=[beta_true[1]],
        mode='markers',
        marker=dict(size=15, color='darkred', symbol='star', line=dict(color='black', width=2)),
        name='True β',
        showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[ols_mean[0]], y=[ols_mean[1]],
        mode='markers',
        marker=dict(size=12, color='orange', symbol='x', line=dict(color='black', width=2)),
        name='Mean (OLS)',
        showlegend=True
    ), row=1, col=2)
    
    # BOTTOM LEFT: PCR histograms
    fig.add_trace(go.Histogram(
        x=bootstrap_pcr[:, 0], name='β₁ (PCR)',
        marker_color='blue', opacity=0.6, nbinsx=25,
        showlegend=True
    ), row=2, col=1)
    
    fig.add_trace(go.Histogram(
        x=bootstrap_pcr[:, 1], name='β₂ (PCR)',
        marker_color='lightblue', opacity=0.6, nbinsx=25,
        showlegend=True
    ), row=2, col=1)
    
    # BOTTOM RIGHT: PCR scatter
    fig.add_trace(go.Scatter(
        x=bootstrap_pcr[:, 0], y=bootstrap_pcr[:, 1],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.4),
        name='PCR samples',
        showlegend=True
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=[beta_true[0]], y=[beta_true[1]],
        mode='markers',
        marker=dict(size=15, color='darkblue', symbol='star', line=dict(color='black', width=2)),
        name='True β',
        showlegend=False  # Already shown in top plot
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=[pcr_mean[0]], y=[pcr_mean[1]],
        mode='markers',
        marker=dict(size=12, color='cyan', symbol='x', line=dict(color='black', width=2)),
        name='Mean (PCR)',
        showlegend=True
    ), row=2, col=2)
    
    # Update axes
    fig.update_xaxes(title_text='Coefficient Value', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_xaxes(title_text='β₁', row=1, col=2)
    fig.update_yaxes(title_text='β₂', row=1, col=2)
    fig.update_xaxes(title_text='Coefficient Value', row=2, col=1)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    fig.update_xaxes(title_text='β₁', row=2, col=2)
    fig.update_yaxes(title_text='β₂', row=2, col=2)
    
    # Add stability improvement annotation
    avg_ratio = (stability_ratio_1 + stability_ratio_2) / 2
    fig.add_annotation(
        text=f'<b>📊 Stability Improvement</b><br>PCR is <b>{avg_ratio:.1f}×</b> more stable than OLS<br>(β₁: {stability_ratio_1:.1f}×, β₂: {stability_ratio_2:.1f}×)',
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=14, color='green'),
        bgcolor="lightgray",
        bordercolor="green",
        borderwidth=2
    )
    
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(x=1.05, y=1),
        title_text="<b>Bootstrap Stability Analysis: OLS vs PCR</b><br><sub>100 bootstrap iterations showing coefficient variability</sub>",
        title_x=0.5
    )
    
    return fig


def plot_pca_2d(data, pc_axes=None, labels=None):
    """
    Plot 2D data with optional PC axes overlay.
    
    Parameters:
    -----------
    data : ndarray, shape (n, 2)
        2D data points
    pc_axes : ndarray, shape (2, 2) (optional)
        Principal component axes
    labels : ndarray (optional)
        Point labels for coloring
    
    Returns:
    --------
    fig : plotly Figure
    """
    fig = go.Figure()
    
    # Data points
    if labels is not None:
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1],
            mode='markers',
            marker=dict(size=8, color=labels, colorscale='Viridis'),
            name='Data'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Data'
        ))
    
    # PC axes
    if pc_axes is not None:
        center = data.mean(axis=0)
        
        for i, pc in enumerate(pc_axes.T):
            scale = 3 * np.std(data @ pc)
            fig.add_trace(go.Scatter(
                x=[center[0] - scale*pc[0], center[0] + scale*pc[0]],
                y=[center[1] - scale*pc[1], center[1] + scale*pc[1]],
                mode='lines',
                line=dict(width=3, color=['red', 'green'][i]),
                name=f'PC{i+1}'
            ))
    
    fig.update_layout(
        title='PCA Visualization',
        xaxis_title='X₁',
        yaxis_title='X₂',
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig


def plot_correlation_heatmap(X, feature_names=None):
    """
    Plot correlation matrix as heatmap.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    feature_names : list (optional)
        Feature names
    
    Returns:
    --------
    fig : plotly Figure
    """
    corr_matrix = np.corrcoef(X.T)
    
    if feature_names is None:
        feature_names = [f'X{i+1}' for i in range(X.shape[1])]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=feature_names,
        y=feature_names,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        height=500
    )
    
    return fig


def plot_loading_heatmap(pca, feature_names=None, n_components=None):
    """
    Plot loading matrix as heatmap showing feature contributions to all PCs.
    
    Parameters:
    -----------
    pca : sklearn PCA object
        Fitted PCA model
    feature_names : list (optional)
        Feature names
    n_components : int (optional)
        Number of components to display (default: all available)
    
    Returns:
    --------
    fig : plotly Figure
    """
    loadings = pca.components_.T  # Transpose to get features x components
    
    if n_components is None:
        n_components = min(10, pca.n_components_)
    else:
        n_components = min(n_components, pca.n_components_)
    
    loadings_display = loadings[:, :n_components]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(loadings.shape[0])]
    
    component_names = [f'PC{i+1}' for i in range(n_components)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=loadings_display.T,  # Transpose so PCs are rows
        x=feature_names,
        y=component_names,
        colorscale='RdBu',
        zmid=0,
        text=np.round(loadings_display.T, 3),
        texttemplate='%{text}',
        textfont={"size": 9},
        colorbar=dict(title="Loading<br>Value")
    ))
    
    fig.update_layout(
        title='PCA Loading Heatmap - Feature Contributions to Components',
        xaxis_title='Features',
        yaxis_title='Principal Components',
        height=max(300, n_components * 60),
        xaxis_tickangle=-45
    )
    
    return fig


def plot_correlation_loading_plot(pca, feature_names=None, pc1=0, pc2=1):
    """
    Plot correlation loading plot with correlation circles.
    Shows variables in principal component space with r=1 and r=0.7 circles.
    
    Parameters:
    -----------
    pca : sklearn PCA object
        Fitted PCA model
    feature_names : list (optional)
        Feature names
    pc1, pc2 : int
        Which principal components to plot (default: PC1 vs PC2)
    
    Returns:
    --------
    fig : plotly Figure
    """
    # Get loadings scaled by sqrt of eigenvalues (for correlation interpretation)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    if feature_names is None:
        feature_names = [f'F{i+1}' for i in range(loadings.shape[0])]
    
    fig = go.Figure()
    
    # Add variable arrows
    for i, feature in enumerate(feature_names):
        x_val = loadings[i, pc1]
        y_val = loadings[i, pc2]
        
        # Arrow line
        fig.add_trace(go.Scatter(
            x=[0, x_val],
            y=[0, y_val],
            mode='lines+text',
            name=feature,
            text=['', feature],
            textposition='top center',
            line=dict(width=2),
            marker=dict(size=8),
            showlegend=False
        ))
    
    # Add correlation circles
    theta = np.linspace(0, 2*np.pi, 100)
    
    # r=1 circle
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        line=dict(color='red', dash='solid', width=2),
        name='r = 1.0',
        showlegend=True
    ))
    
    # r=0.7 circle
    fig.add_trace(go.Scatter(
        x=0.7*np.cos(theta),
        y=0.7*np.sin(theta),
        mode='lines',
        line=dict(color='orange', dash='dash', width=1.5),
        name='r = 0.7',
        showlegend=True
    ))
    
    var_pc1 = pca.explained_variance_ratio_[pc1] * 100
    var_pc2 = pca.explained_variance_ratio_[pc2] * 100
    
    fig.update_layout(
        title='Correlation Loading Plot',
        xaxis_title=f'PC{pc1+1} ({var_pc1:.1f}%)',
        yaxis_title=f'PC{pc2+1} ({var_pc2:.1f}%)',
        height=600,
        width=600,
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            range=[-1.1, 1.1],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            range=[-1.1, 1.1],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgray'
        )
    )
    
    return fig


def plot_communalities(pca, feature_names=None, n_components=None):
    """
    Plot communalities - variance of each variable explained by selected PCs.
    Communalities = sum of squared loadings for each variable.
    
    Parameters:
    -----------
    pca : sklearn PCA object
        Fitted PCA model
    feature_names : list (optional)
        Feature names
    n_components : int (optional)
        Number of components to include (default: all)
    
    Returns:
    --------
    fig : plotly Figure
    """
    loadings = pca.components_.T
    
    if n_components is None:
        n_components = pca.n_components_
    else:
        n_components = min(n_components, pca.n_components_)
    
    # Calculate communalities (sum of squared loadings)
    communalities = np.sum(loadings[:, :n_components]**2, axis=1)
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(len(communalities))]
    
    # Color code: green if > 0.7, orange if 0.5-0.7, red if < 0.5
    colors = ['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' 
              for c in communalities]
    
    fig = go.Figure(data=go.Bar(
        x=feature_names,
        y=communalities,
        marker_color=colors,
        text=np.round(communalities, 3),
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                  annotation_text="Good (0.7)", annotation_position="right")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="Acceptable (0.5)", annotation_position="right")
    
    fig.update_layout(
        title=f'Communalities (r={n_components} components)',
        xaxis_title='Features',
        yaxis_title='Communality',
        yaxis_range=[0, min(1.1, max(communalities) * 1.1)],
        height=500,
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig


def nipals_algorithm(X, max_iter=50, tol=1e-6):
    """
    NIPALS algorithm with step-by-step history for visualization.
    
    Parameters:
    -----------
    X : ndarray
        Data matrix (will be centered)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    history : list of dicts
        Each dict contains: iteration, p (loading), t (score), change, angle
    final_t : ndarray
        Final score vector
    final_p : ndarray
        Final loading vector
    """
    # Center data
    X_centered = X - np.mean(X, axis=0)
    n, k = X_centered.shape
    
    # Initialize t with first column
    t = X_centered[:, 0].copy().reshape(-1, 1)
    
    history = []
    
    for iteration in range(max_iter):
        # Step 2: Regress X onto t to get p (loadings)
        p = (X_centered.T @ t) / (t.T @ t)
        
        # Step 3: Normalize p
        p = p / np.linalg.norm(p)
        
        # Step 4: Regress X onto p to get new t (scores)
        t_new = (X_centered @ p) / (p.T @ p)
        
        # Calculate change
        change = np.linalg.norm(t_new - t)
        
        # Calculate angle (for 2D visualization)
        angle = np.degrees(np.arctan2(p[1, 0], p[0, 0])) if k >= 2 else 0
        
        # Store history
        history.append({
            'iteration': iteration + 1,
            'p': p.flatten(),
            't': t_new.flatten(),
            'change': change,
            'angle': angle
        })
        
        # Check convergence
        if change < tol:
            break
        
        t = t_new
    
    return history, t, p


def plot_nipals_convergence(history):
    """
    Plot NIPALS convergence: change over iterations and angle evolution.
    
    Parameters:
    -----------
    history : list of dicts
        History from nipals_algorithm function
    
    Returns:
    --------
    fig : plotly Figure
    """
    from plotly.subplots import make_subplots
    
    iterations = [h['iteration'] for h in history]
    changes = [h['change'] for h in history]
    angles = [h['angle'] for h in history]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Convergence (Change in Score Vector)', 
                       'Loading Angle vs Iteration')
    )
    
    # Left plot: Convergence (log scale)
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=changes,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2),
            name='Change'
        ),
        row=1, col=1
    )
    
    # Right plot: Angle evolution
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=angles,
            mode='lines+markers',
            marker=dict(size=8, color='red'),
            line=dict(width=2, color='red'),
            name='PC1 Angle'
        ),
        row=1, col=2
    )
    
    fig.update_yaxes(title='Change (log scale)', type='log', row=1, col=1)
    fig.update_yaxes(title='PC1 Angle (degrees)', row=1, col=2)
    fig.update_xaxes(title='Iteration', row=1, col=1)
    fig.update_xaxes(title='Iteration', row=1, col=2)
    
    fig.update_layout(
        title='NIPALS Convergence Analysis',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_nipals_step(X, history, iteration_idx):
    """
    Plot NIPALS at a specific iteration showing data and loading vector.
    
    Parameters:
    -----------
    X : ndarray
        Original data (will be centered)
    history : list of dicts
        History from nipals_algorithm
    iteration_idx : int
        Which iteration to display (0-indexed)
    
    Returns:
    --------
    fig : plotly Figure
    """
    X_centered = X - np.mean(X, axis=0)
    
    idx = min(iteration_idx, len(history) - 1)
    state = history[idx]
    p = state['p']
    
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=X_centered[:, 0],
        y=X_centered[:, 1],
        mode='markers',
        marker=dict(size=6, color='blue', opacity=0.5),
        name='Data'
    ))
    
    # Current loading vector (PC1 direction)
    L = np.max(np.abs(X_centered)) * 0.9
    fig.add_trace(go.Scatter(
        x=[-p[0]*L, p[0]*L],
        y=[-p[1]*L, p[1]*L],
        mode='lines',
        line=dict(color='red', width=4),
        name=f'PC1 (iter {state["iteration"]})'
    ))
    
    # Show projections onto current PC
    if len(p) >= 2:
        proj_coords = X_centered @ p
        proj_points = np.outer(proj_coords, p)
        
        # Sample projection lines (every 10th point to avoid clutter)
        for i in range(0, len(X_centered), 10):
            fig.add_trace(go.Scatter(
                x=[X_centered[i, 0], proj_points[i, 0]],
                y=[X_centered[i, 1], proj_points[i, 1]],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ))
        
        # Projected points
        fig.add_trace(go.Scatter(
            x=proj_points[:, 0],
            y=proj_points[:, 1],
            mode='markers',
            marker=dict(size=4, color='red', opacity=0.6),
            name='Projections'
        ))
    
    L = np.max(np.abs(X_centered)) * 1.1
    fig.update_layout(
        title=f"NIPALS Iteration {state['iteration']} | Change: {state['change']:.2e} | Angle: {state['angle']:.1f}°",
        xaxis=dict(range=[-L, L], title='X₁', scaleanchor='y'),
        yaxis=dict(range=[-L, L], title='X₂'),
        height=600,
        width=700
    )
    
    return fig


def plot_l1_l2_geometry(beta_ols, beta_ridge, beta_lasso, lambda_val=1.0):
    """
    Plot L1 (diamond) vs L2 (circle) constraint regions with OLS and regularized solutions.
    Shows geometric interpretation of why L1 produces sparsity.
    
    Parameters:
    -----------
    beta_ols : array
        OLS coefficients (2D)
    beta_ridge : array
        Ridge coefficients
    beta_lasso : array
        Lasso coefficients
    lambda_val : float
        Regularization parameter (for constraint radius)
    
    Returns:
    --------
    fig : plotly Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('L2 Regularization (Ridge)', 'L1 Regularization (Lasso)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Create contour levels for OLS cost function (ellipses)
    theta = np.linspace(0, 2*np.pi, 100)
    
    # L2 plot (LEFT): Circle constraint
    # Circle constraint region
    r_l2 = np.linalg.norm(beta_ridge)
    fig.add_trace(go.Scatter(
        x=r_l2 * np.cos(theta),
        y=r_l2 * np.sin(theta),
        mode='lines',
        line=dict(color='blue', width=3),
        name='L2 Constraint (||β||₂ = c)',
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ), row=1, col=1)
    
    # OLS contours (ellipses centered at OLS solution)
    for scale in [0.5, 1.0, 1.5]:
        a, b = scale, scale * 0.6  # Ellipse axes
        ellipse_x = beta_ols[0] + a * np.cos(theta)
        ellipse_y = beta_ols[1] + b * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
    
    # OLS solution
    fig.add_trace(go.Scatter(
        x=[beta_ols[0]],
        y=[beta_ols[1]],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='circle'),
        text=['β̂_OLS'],
        textposition='top center',
        name='OLS Solution'
    ), row=1, col=1)
    
    # Ridge solution
    fig.add_trace(go.Scatter(
        x=[beta_ridge[0]],
        y=[beta_ridge[1]],
        mode='markers+text',
        marker=dict(size=12, color='blue', symbol='star'),
        text=['β̂_Ridge'],
        textposition='bottom center',
        name='Ridge Solution'
    ), row=1, col=1)
    
    # L1 plot (RIGHT): Diamond constraint
    # Diamond constraint region
    r_l1 = np.sum(np.abs(beta_lasso))
    diamond_x = np.array([r_l1, 0, -r_l1, 0, r_l1])
    diamond_y = np.array([0, r_l1, 0, -r_l1, 0])
    
    fig.add_trace(go.Scatter(
        x=diamond_x,
        y=diamond_y,
        mode='lines',
        line=dict(color='green', width=3),
        name='L1 Constraint (||β||₁ = c)',
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ), row=1, col=2)
    
    # OLS contours
    for scale in [0.5, 1.0, 1.5]:
        a, b = scale, scale * 0.6
        ellipse_x = beta_ols[0] + a * np.cos(theta)
        ellipse_y = beta_ols[1] + b * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=2)
    
    # OLS solution
    fig.add_trace(go.Scatter(
        x=[beta_ols[0]],
        y=[beta_ols[1]],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='circle'),
        text=['β̂_OLS'],
        textposition='top center',
        showlegend=False
    ), row=1, col=2)
    
    # Lasso solution (often sparse - on axis)
    fig.add_trace(go.Scatter(
        x=[beta_lasso[0]],
        y=[beta_lasso[1]],
        mode='markers+text',
        marker=dict(size=12, color='green', symbol='diamond'),
        text=['β̂_Lasso'],
        textposition='bottom center',
        name='Lasso Solution'
    ), row=1, col=2)
    
    # Add axes
    max_val = max(abs(beta_ols[0]), abs(beta_ols[1])) * 1.5
    for col in [1, 2]:
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=1, col=col)
        fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1, row=1, col=col)
        
        fig.update_xaxes(title='β₁', range=[-max_val, max_val], row=1, col=col)
        fig.update_yaxes(title='β₂', range=[-max_val, max_val], row=1, col=col, scaleanchor='x', scaleratio=1)
    
    fig.update_layout(
        title='Geometric Interpretation: L1 vs L2 Regularization',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_regularization_paths(alphas, coef_paths, feature_names=None):
    """
    Plot coefficient paths as function of regularization parameter.
    
    Parameters:
    -----------
    alphas : array
        Regularization parameter values
    coef_paths : array
        Coefficient values (n_alphas, n_features)
    feature_names : list (optional)
        Feature names
    
    Returns:
    --------
    fig : plotly Figure
    """
    n_features = coef_paths.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    fig = go.Figure()
    
    # Plot each coefficient path
    for i in range(n_features):
        fig.add_trace(go.Scatter(
            x=alphas,
            y=coef_paths[:, i],
            mode='lines',
            name=feature_names[i],
            line=dict(width=2),
            hovertemplate=f'{feature_names[i]}<br>α=%{{x:.3f}}<br>coef=%{{y:.3f}}<extra></extra>'
        ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='Regularization Path: Coefficient Values vs α',
        xaxis_title='Regularization Parameter (α / λ)',
        yaxis_title='Coefficient Value',
        xaxis_type='log',
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_gradient_descent_path(X, y, theta_history, contour_resolution=50):
    """
    Plot gradient descent path on 2D contour plot of cost function.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (for 2D: n_samples x 2)
    y : ndarray
        Target values
    theta_history : ndarray
        History of theta values during optimization (n_iterations x 2)
    contour_resolution : int
        Resolution of contour grid
    
    Returns:
    --------
    fig : plotly Figure
    """
    # Create contour grid
    theta_history = np.array(theta_history)
    margin = 1.0
    theta0_min = min(theta_history[:, 0].min() - margin, -margin)
    theta0_max = max(theta_history[:, 0].max() + margin, margin)
    theta1_min = min(theta_history[:, 1].min() - margin, -margin)
    theta1_max = max(theta_history[:, 1].max() + margin, margin)
    
    theta0_range = np.linspace(theta0_min, theta0_max, contour_resolution)
    theta1_range = np.linspace(theta1_min, theta1_max, contour_resolution)
    Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
    
    # Calculate cost
    Cost = np.zeros_like(Theta0)
    for i in range(contour_resolution):
        for j in range(contour_resolution):
            theta = np.array([Theta0[i, j], Theta1[i, j]])
            y_pred = X @ theta
            Cost[i, j] = np.mean((y - y_pred) ** 2)
    
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(
        x=theta0_range,
        y=theta1_range,
        z=Cost,
        colorscale='Viridis',
        contours=dict(showlabels=True),
        name='Cost'
    ))
    
    # Add gradient descent path
    fig.add_trace(go.Scatter(
        x=theta_history[:, 0],
        y=theta_history[:, 1],
        mode='lines+markers',
        marker=dict(size=6, color='red'),
        line=dict(color='red', width=2),
        name='GD Path'
    ))
    
    # Mark start and end
    fig.add_trace(go.Scatter(
        x=[theta_history[0, 0]],
        y=[theta_history[0, 1]],
        mode='markers',
        marker=dict(size=15, color='green', symbol='star'),
        name='Start'
    ))
    
    fig.add_trace(go.Scatter(
        x=[theta_history[-1, 0]],
        y=[theta_history[-1, 1]],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='diamond'),
        name='End'
    ))
    
    fig.update_layout(
        title='Gradient Descent Path on Cost Surface',
        xaxis_title='θ₀',
        yaxis_title='θ₁',
        height=600,
        showlegend=True
    )
    
    return fig


def plot_outlier_effect(X, y, outlier_idx=None):
    """
    Compare regression with and without outliers.
    
    Parameters:
    -----------
    X : ndarray
        Feature values
    y : ndarray
        Target values
    outlier_idx : int or list (optional)
        Indices of outlier points
    
    Returns:
    --------
    fig : plotly Figure
    """
    from sklearn.linear_model import LinearRegression
    
    # Fit with all data
    model_with = LinearRegression()
    model_with.fit(X.reshape(-1, 1), y)
    
    # Fit without outliers
    if outlier_idx is not None:
        mask = np.ones(len(y), dtype=bool)
        mask[outlier_idx] = False
        X_clean = X[mask]
        y_clean = y[mask]
        
        model_without = LinearRegression()
        model_without.fit(X_clean.reshape(-1, 1), y_clean)
    else:
        model_without = model_with
    
    # Create predictions
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred_with = model_with.predict(X_line)
    y_pred_without = model_without.predict(X_line)
    
    fig = go.Figure()
    
    # Data points
    if outlier_idx is not None:
        # Normal points
        mask = np.ones(len(y), dtype=bool)
        mask[outlier_idx] = False
        fig.add_trace(go.Scatter(
            x=X[mask],
            y=y[mask],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Normal Data'
        ))
        
        # Outliers
        fig.add_trace(go.Scatter(
            x=X[outlier_idx],
            y=y[outlier_idx],
            mode='markers',
            marker=dict(size=12, color='red', symbol='x'),
            name='Outliers'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=X,
            y=y,
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Data'
        ))
    
    # Regression lines
    fig.add_trace(go.Scatter(
        x=X_line.ravel(),
        y=y_pred_with,
        mode='lines',
        line=dict(color='red', width=3),
        name='With Outliers'
    ))
    
    if outlier_idx is not None:
        fig.add_trace(go.Scatter(
            x=X_line.ravel(),
            y=y_pred_without,
            mode='lines',
            line=dict(color='green', width=3, dash='dash'),
            name='Without Outliers'
        ))
    
    fig.update_layout(
        title='Effect of Outliers on Least Squares',
        xaxis_title='X',
        yaxis_title='y',
        height=500
    )
    
    return fig


# Quick test
if __name__ == "__main__":
    # Test visualization
    X = np.linspace(0, 10, 50)
    y = 2.5 * X + 1 + np.random.randn(50)
    
    fig = plot_3d_cost_surface(X, y)
    print("3D surface created successfully")
