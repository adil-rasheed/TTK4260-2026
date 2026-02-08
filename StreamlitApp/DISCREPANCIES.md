# Discrepancies Between Notebooks and Streamlit App

## Overview
This document lists visualizations present in the Jupyter notebooks that may be missing or differ in the Streamlit app.

## PCA Explorer (02_PCA_Explorer.py)

### Missing Visualizations from PCA_CityTemp_Plots.ipynb:

1. **Loading Heatmap** (Section 6.2)
   - Shows all loadings across multiple components in a heatmap format
   - Color-coded by loading strength
   - Currently: Basic loading plots exist, but not comprehensive heatmap

2. **Correlation Loading Plot** (Section 6.3)
   - Circular plot with correlation circles (r=1, r=0.7)
   - Shows variable correlations in PC space
   - Currently: May be missing or simplified

3. **Communalities Plot** (Section 6.4)
   - Bar chart showing variance explained per variable
   - Currently: Not implemented

4. **Score Line Plot** (Section 2.5)
   - Time-series style visualization of scores
   - Useful for trajectory analysis
   - Currently: Not implemented

5. **3D Score Plot** (Section 2.6)
   - Interactive 3D scatter of first 3 PCs
   - Currently: May be simplified or missing

### Missing from PCA_InteractiveUnderstanding.ipynb:

1. **2D PCA Geometric Visualization**
   - Interactive data scatter with projection lines
   - Projection error curves as function of angle
   - Currently: Not in app (notebook-specific)

2. **3D Projections onto Cartesian Planes**
   - Multiple 2D projections of 3D data
   - Currently: Not in app

3. **Euler Angle Projections**
   - Rotation demonstrations
   - Currently: Not in app (advanced geometric concepts)

### Missing from PCA-Algorithms.ipynb:

1. **NIPALS Convergence Visualization**
   - Step-by-step iterative convergence
   - Loading vector evolution animation
   - Currently: Not in app

2. **Covariance Matrix Heatmap**
   - Visual representation of covariance structure
   - Currently: Not in app

3. **Eigenvalue Spectrum Bar Chart**
   - Comparison of eigenvalues
   - Currently: May be replaced by scree plot

4. **Algorithm Comparison Plots**
   - NIPALS vs Eigen vs sklearn comparison
   - Score correlations between methods
   - Currently: Not in app

### Missing from ComprehensivePCA.ipynb:

1. **Biplot with Multiple Scaling Options**
   - Different scaling factors for loadings
   - Currently: Basic biplot exists but may not have all options

2. **Contribution Plots**
   - Individual observation contributions to statistics
   - Currently: Not fully implemented

## PCR Interactive (01_PCR_Interactive.py)

### Potential Issues from PCR_Interactive.ipynb:

1. **Bootstrap Visualization Layout**
   - Notebook uses 2x2 matplotlib grid:
     - Top left: OLS coefficient histograms
     - Top right: OLS coefficient scatter
     - Bottom left: PCR coefficient histograms
     - Bottom right: PCR coefficient scatter
   - App: May use different Plotly layout

2. **Contour Plots**
   - `plot_ols_vs_pcr_contours()` function in notebook
   - Side-by-side comparison of OLS/PCR instability
   - App: Verify if same visualization exists

3. **Text Annotations**
   - Notebook includes:
     - "UNSTABLE" vs "STABLE" labels
     - "WIDE SPREAD" vs "TIGHT CLUSTER"
     - Stability improvement ratio: "PCR is X× more stable"
     - Standard deviations displayed
   - App: Verify these annotations exist

4. **Condition Number Display**
   - Notebook prominently displays condition number
   - App: Verify if shown with same prominence

## Least Squares (04_Least_Squares.py)

### From LSInteractiveLecture.ipynb:

1. **Cost Surface 3D Plot**
   - Interactive 3D surface of cost function
   - Currently: Verify if in app

2. **Cost Contour Plot**
   - 2D contour lines with solution path
   - Currently: Verify if in app

3. **Outlier Effect Visualization**
   - Before/after comparison with outliers
   - Currently: Verify implementation

## MLR Regularization (03_MLR_Regularization.py)

### From MLR-regularization.ipynb:

1. **Fixed Minima Contour Plots**
   - L1 vs L2 regularization geometry
   - Diamond (L1) vs circle (L2) constraints
   - Currently: Verify if in app

2. **Axis Intersections Visualization**
   - Shows why L1 produces sparsity
   - Currently: Verify if in app

3. **Regularization Path Plots**
   - Coefficient values vs lambda
   - Currently: Verify if in app

## Performance Metrics (06_Performance_Metrics.py)

### From performance_metrics_ml.ipynb:

1. **Confusion Matrix Explained**
   - Interactive confusion matrix breakdown
   - Currently: Verify if in app

2. **ROC Curve**
   - With AUC calculation
   - Currently: Verify if in app

3. **Precision-Recall Curve**
   - Currently: Verify if in app

## MLE (05_Maximum_Likelihood.py)

### From MLE-Normal.ipynb:

1. **Interactive Normal Distribution**
   - Sliders for μ and σ
   - Currently: Verify if in app

2. **Likelihood Surface**
   - 3D visualization of likelihood function
   - Currently: Verify if in app

## Recommendations

### Priority 1 (Core Educational Value):
1. Add NIPALS convergence visualization to PCA page
2. Implement loading heatmap in PCA page
3. Verify PCR bootstrap layout matches notebook (2x2 grid)
4. Add correlation loading plot (circular) to PCA page

### Priority 2 (Enhanced Understanding):
1. Add communalities plot to PCA page
2. Implement 3D score plot in PCA page
3. Add score line plot to PCA page
4. Verify all text annotations in PCR page

### Priority 3 (Advanced Features):
1. Add covariance matrix heatmap to PCA page
2. Implement algorithm comparison in PCA page
3. Add contribution plots to PCA page
4. Geometric projection visualizations (consider if needed for app)

## Notes

- Some notebook visualizations are specifically educational and may not translate well to a Streamlit app
- Interactive widgets in notebooks (sliders, dropdowns) should have Streamlit equivalents
- The app uses Plotly while notebooks use mix of matplotlib, seaborn, and plotly
- Some visualizations are dataset-specific and may need adaptation

## Action Items

1. Review each page systematically against its corresponding notebook
2. Identify which missing visualizations are essential vs nice-to-have
3. Prioritize based on educational value and implementation complexity
4. Update app to match key visualizations from notebooks
5. Consider creating a "visualization completeness" checklist

---

Last Updated: 2025
