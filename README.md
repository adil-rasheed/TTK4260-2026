# TTK4260-2026

Multivariate Statistical Analysis and Machine Learning Course Materials

This repository contains interactive Jupyter notebooks for various statistical analysis and machine learning techniques:

- **Independent Component Analysis (ICA)** - Blind source separation, image and audio processing
- **Principal Component Analysis (PCA)** - Dimensionality reduction, data visualization
- **Partial Least Squares Regression (PLSR)** - Predictive modeling with correlated features
- **Principal Component Regression (PCR)** - Regression with dimensionality reduction
- **Least Squares and Maximum Likelihood** - Parameter estimation fundamentals
- **Performance Metrics** - Model evaluation and validation
- **Multiple Linear Regression (MLR)** - Linear modeling with regularization

## 🚀 Quick Setup with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management.

### Windows (PowerShell)

```powershell
# Run the setup script
.\uv-setup.ps1
```

Or manually:

```powershell
# 1. Create virtual environment
uv venv

# 2. Install dependencies
uv pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn opencv-python imageio ipywidgets jupyter notebook ipykernel plotly Pillow

# 3. Register Jupyter kernel
.venv\Scripts\python.exe -m ipykernel install --user --name ttk4260-uv --display-name "Python (TTK4260-uv)"
```

### Unix/MacOS

```bash
# 1. Create virtual environment
uv venv

# 2. Install dependencies
uv pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn opencv-python imageio ipywidgets jupyter notebook ipykernel plotly Pillow

# 3. Register Jupyter kernel
.venv/bin/python -m ipykernel install --user --name ttk4260-uv --display-name "Python (TTK4260-uv)"
```

## 📓 Using the Notebooks in VS Code

1. Open a notebook (`.ipynb` file)
2. Click the **kernel selector** in the top-right corner
3. Select **"Python (TTK4260-uv)"** from the list
4. Run the cells!

## 📦 Managing Dependencies

```bash
# Install a new package
uv pip install <package-name>

# List installed packages
uv pip list

# Update all packages
uv pip install --upgrade numpy pandas scipy matplotlib
```

## 🔧 Troubleshooting

### Kernel not showing up?

Restart VS Code or reload the window (Ctrl+Shift+P → "Developer: Reload Window")

### Import errors?

Make sure you've selected the correct kernel (Python (TTK4260-uv)) in the notebook.

### Need to reinstall?

```powershell
# Remove environment
Remove-Item -Recurse -Force .venv

# Run setup again
.\uv-setup.ps1
```

## 📚 Course Topics

Each folder contains notebooks with detailed explanations and interactive examples:

- `IndependentComponentAnalysis/` - ICA theory and applications
- `PrincipalComponentAnalysis/` - PCA techniques and visualizations
- `PLSR/` - Partial Least Squares Regression
- `PrincipalComponentRegression/` - PCR applications
- `LeastSquares/` - LS estimation
- `MaximumLikelihood/` - MLE theory
- `PerformanceMetrics/` - Model evaluation
- `MultiPleLinearRegression/` - MLR and regularization
- `StreamlitApp/` - Interactive web applications

