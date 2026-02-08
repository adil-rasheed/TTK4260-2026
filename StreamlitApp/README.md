# TTK4260 Interactive Machine Learning Course App

A comprehensive Streamlit web application for exploring machine learning algorithms and statistical methods interactively.

## 🚀 Features

- **8 Interactive Topics**: Least Squares, Maximum Likelihood, MLR & Regularization, PCA Explorer, PCA Application (Real Data), PCR Interactive Demo, PCR Application (Real Data), Performance Metrics
- **Real-time Visualizations**: 3D surfaces, contour plots, interactive parameter adjustment
- **Model Comparisons**: Side-by-side algorithm evaluation
- **Bootstrap Analysis**: Stability assessment and confidence intervals
- **Real-World Datasets**: Norwegian energy data, McDonald's nutrition data

## 📋 Requirements

- Python 3.9+
- See `pyproject.toml` or `requirements.txt` for package dependencies

## 🔧 Installation

### Option 1: Using UV (Recommended - Fast & Modern)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install UV (if not already installed)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Mac/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the StreamlitApp folder
cd StreamlitApp

# Install dependencies (UV automatically creates/uses virtual environment)
uv sync

# Run the app
uv run streamlit run app.py
```

### Option 2: Using pip

```bash
# Clone or navigate to the StreamlitApp folder
cd StreamlitApp

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using conda

```bash
conda create -n ttk4260 python=3.11
conda activate ttk4260
pip install -r requirements.txt
```

## ▶️ Running the App

### With UV (Recommended)
```bash
# From the StreamlitApp directory
uv run streamlit run app.py
```

### With pip/conda
```bash
# Make sure virtual environment is activated
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📚 Course Topics

### 1. Principal Component Regression (PCR)
- Multicollinearity demonstration
- OLS vs PCR loss surface comparison
- Bootstrap stability analysis
- Method comparison (OLS, Ridge, Lasso, PCR)

### 2. Principal Component Analysis (PCA)
- 2D/3D data visualization
- Variance explained analysis
- Real dataset analysis (CityTemp, McDonald's)
- Outlier detection

### 3. Multiple Linear Regression & Regularization
- L1 (Lasso) and L2 (Ridge) regularization
- ElasticNet
- Regularization path visualization
- Cross-validation for hyperparameter tuning

### 4. Least Squares
- OLS fundamentals
- Gradient descent optimization
- 3D cost surface visualization
- Multiple optimizer comparison

### 5. Maximum Likelihood Estimation
- Distribution gallery (Normal, Bernoulli, Poisson, Exponential)
- Interactive likelihood surface
- MLE vs LS connection
- Parameter estimation

### 6. Performance Metrics
- Regression metrics (MAE, MSE, RMSE, R²)
- Classification metrics (Precision, Recall, F1, ROC-AUC)
- Confusion matrices
- ROC and PR curves

## 📁 Project Structure

```
StreamlitApp/
├── app.py                          # Main application entry point
├── pages/                          # Individual topic pages
│   ├── 01_PCR_Interactive.py      # Principal Component Regression (demo)
│   ├── 02_PCA_Explorer.py         # Principal Component Analysis (explorer)
│   ├── 03_MLR_Regularization.py   # Multiple Linear Regression
│   ├── 04_Least_Squares.py        # OLS and optimization
│   ├── 05_Maximum_Likelihood.py   # MLE methods
│   ├── 06_Performance_Metrics.py  # Model evaluation
│   ├── 07_PCR_Application.py      # PCR on Norwegian Energy Data
│   └── 08_PCA_Application.py      # PCA on McDonald's Menu Data
├── utils/                          # Utility modules
│   ├── data_generator.py          # Synthetic data generation
│   ├── models.py                  # ML model implementations
│   ├── visualizations.py          # Plotting functions
│   └── metrics.py                 # Performance metrics
├── data/                           # Real datasets
│   ├── CityTemp.xlsx
│   ├── macdonald.csv
│   ├── X.txt                      # Norwegian weather data (predictors)
│   └── Y.txt                      # Norwegian energy output (responses)
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎯 Usage Guide

### Getting Started

1. **Launch the app** using `streamlit run app.py`
2. **Select a topic** from the sidebar or page navigation
3. **Adjust parameters** using interactive controls
4. **Generate or upload data** as needed
5. **Explore visualizations** and experiment freely!

### Interactive Controls

- **Sliders**: Adjust continuous parameters (correlation, noise, sample size)
- **Number inputs**: Set specific values (random seed, iterations)
- **Selectboxes**: Choose between options (distributions, methods)
- **Buttons**: Trigger computations and updates
- **Expanders**: Show/hide additional information

### Tips for Best Experience

- Start with **PCR Interactive** to see dramatic multicollinearity effects
- Use **consistent random seeds** for reproducible results
- Try **extreme parameter values** to understand algorithm behavior
- **Compare methods** side-by-side to see trade-offs
- **Hover** over info icons (ℹ️) for explanations

## 🎓 For Students

This app is designed to complement the TTK4260 course at NTNU. Each page corresponds to lecture topics and provides hands-on exploration of concepts covered in class.

### Recommended Learning Path

1. Start with **Least Squares** to understand optimization fundamentals
2. Move to **MLR & Regularization** to see why regularization matters
3. Explore **PCA** to understand dimensionality reduction
4. Study **PCR** to see how PCA solves multicollinearity
5. Review **MLE** for parameter estimation theory
6. Master **Performance Metrics** for model evaluation

## 🐛 Troubleshooting

### App won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Try clearing Streamlit cache: `streamlit cache clear`

### Plots not showing
- Ensure plotly is installed: `pip install plotly`
- Try refreshing the browser page
- Check browser console for errors (F12)

### Slow performance
- Reduce number of samples/bootstrap iterations
- Use smaller parameter ranges
- Enable caching in the code (already implemented)

## 🔄 Updates and Extensions

### Adding New Pages

1. Create new file in `pages/` folder (e.g., `07_New_Topic.py`)
2. Import utilities from `utils/` folder
3. Follow existing page structure for consistency
4. Add page documentation in sidebar

### Adding New Datasets

1. Place dataset in `data/` folder
2. Create loader function in `utils/data_generator.py`
3. Add dataset option in relevant pages

## 📧 Contact

For questions or issues related to this app, please contact the course instructor or TA.

## 📄 License

This application is developed for educational purposes as part of TTK4260 course at NTNU.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/python/)
- ML algorithms implemented using [scikit-learn](https://scikit-learn.org/)
- Based on course materials from TTK4260-2026

---

**Happy Exploring! 🎓📊**
