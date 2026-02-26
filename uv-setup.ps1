# Quick start script for uv environment
# Usage: .\uv-setup.ps1

Write-Host "🚀 TTK4260 Environment Setup with uv" -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "❌ uv is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "   PowerShell: powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ uv is installed" -ForegroundColor Green

# Check if venv exists
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    uv venv
} else {
    Write-Host "✓ Virtual environment exists" -ForegroundColor Green
}

# Install dependencies
Write-Host "📥 Installing/updating dependencies..." -ForegroundColor Yellow
uv pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn opencv-python imageio ipywidgets jupyter notebook ipykernel plotly Pillow

# Install Jupyter kernel
Write-Host "🔧 Registering Jupyter kernel..." -ForegroundColor Yellow
.venv\Scripts\python.exe -m ipykernel install --user --name ttk4260-uv --display-name "Python (TTK4260-uv)"

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Summary:" -ForegroundColor Cyan
Write-Host "  • Environment: .venv (Python $(& .venv\Scripts\python.exe --version | Out-String).Trim())"
Write-Host "  • Kernel: Python (TTK4260-uv)"
Write-Host "  • Manager: uv"
Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Cyan
Write-Host "  1. In VS Code, open a notebook"
Write-Host "  2. Click the kernel selector (top-right)"
Write-Host "  3. Select 'Python (TTK4260-uv)'"
Write-Host ""
Write-Host "💡 Useful commands:" -ForegroundColor Cyan
Write-Host "  • Install package: uv pip install <package>"
Write-Host "  • List packages:   uv pip list"
Write-Host "  • Activate env:    .venv\Scripts\activate"
Write-Host ""
