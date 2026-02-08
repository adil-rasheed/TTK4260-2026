@echo off
REM Quick start script for UV package manager (Windows)

echo.
echo 🚀 TTK4260 Streamlit App - UV Quick Start
echo ==========================================
echo.

REM Check if UV is installed
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ UV is not installed. Installing UV...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo ✅ UV installed successfully!
    echo.
    echo ⚠️  Please restart your terminal and run this script again.
    pause
    exit /b 0
)

echo ✅ UV is installed
echo.

REM Sync dependencies
echo 📦 Installing dependencies with UV...
uv sync

echo.
echo ✅ Dependencies installed!
echo.
echo 🌐 Starting Streamlit app...
echo.

REM Run the app
uv run streamlit run app.py
