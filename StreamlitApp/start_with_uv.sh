#!/usr/bin/env bash
# Quick start script for UV package manager

set -e

echo "🚀 TTK4260 Streamlit App - UV Quick Start"
echo "=========================================="
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ UV installed successfully!"
    echo ""
    echo "⚠️  Please restart your terminal and run this script again."
    exit 0
fi

echo "✅ UV is installed"
echo ""

# Sync dependencies
echo "📦 Installing dependencies with UV..."
uv sync

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "🌐 Starting Streamlit app..."
echo ""

# Run the app
uv run streamlit run app.py
