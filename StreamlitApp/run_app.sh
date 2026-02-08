#!/bin/bash

echo "===================================="
echo " TTK4260 ML Course App Launcher"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

echo "Activating virtual environment..."
source venv/bin/activate
echo ""

echo "Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo ""

echo "===================================="
echo " Starting Streamlit App..."
echo "===================================="
echo ""
echo "App will open in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "===================================="
echo ""

streamlit run app.py
