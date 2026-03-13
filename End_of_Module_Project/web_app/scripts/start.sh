#!/bin/bash

# Tubonge - Quick Start Script

echo "=========================================="
echo "🎙️  Tubonge - Speech Analytics"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies (force reinstall to ensure all are present)
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Start the server
echo "🚀 Starting Tubonge server..."
echo ""
echo "📍 Application: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🔧 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

python main.py
