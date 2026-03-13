#!/bin/bash

# Quick Modal Deployment Script for Tubonge

echo "🎙️  Tubonge Modal Deployment"
echo "================================"

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal not found. Installing..."
    pip install modal
fi

# Check if authenticated
if ! modal token check &> /dev/null; then
    echo "🔐 Please authenticate with Modal:"
    modal token new
fi

echo ""
echo "Choose deployment option:"
echo "1) Deploy with A100 GPU (fast, for demo)"
echo "2) Deploy with CPU only (slower, cheaper)"
echo "3) Serve locally (development)"
echo "4) Stop deployment"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Deploying with A100 GPU..."
        modal deploy modal_app.py
        echo ""
        echo "✅ Deployment complete!"
        echo "📝 Remember to stop the app when done: modal app stop tubonge"
        ;;
    2)
        echo "🚀 Deploying with CPU only..."
        modal deploy modal_app_cpu.py
        echo ""
        echo "✅ Deployment complete!"
        ;;
    3)
        echo "🔧 Starting local development server..."
        modal serve modal_app.py
        ;;
    4)
        echo "🛑 Stopping deployment..."
        modal app stop tubonge
        modal app stop tubonge-cpu
        echo "✅ Apps stopped"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
