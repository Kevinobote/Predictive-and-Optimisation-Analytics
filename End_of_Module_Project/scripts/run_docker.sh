#!/bin/bash

# Kiswahili Speech Analytics - Docker Deployment Script

echo "=========================================="
echo "Kiswahili Speech Analytics System"
echo "Docker Deployment"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Build and run
echo "🔨 Building Docker image..."
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
    echo ""
    echo "🚀 Starting container..."
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        echo "✅ Container started successfully"
        echo ""
        echo "=========================================="
        echo "System is running!"
        echo "=========================================="
        echo "🌐 Web Interface: http://localhost:8000/static/index.html"
        echo "📡 API Endpoint: http://localhost:8000/analyze"
        echo "💚 Health Check: http://localhost:8000/health"
        echo ""
        echo "📋 Useful commands:"
        echo "  - View logs: docker-compose logs -f"
        echo "  - Stop system: docker-compose down"
        echo "  - Restart: docker-compose restart"
        echo "=========================================="
    else
        echo "❌ Failed to start container"
        exit 1
    fi
else
    echo "❌ Failed to build Docker image"
    exit 1
fi
