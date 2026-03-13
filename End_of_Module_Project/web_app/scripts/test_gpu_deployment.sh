#!/bin/bash

# Test GPU-Accelerated Deployment

echo "🧪 Testing GPU-Accelerated Tubonge Deployment"
echo "=============================================="

# Check if URL is provided
if [ -z "$1" ]; then
    echo "Usage: ./test_gpu_deployment.sh <modal-url>"
    echo "Example: ./test_gpu_deployment.sh https://username--tubonge-fastapi-app.modal.run"
    exit 1
fi

URL=$1

echo ""
echo "📍 Testing URL: $URL"
echo ""

# Test 1: Health Check
echo "1️⃣  Testing Health Endpoint..."
HEALTH=$(curl -s "$URL/health")
echo "$HEALTH" | python3 -m json.tool

GPU_AVAILABLE=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('gpu_available', False))")
GPU_NAME=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('gpu_name', 'Unknown'))")

if [ "$GPU_AVAILABLE" = "True" ]; then
    echo "✅ GPU Detected: $GPU_NAME"
else
    echo "❌ GPU Not Available"
fi

echo ""

# Test 2: Stats Endpoint
echo "2️⃣  Testing Stats Endpoint..."
STATS=$(curl -s "$URL/api/stats")
echo "$STATS" | python3 -m json.tool

echo ""

# Test 3: Audio Analysis (if test file exists)
if [ -f "test_audio.mp3" ] || [ -f "test_audio.wav" ]; then
    echo "3️⃣  Testing Audio Analysis..."
    
    if [ -f "test_audio.mp3" ]; then
        TEST_FILE="test_audio.mp3"
    else
        TEST_FILE="test_audio.wav"
    fi
    
    echo "   Uploading: $TEST_FILE"
    START_TIME=$(date +%s)
    
    RESULT=$(curl -s -X POST "$URL/api/analyze" \
        -F "audio=@$TEST_FILE" \
        -F "language=en")
    
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    
    echo "$RESULT" | python3 -m json.tool
    
    PROCESSING_TIME=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('processing_time', 0))")
    GPU_USED=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('gpu_used', False))")
    
    echo ""
    echo "   Processing Time: ${PROCESSING_TIME}s"
    echo "   Total Time: ${TOTAL_TIME}s"
    echo "   GPU Used: $GPU_USED"
    
    if [ "$GPU_USED" = "True" ]; then
        echo "   ✅ GPU Acceleration Working!"
    else
        echo "   ⚠️  GPU Not Used in Processing"
    fi
else
    echo "3️⃣  Skipping Audio Test (no test_audio.mp3 or test_audio.wav found)"
    echo "   Create a test file to run full test"
fi

echo ""
echo "=============================================="
echo "✅ Testing Complete"
echo ""
echo "Expected GPU Performance:"
echo "  - Processing Time: 2-3s for 30s audio"
echo "  - GPU Used: true"
echo "  - GPU Name: NVIDIA A100-SXM4-40GB"
