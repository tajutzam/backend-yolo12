#!/bin/bash
# Test script untuk YOLO v12 Eye Disease Detection API

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}YOLO v12 Eye Disease Detection API${NC}"
echo -e "${BLUE}Test Script${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Configuration
API_URL="http://localhost:8000"
TIMEOUT=30

# 1. Health Check
echo -e "${YELLOW}[1] Health Check${NC}"
if curl -s --max-time $TIMEOUT "$API_URL/v1/health" > /dev/null; then
    echo -e "${GREEN}✓ API is running${NC}\n"
else
    echo -e "${RED}✗ API is not accessible at $API_URL${NC}"
    exit 1
fi

# 2. Model Info
echo -e "${YELLOW}[2] Model Information${NC}"
curl -s --max-time $TIMEOUT "$API_URL/v1/model/info" | jq '.model' 2>/dev/null || echo "Could not retrieve model info"
echo ""

# 3. Supported Formats
echo -e "${YELLOW}[3] Supported Formats${NC}"
curl -s --max-time $TIMEOUT "$API_URL/v1/supported-formats" | jq '.' 2>/dev/null || echo "Could not retrieve supported formats"
echo ""

# 4. Test Detection with Sample Image
if [ -f "bus.jpg" ]; then
    echo -e "${YELLOW}[4] Detection Test (bus.jpg)${NC}"
    echo "Uploading image and running inference..."
    
    RESPONSE=$(curl -s --max-time $TIMEOUT \
        -F "file=@bus.jpg" \
        -F "conf=0.25" \
        -F "iou=0.45" \
        -F "return_image=true" \
        "$API_URL/v1/detect")
    
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
    echo ""
else
    echo -e "${YELLOW}[4] Detection Test${NC}"
    echo -e "${RED}⚠ bus.jpg not found. Skipping test.${NC}\n"
fi

# 5. Python Client Test
echo -e "${YELLOW}[5] Python Client Test${NC}"
if command -v python3 &> /dev/null; then
    python3 client_v2.py --url "$API_URL" --health
else
    echo -e "${RED}Python3 not found${NC}"
fi

echo -e "\n${GREEN}Test completed!${NC}"
