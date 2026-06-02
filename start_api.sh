#!/bin/bash
# Startup script untuk YOLO v12 Eye Disease Detection API

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}YOLO v12 Eye Disease Detection API${NC}"
echo -e "${BLUE}Startup Script${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Configuration
MODEL_PATH="runs/detect/train4/weights/best.pt"
PORT=${PORT:-8000}
WORKERS=${WORKERS:-4}
HOST=${HOST:-0.0.0.0}

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}✗ Model not found at: $MODEL_PATH${NC}"
    echo "Please ensure model is trained and saved at: runs/detect/train4/weights/best.pt"
    exit 1
fi

echo -e "${GREEN}✓ Model found: $MODEL_PATH${NC}\n"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python $python_version${NC}"

# Check required packages
required_packages=("fastapi" "uvicorn" "torch" "ultralytics" "cv2" "numpy")
for package in "${required_packages[@]}"; do
    if python3 -c "import ${package}" 2>/dev/null; then
        echo -e "${GREEN}✓ ${package} installed${NC}"
    else
        echo -e "${RED}✗ ${package} not installed${NC}"
        echo "Run: pip install -r requirements-api-v2.txt"
        exit 1
    fi
done

echo ""

# Create necessary directories
mkdir -p logs uploads runs/detect/train4/weights

# Check GPU availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}✓ GPU available: $GPU_NAME${NC}"
    DEVICE="cuda"
else
    echo -e "${YELLOW}⚠ GPU not available, using CPU${NC}"
    DEVICE="cpu"
fi

echo ""

# Start API server
echo -e "${YELLOW}Starting API server...${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "  - Host: $HOST"
echo "  - Port: $PORT"
echo "  - Workers: $WORKERS"
echo "  - Device: $DEVICE"
echo "  - Model: $MODEL_PATH"
echo ""

echo -e "${GREEN}Server running at:${NC}"
echo -e "${BLUE}  - API Documentation: http://localhost:$PORT/v1/docs${NC}"
echo -e "${BLUE}  - Alternative Docs: http://localhost:$PORT/v1/redoc${NC}"
echo -e "${BLUE}  - Health Check: http://localhost:$PORT/v1/health${NC}"
echo ""

# Run server
python3 -m uvicorn api_v2:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --reload \
    --log-level info
