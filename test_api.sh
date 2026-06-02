#!/bin/bash
# YOLOv12 Detection API - cURL Examples

echo "╔════════════════════════════════════════════════════════╗"
echo "║     YOLOv12 Detection API - cURL Examples             ║"
echo "╚════════════════════════════════════════════════════════╝"

# Set API URL
API_URL="http://localhost:8000"

# ============================================================
# 1. Health Check
# ============================================================
echo -e "\n\n=== 1. Health Check ==="
curl -X GET "${API_URL}/health" \
  -H "accept: application/json" | jq .


# ============================================================
# 2. Get Available Models
# ============================================================
echo -e "\n\n=== 2. Get Available Models ==="
curl -X GET "${API_URL}/models" \
  -H "accept: application/json" | jq .


# ============================================================
# 3. Predict dari File Upload
# ============================================================
echo -e "\n\n=== 3. Predict from File Upload ==="
# Pastikan Anda memiliki file image.jpg di directory ini
if [ -f "image.jpg" ]; then
  curl -X POST "${API_URL}/predict" \
    -H "accept: application/json" \
    -F "file=@image.jpg" \
    -F "model_name=yolov12m.pt" \
    -F "conf_threshold=0.25" \
    -F "img_size=640" | jq .
else
  echo "File image.jpg not found. Please provide an image file."
fi


# ============================================================
# 4. Predict dari URL
# ============================================================
echo -e "\n\n=== 4. Predict from URL ==="
curl -X POST "${API_URL}/predict-url" \
  -H "accept: application/json" \
  -G \
  --data-urlencode "image_url=https://ultralytics.com/images/bus.jpg" \
  --data-urlencode "model_name=yolov12m.pt" \
  --data-urlencode "conf_threshold=0.25" \
  --data-urlencode "img_size=640" | jq .


# ============================================================
# 5. Switch Model
# ============================================================
echo -e "\n\n=== 5. Switch Model ==="
curl -X POST "${API_URL}/switch-model" \
  -H "accept: application/json" \
  -G \
  --data-urlencode "model_name=yolov12l.pt" | jq .


# ============================================================
# 6. Different Confidence Thresholds
# ============================================================
echo -e "\n\n=== 6. Predict with Different Confidence Thresholds ==="

for conf_thresh in 0.25 0.50 0.75; do
  echo -e "\n--- Confidence Threshold: $conf_thresh ---"
  curl -X POST "${API_URL}/predict-url" \
    -H "accept: application/json" \
    -G \
    --data-urlencode "image_url=https://ultralytics.com/images/bus.jpg" \
    --data-urlencode "model_name=yolov12m.pt" \
    --data-urlencode "conf_threshold=$conf_thresh" \
    --data-urlencode "img_size=640" | jq '.detection_count'
done


# ============================================================
# 7. Different Image Sizes
# ============================================================
echo -e "\n\n=== 7. Predict with Different Image Sizes ==="

for img_size in 320 640 1280; do
  echo -e "\n--- Image Size: $img_size ---"
  curl -X POST "${API_URL}/predict-url" \
    -H "accept: application/json" \
    -G \
    --data-urlencode "image_url=https://ultralytics.com/images/bus.jpg" \
    --data-urlencode "model_name=yolov12m.pt" \
    --data-urlencode "conf_threshold=0.25" \
    --data-urlencode "img_size=$img_size" | jq '.detection_count'
done


# ============================================================
# 8. Test All Available Models
# ============================================================
echo -e "\n\n=== 8. Test All Available Models ==="

models=("yolov12n.pt" "yolov12s.pt" "yolov12m.pt" "yolov12l.pt" "yolov12x.pt")

for model in "${models[@]}"; do
  echo -e "\n--- Model: $model ---"
  curl -s -X POST "${API_URL}/predict-url" \
    -H "accept: application/json" \
    -G \
    --data-urlencode "image_url=https://ultralytics.com/images/bus.jpg" \
    --data-urlencode "model_name=$model" \
    --data-urlencode "conf_threshold=0.25" \
    --data-urlencode "img_size=640" | jq '{model: .model, detections: .detection_count}'
done

echo -e "\n\n✓ All examples completed!"
