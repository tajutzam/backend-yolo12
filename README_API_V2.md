# YOLO v12 Eye Disease Detection API - Documentation

## 📋 Overview

FastAPI application untuk **Deteksi Otomatis Penyakit Mata Anterior pada Citra Non-Fundus menggunakan YOLOv12 Berbasis Attention dan Explainable AI**.

### Research Title

> DETEKSI OTOMATIS PENYAKIT MATA ANTERIOR PADA CITRA NON-FUNDUS MENGGUNAKAN YOLOV12 BERBASIS ATTENTION DAN EXPLAINABLE AI
>
> Automatic Detection of Anterior Eye Diseases from Non-Fundus Images using YOLOv12 with Attention Mechanisms and Explainable AI

### Key Features

- ✅ **Real-time Detection**: YOLO v12 dengan attention mechanism (CBAM)
- ✅ **Multi-format Support**: JPEG, PNG, BMP, WebP
- ✅ **Batch Processing**: Process multiple images efficiently
- ✅ **Explainable AI**: Confidence scores, bounding boxes, heatmaps
- ✅ **RESTful API**: Standard REST endpoints dengan Swagger documentation
- ✅ **Production Ready**: Docker, error handling, logging
- ✅ **CORS Enabled**: Cross-origin requests supported

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd yolov12

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-api-v2.txt
```

### 2. Verify Model Path

Ensure trained model exists:

```
runs/detect/train4/weights/best.pt
```

### 3. Run API Server

```bash
# Development mode (with auto-reload)
python -m uvicorn api_v2:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn api_v2:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Access API

- **Swagger UI**: http://localhost:8000/v1/docs
- **ReDoc**: http://localhost:8000/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/v1/openapi.json

---

## 📚 API Endpoints

### Health Check

```http
GET /v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_info": {
    "model_path": "runs/detect/train4/weights/best.pt",
    "num_classes": 5,
    "classes": {
      "0": "Cataracts",
      "1": "Pterygium",
      "2": "Diabetes_Retinopathy",
      "3": "Keratoconus",
      "4": "Normal"
    }
  },
  "timestamp": "2026-04-19T10:30:00.123456"
}
```

---

### Model Information

```http
GET /v1/model/info
```

**Response:**

```json
{
  "success": true,
  "model": {
    "name": "YOLOv12 with Attention Mechanism",
    "task": "Object Detection",
    "path": "runs/detect/train4/weights/best.pt",
    "device": "cuda",
    "num_classes": 5,
    "classes": {
      "0": "Cataracts",
      "1": "Pterygium",
      "2": "Diabetes_Retinopathy",
      "3": "Keratoconus",
      "4": "Normal"
    }
  },
  "research": {
    "title": "DETEKSI OTOMATIS PENYAKIT MATA ANTERIOR...",
    "framework": "YOLOv12",
    "enhancement": "CBAM (Convolutional Block Attention Module)",
    "interpretability": "Grad-CAM for explainable AI"
  },
  "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
  "api_version": "2.0.0",
  "timestamp": "2026-04-19T10:30:00.123456"
}
```

---

### Single Image Detection

```http
POST /v1/detect
```

**Parameters:**

- `file` (required): Image file (multipart/form-data)
- `conf` (optional): Confidence threshold, default 0.25, range [0.1, 1.0]
- `iou` (optional): IOU threshold for NMS, default 0.45, range [0.1, 1.0]
- `return_image` (optional): Return annotated image in base64, default false

**Example Request:**

```bash
curl -X POST "http://localhost:8000/v1/detect" \
  -F "file=@eye_image.jpg" \
  -F "conf=0.3" \
  -F "iou=0.45" \
  -F "return_image=true"
```

**Response:**

```json
{
  "success": true,
  "message": "Deteksi selesai: 2 penyakit ditemukan",
  "timestamp": "2026-04-19T10:30:00.123456",
  "image_metadata": {
    "filename": "eye_image.jpg",
    "size": 45234,
    "width": 640,
    "height": 480,
    "channels": 3
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "Cataracts",
      "confidence": 0.892,
      "bbox": {
        "x1": 100.5,
        "y1": 120.3,
        "x2": 250.8,
        "y2": 300.2
      },
      "bbox_normalized": {
        "x1": 0.157,
        "y1": 0.251,
        "x2": 0.392,
        "y2": 0.625
      },
      "area_pixels": 27500
    },
    {
      "class_id": 1,
      "class_name": "Pterygium",
      "confidence": 0.756,
      "bbox": {
        "x1": 350.2,
        "y1": 150.5,
        "x2": 480.9,
        "y2": 350.8
      },
      "bbox_normalized": {
        "x1": 0.547,
        "y1": 0.314,
        "x2": 0.752,
        "y2": 0.731
      },
      "area_pixels": 35000
    }
  ],
  "statistics": {
    "total_detections": 2,
    "average_confidence": 0.824,
    "min_confidence": 0.756,
    "max_confidence": 0.892,
    "image_area_pixels": 307200,
    "detection_density": 0.000651
  },
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAUA..."
}
```

---

### Batch Detection

```http
POST /v1/detect/batch
```

**Parameters:**

- `files` (required): Multiple image files (multipart/form-data)
- `conf` (optional): Confidence threshold, default 0.25
- `iou` (optional): IOU threshold, default 0.45

**Maximum:** 50 images per batch

**Example Request:**

```bash
curl -X POST "http://localhost:8000/v1/detect/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  -F "files=@image3.jpg" \
  -F "conf=0.3" \
  -F "iou=0.45"
```

**Response:**

```json
{
  "success": true,
  "message": "Batch processing selesai: 3/3 berhasil",
  "timestamp": "2026-04-19T10:30:00.123456",
  "total_images": 3,
  "processed_images": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "detections": 2,
      "image_shape": { "width": 640, "height": 480 }
    },
    {
      "filename": "image2.png",
      "success": true,
      "detections": 1,
      "image_shape": { "width": 800, "height": 600 }
    },
    {
      "filename": "image3.jpg",
      "success": true,
      "detections": 0,
      "image_shape": { "width": 720, "height": 540 }
    }
  ]
}
```

---

### Supported Formats

```http
GET /v1/supported-formats
```

**Response:**

```json
{
  "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
  "max_file_size_mb": 50,
  "max_batch_size": 50
}
```

---

## 🐍 Python Client Usage

### Installation

```bash
pip install requests pillow
```

### Basic Usage

```python
from client_v2 import EyeDiseaseAPIClient

# Initialize client
client = EyeDiseaseAPIClient("http://localhost:8000")

# Health check
health = client.health_check()
print(health)

# Get model info
info = client.get_model_info()
print(info)

# Single image detection
response = client.detect(
    "eye_image.jpg",
    conf=0.3,
    iou=0.45,
    return_image=True
)

# Process results
if response['success']:
    print(f"Found {len(response['detections'])} detections")
    for det in response['detections']:
        print(f"  - {det['class_name']}: {det['confidence']:.3f}")

    # Save annotated image
    client.save_annotated_image(response, "output_annotated.png")

# Batch detection
batch_response = client.batch_detect(
    ["image1.jpg", "image2.jpg", "image3.jpg"],
    conf=0.3
)
print(f"Processed: {batch_response['processed_images']}/{batch_response['total_images']}")
```

### Command Line Usage

```bash
# Health check
python client_v2.py --health

# Model information
python client_v2.py --info

# Single image detection
python client_v2.py --image eye_image.jpg --conf 0.3 --save-image output.png

# Batch detection
python client_v2.py --batch image1.jpg image2.jpg image3.jpg --conf 0.3

# Custom API URL
python client_v2.py --url http://remote-api.com:8000 --image eye_image.jpg
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -f Dockerfile-api-v2 -t yolov12-eye-disease-api:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -v $(pwd)/runs:/app/runs:ro \
  -e CUDA_VISIBLE_DEVICES=0 \
  yolov12-eye-disease-api:latest
```

### Docker Compose

```bash
# Development mode
docker-compose -f docker-compose-api-v2.yml up

# Production mode (detached)
docker-compose -f docker-compose-api-v2.yml up -d

# View logs
docker-compose -f docker-compose-api-v2.yml logs -f api

# Stop services
docker-compose -f docker-compose-api-v2.yml down
```

### With GPU Support

Uncomment GPU section in docker-compose-api-v2.yml:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 🧪 Testing

### Run Test Script

```bash
bash test_api_v2.sh
```

### Test with cURL

```bash
# Health check
curl http://localhost:8000/v1/health | jq

# Model info
curl http://localhost:8000/v1/model/info | jq

# Detection
curl -X POST http://localhost:8000/v1/detect \
  -F "file=@bus.jpg" \
  -F "conf=0.25" | jq
```

### Test with Python

```bash
python client_v2.py --health
python client_v2.py --info
python client_v2.py --image bus.jpg --save-image output.png
```

---

## 📊 Performance Optimization

### Model Parameters

- **Confidence Threshold**: Higher = fewer detections, higher precision
- **IOU Threshold**: For Non-Maximum Suppression (NMS), typically 0.45

### API Optimization

- **Batch Processing**: Use `/v1/detect/batch` for multiple images
- **Worker Processes**: `--workers 4` for production
- **GPU**: Enable CUDA for faster inference

### Model Inference Time

- **GPU (NVIDIA RTX)**: ~30-50ms per image (depending on model size)
- **CPU**: ~200-500ms per image

---

## 🔐 Security Considerations

1. **Input Validation**: File size limited to 50MB
2. **Supported Formats**: Only image extensions accepted
3. **CORS**: Configured for development (restrict in production)
4. **Error Handling**: Detailed errors in development, generic in production

### Production Recommendations

1. Use environment variables for configuration
2. Implement API authentication (JWT, API keys)
3. Add rate limiting
4. Use HTTPS with SSL certificates
5. Deploy behind Nginx reverse proxy
6. Monitor logs and metrics

---

## 📈 Monitoring & Logging

### Log Levels

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: An indication that something unexpected happened
- `ERROR`: A serious problem, something could not be performed

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Files Location

```
logs/api.log
logs/inference.log
```

---

## 🐛 Troubleshooting

### Model Not Loading

```
Error: Model file tidak ditemukan
Solution: Verify path: runs/detect/train4/weights/best.pt
```

### CUDA Out of Memory

```
Error: RuntimeError: CUDA out of memory
Solution: Reduce batch size or use CPU
```

### Image Format Not Supported

```
Error: File format tidak didukung
Solution: Use JPEG, PNG, BMP, or WebP formats
```

### Slow Inference

```
Solutions:
1. Reduce image size (imgsz parameter)
2. Reduce confidence threshold
3. Use GPU instead of CPU
4. Batch processing instead of single images
```

---

## 📚 References

### Research Papers

- YOLOv12: [arxiv.org/papers]
- CBAM: Convolutional Block Attention Module
- Grad-CAM: Visual Explanations from Deep Networks

### Documentation

- FastAPI: https://fastapi.tiangolo.com/
- Ultralytics YOLO: https://docs.ultralytics.com/
- OpenCV: https://docs.opencv.org/

---

## 📝 API Changelog

### Version 2.0.0 (Current)

- ✅ FastAPI implementation
- ✅ Batch processing support
- ✅ Explainable AI features
- ✅ Comprehensive error handling
- ✅ Docker deployment

### Version 1.0.0 (Previous)

- ✅ Gradio web interface

---

## 📧 Support

For issues or questions:

1. Check logs: `docker logs yolov12-eye-disease-api`
2. Review API docs: http://localhost:8000/v1/docs
3. Test with client: `python client_v2.py --health`

---

**Last Updated**: April 2026  
**API Version**: 2.0.0  
**Framework**: FastAPI 0.104.1 + YOLOv12
