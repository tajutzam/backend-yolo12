# YOLO v12 Eye Disease Detection - Quick Start Guide

## 🚀 Installation & Setup (5 minutes)

### 1. Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM (minimum)
- 5GB disk space (for model + dependencies)

### 2. Install Dependencies

```bash
# Navigate to project directory
cd yolov12

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements-api-v2.txt
```

### 3. Verify Model

```bash
# Check if trained model exists
ls runs/detect/train4/weights/best.pt

# If model is missing, see training instructions in YOLO12_FIX_BET_ANJ (1).ipynb
```

### 4. Start API Server

#### Option A: Direct Python (Development)

```bash
python -m uvicorn api_v2:app --reload --host 0.0.0.0 --port 8000
```

#### Option B: Startup Scripts

```bash
# Linux/Mac
chmod +x start_api.sh
./start_api.sh

# Windows
start_api.bat
```

#### Option C: Docker

```bash
# Build image
docker build -f Dockerfile-api-v2 -t yolo12-eye-api .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/runs:/app/runs:ro \
  yolo12-eye-api

# Or use Docker Compose
docker-compose -f docker-compose-api-v2.yml up
```

### 5. Access API

Open browser and go to:

- **Swagger UI**: http://localhost:8000/v1/docs
- **ReDoc**: http://localhost:8000/v1/redoc

---

## 📝 API Usage Examples

### Health Check

```bash
curl http://localhost:8000/v1/health | jq
```

### Single Image Detection

```bash
curl -X POST http://localhost:8000/v1/detect \
  -F "file=@eye_image.jpg" \
  -F "conf=0.25" \
  -F "return_image=true" | jq
```

### Batch Detection

```bash
curl -X POST http://localhost:8000/v1/detect/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" | jq
```

### Python Client

```python
from client_v2 import EyeDiseaseAPIClient

client = EyeDiseaseAPIClient("http://localhost:8000")

# Single image
response = client.detect("eye_image.jpg", return_image=True)
print(f"Detections: {len(response['detections'])}")

# Save annotated image
client.save_annotated_image(response, "output.png")
```

---

## 🔧 Configuration

Create `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` to customize:

- API port and host
- Model confidence/IOU thresholds
- CUDA device selection
- Logging level

---

## 🧪 Testing

### Run Test Script

```bash
bash test_api_v2.sh
```

### Test with Python Client

```bash
python client_v2.py --health
python client_v2.py --image eye_image.jpg --save-image output.png
```

### Test Multiple Images

```bash
python client_v2.py --batch image1.jpg image2.jpg image3.jpg
```

---

## 📊 Performance Tips

### Optimize for Speed

```python
# Reduce resolution for faster inference
response = client.detect("image.jpg", conf=0.35)  # Higher threshold = fewer detections
```

### Use Batch Processing

```python
# Better than sequential calls
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_result = client.batch_detect(images)
```

### Enable GPU

```bash
# Automatically detected, but can be forced:
export CUDA_VISIBLE_DEVICES=0
python -m uvicorn api_v2:app --host 0.0.0.0 --port 8000
```

---

## 🐛 Troubleshooting

| Problem            | Solution                                                     |
| ------------------ | ------------------------------------------------------------ |
| Model not found    | Verify: `runs/detect/train4/weights/best.pt` exists          |
| CUDA out of memory | Reduce batch size or use CPU                                 |
| Connection refused | Ensure API is running on correct port (default: 8000)        |
| Slow inference     | Use GPU, reduce image size, or increase confidence threshold |
| File format error  | Use JPEG/PNG/BMP/WebP formats only                           |

---

## 📚 Key Endpoints

| Method | Endpoint                | Purpose                |
| ------ | ----------------------- | ---------------------- |
| GET    | `/v1/health`            | Health check           |
| GET    | `/v1/model/info`        | Model information      |
| POST   | `/v1/detect`            | Single image detection |
| POST   | `/v1/detect/batch`      | Batch detection        |
| GET    | `/v1/supported-formats` | File format info       |

---

## 🔐 Production Deployment

### Using Nginx + Docker Compose

```bash
# Production deployment with reverse proxy
docker-compose -f docker-compose-api-v2.yml up -d

# View logs
docker-compose -f docker-compose-api-v2.yml logs -f api
```

### Security Recommendations

1. ✅ Use HTTPS with SSL certificates
2. ✅ Implement API authentication
3. ✅ Add rate limiting
4. ✅ Monitor logs and metrics
5. ✅ Regular backups of results
6. ✅ Update dependencies regularly

---

## 📞 Support

### Check Logs

```bash
# Docker logs
docker logs yolov12-eye-disease-api

# Or check file
tail -f logs/api.log
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run API with:
python -m uvicorn api_v2:app --log-level debug
```

---

## 📖 More Documentation

- **Full API Docs**: See `README_API_V2.md`
- **Model Training**: See `YOLO12_FIX_BET_ANJ (1).ipynb`
- **Research Details**: See individual research documentation
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Ultralytics Docs**: https://docs.ultralytics.com/

---

## ✨ Research Highlights

**Penelitian:**

- **Judul**: DETEKSI OTOMATIS PENYAKIT MATA ANTERIOR PADA CITRA NON-FUNDUS MENGGUNAKAN YOLOV12 BERBASIS ATTENTION DAN EXPLAINABLE AI
- **Model**: YOLOv12 dengan CBAM Attention Module
- **Interpretability**: Grad-CAM untuk visual explanations
- **Dataset**: Anterior eye disease images (non-fundus)
- **Performance**: Real-time detection dengan high accuracy

---

**Version**: 2.0.0 | **Last Updated**: April 2026
