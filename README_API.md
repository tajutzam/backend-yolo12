# YOLOv12 FastAPI Backend - Complete Setup

Ini adalah backend FastAPI yang sudah sepenuhnya siap untuk deteksi objek menggunakan model YOLOv12 yang sudah dilatih.

## 📁 Files Created

### Core Files

- **`main.py`** - FastAPI backend dengan semua endpoint
- **`requirements-api.txt`** - Dependencies untuk API

### Documentation

- **`API_DOCUMENTATION.md`** - Dokumentasi lengkap API (endpoints, parameters, response format)
- **`QUICK_START.md`** - Quick start guide untuk menjalankan API
- **`README.md`** (file ini) - Overview project

### Examples & Testing

- **`example_client.py`** - Python client dengan berbagai contoh penggunaan
- **`test_api.sh`** - Bash script untuk test API menggunakan curl
- **`frontend-integration.js`** - Contoh integrasi dengan JavaScript/React/Node.js

### Docker

- **`Dockerfile-api`** - Docker image untuk API
- **`docker-compose-api.yml`** - Docker Compose configuration

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-api.txt
```

### 2. Run API

```bash
python main.py
```

Server akan berjalan di `http://localhost:8000`

### 3. Access Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📡 API Endpoints

| Endpoint        | Method | Description              |
| --------------- | ------ | ------------------------ |
| `/`             | GET    | Info API                 |
| `/health`       | GET    | Health check             |
| `/models`       | GET    | List available models    |
| `/predict`      | POST   | Predict dari file upload |
| `/predict-url`  | POST   | Predict dari URL         |
| `/switch-model` | POST   | Switch model             |

---

## 🎯 Main Features

✅ **Model Loading** - Load YOLOv12 models (n, s, m, l, x sizes)  
✅ **File Upload** - Upload gambar untuk deteksi  
✅ **URL Support** - Deteksi dari URL image  
✅ **Model Switching** - Switch model on-the-fly tanpa restart  
✅ **Model Caching** - Models cached untuk performa optimal  
✅ **CORS Support** - Siap untuk frontend integration  
✅ **Detailed Results** - Bounding boxes, confidence scores, class names  
✅ **Configurable** - Tunable confidence threshold & image size  
✅ **Docker Ready** - Docker & Docker Compose support

---

## 💡 Usage Examples

### Python Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/predict-url",
    params={
        "image_url": "https://ultralytics.com/images/bus.jpg",
        "model_name": "yolov12m.pt",
        "conf_threshold": 0.25,
        "img_size": 640
    }
)

result = response.json()
print(f"Found {result['detection_count']} objects")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict-url" \
  -G \
  --data-urlencode "image_url=https://ultralytics.com/images/bus.jpg" \
  --data-urlencode "model_name=yolov12m.pt"
```

### JavaScript

```javascript
const client = new YOLOv12APIClient("http://localhost:8000");

const result = await client.predictFromUrl(
  "https://ultralytics.com/images/bus.jpg",
  "yolov12m.pt",
  0.25,
  640,
);

console.log(`Found ${result.detection_count} objects`);
```

---

## 🔧 Configuration

### Model Selection

- `yolov12n.pt` - Nano (Paling cepat)
- `yolov12s.pt` - Small
- `yolov12m.pt` - Medium (default)
- `yolov12l.pt` - Large
- `yolov12x.pt` - XLarge (Paling akurat)

### Parameters

- **conf_threshold** (0.0-1.0): Confidence threshold (default: 0.25)
- **img_size** (320-1280): Input image size (default: 640)

### Performance Tips

1. Use smaller models (n, s) for faster inference
2. Use img_size=320 for speed, 1280 for accuracy
3. Increase conf_threshold to reduce false positives
4. Models are cached after first load

---

## 🐳 Docker Usage

### Build & Run

```bash
docker build -f Dockerfile-api -t yolov12-api .
docker run -p 8000:8000 yolov12-api
```

### Docker Compose

```bash
docker-compose -f docker-compose-api.yml up
```

### GPU Support

Uncomment GPU configuration di `docker-compose-api.yml`:

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

## 📊 Response Format

### Success Response (200)

```json
{
  "status": "success",
  "model": "yolov12m.pt",
  "image_size": 640,
  "confidence_threshold": 0.25,
  "detection_count": 3,
  "detections": [
    {
      "class": 0,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": {
        "x1": 100.5,
        "y1": 150.2,
        "x2": 300.7,
        "y2": 450.3
      },
      "width": 200.2,
      "height": 300.1
    }
  ]
}
```

### Error Response (400/500)

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## 🧪 Testing

### Health Check

```bash
curl http://localhost:8000/health
```

### Test dengan Example Script

```bash
python example_client.py
```

### Full Test Suite

```bash
bash test_api.sh
```

---

## 📚 Documentation Files

Untuk informasi lebih detail, baca:

- [QUICK_START.md](QUICK_START.md) - Panduan singkat
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Dokumentasi lengkap API
- [example_client.py](example_client.py) - Contoh penggunaan Python
- [frontend-integration.js](frontend-integration.js) - Contoh integrasi frontend

---

## 🔧 Troubleshooting

### Model tidak terdownload

- Pastikan internet connection stabil
- Models akan download otomatis dari Ultralytics hub
- Bisa set custom model path di `load_model()` function

### Memory/Performance Issues

- Gunakan model yang lebih kecil (yolov12n.pt)
- Turunkan img_size (default 640, min 320)
- Naikkan conf_threshold untuk filter predictions
- Enable GPU jika tersedia

### Port Sudah Terpakai

- Ubah port di `main.py`: `uvicorn.run(..., port=8001)`
- Atau kill process yang menggunakan port: `lsof -i :8000`

### CORS Issues

- CORS sudah enabled untuk semua origins
- Jika perlu custom CORS, edit middleware di `main.py`

---

## 🎓 Project Structure

```
yolov12/
├── main.py                          # FastAPI backend
├── requirements-api.txt             # Dependencies
├── Dockerfile-api                   # Docker image
├── docker-compose-api.yml           # Docker Compose
│
├── QUICK_START.md                   # Quick start guide
├── API_DOCUMENTATION.md             # Full API docs
├── README.md                        # Project overview (this file)
│
├── example_client.py                # Python client examples
├── test_api.sh                      # Bash test suite
├── frontend-integration.js          # JavaScript examples
│
├── app.py                           # Original Gradio app
├── requirements.txt                 # Original requirements
└── ... (other project files)
```

---

## 🚀 Next Steps

1. ✅ Install dependencies: `pip install -r requirements-api.txt`
2. ✅ Run server: `python main.py`
3. ✅ Open docs: http://localhost:8000/docs
4. ✅ Try example: `python example_client.py`
5. ✅ Integrate dengan aplikasi Anda

---

## 📝 Notes

- Model akan di-download otomatis saat pertama kali digunakan
- Model di-cache untuk performa optimal
- API production-ready dengan error handling
- Semua endpoint mendukung CORS
- Dokumentasi API auto-generated di `/docs`

---

## 🎉 Enjoy!

Backend API YOLOv12 Anda sudah siap digunakan. Untuk pertanyaan atau issues, check API documentation atau examples.

Happy detecting! 🎯
