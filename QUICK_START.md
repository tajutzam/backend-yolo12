# 🚀 Quick Start - YOLOv12 Detection API

## Persyaratan

- Python 3.8+
- pip atau conda

## Installation & Running

### Option 1: Local Python Environment

1. **Install Dependencies**

   ```bash
   pip install -r requirements-api.txt
   ```

2. **Jalankan Server**

   ```bash
   python main.py
   ```

   Output yang diharapkan:

   ```
   INFO:     Started server process [12345]
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

3. **Akses API**
   - API Documentation: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - API Base: http://localhost:8000

---

### Option 2: Docker

1. **Build Image**

   ```bash
   docker build -f Dockerfile-api -t yolov12-api .
   ```

2. **Run Container**
   ```bash
   docker run -p 8000:8000 yolov12-api
   ```

---

### Option 3: Docker Compose

1. **Start Service**

   ```bash
   docker-compose -f docker-compose-api.yml up
   ```

2. **Stop Service**
   ```bash
   docker-compose -f docker-compose-api.yml down
   ```

---

## Test API

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Get Available Models

```bash
curl http://localhost:8000/models
```

### 3. Predict dari URL

```bash
curl -X POST "http://localhost:8000/predict-url?image_url=https://ultralytics.com/images/bus.jpg"
```

### 4. Predict dari File

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

---

## Python Client Example

```python
import requests

# Predict dari URL
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
print(f"Found {result['detection_count']} objects:")
for det in result['detections']:
    print(f"  - {det['class_name']}: {det['confidence']:.2%}")
```

---

## Run Examples

```bash
# Run all health checks and available models
python example_client.py

# Or run specific tests with curl
bash test_api.sh
```

---

## API Endpoints Summary

| Method | Endpoint        | Description              |
| ------ | --------------- | ------------------------ |
| GET    | `/`             | API Info                 |
| GET    | `/health`       | Health Check             |
| GET    | `/models`       | List Available Models    |
| POST   | `/predict`      | Predict dari File Upload |
| POST   | `/predict-url`  | Predict dari URL         |
| POST   | `/switch-model` | Switch Model             |

---

## Troubleshooting

### Port Already in Use

```bash
# Change port in main.py
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use port 8001 instead
```

### Model Download Issues

Models akan di-download otomatis dari Ultralytics hub. Pastikan internet connection stabil.

### Memory Issues

Gunakan model yang lebih kecil:

```bash
curl -X POST "http://localhost:8000/switch-model?model_name=yolov12n.pt"
```

### GPU Support

Untuk mengaktifkan GPU di Docker:

```yaml
# docker-compose-api.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Dokumentasi Lengkap

Lihat [API_DOCUMENTATION.md](API_DOCUMENTATION.md) untuk informasi lengkap tentang semua endpoint dan parameter.

---

## Performance Tips

1. **Model Selection**
   - `yolov12n.pt`: Cepat, cocok untuk real-time
   - `yolov12m.pt`: Balance (default)
   - `yolov12l.pt`/`yolov12x.pt`: Akurat, cocok untuk batch

2. **Image Size**
   - 320: Tercepat
   - 640: Balance (default)
   - 1280: Paling akurat

3. **Confidence Threshold**
   - 0.25: Default, banyak detections
   - 0.50: Moderate filtering
   - 0.75+: Hanya high-confidence detections

---

## Next Steps

1. ✅ API sudah berjalan
2. 📚 Pelajari endpoint di http://localhost:8000/docs
3. 🧪 Test dengan gambar Anda sendiri
4. 🔌 Integrate dengan aplikasi Anda

Enjoy! 🎉
