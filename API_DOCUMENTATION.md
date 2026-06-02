# YOLOv12 Detection API

Backend API FastAPI untuk deteksi objek menggunakan model YOLOv12 yang sudah dilatih.

## Instalasi

### 1. Install Dependencies

```bash
pip install -r requirements-api.txt
```

### 2. Jalankan Server

```bash
python main.py
```

Server akan berjalan di `http://localhost:8000`

## API Documentation

Dokumentasi interaktif tersedia di:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### 1. Health Check

**GET** `/health`

Check apakah API sedang berjalan dan model sudah terload.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 2. List Available Models

**GET** `/models`

Mendapatkan list semua model yang tersedia.

**Response:**

```json
{
  "available_models": [
    "yolov12n.pt",
    "yolov12s.pt",
    "yolov12m.pt",
    "yolov12l.pt",
    "yolov12x.pt"
  ],
  "currently_loaded": ["yolov12m.pt"]
}
```

---

### 3. Predict dari Upload File

**POST** `/predict`

Melakukan deteksi objek pada gambar yang di-upload.

**Parameters:**

- `file` (required): Gambar dalam format JPG/PNG
- `model_name` (optional): Nama model (default: yolov12m.pt)
- `conf_threshold` (optional): Confidence threshold 0.0-1.0 (default: 0.25)
- `img_size` (optional): Ukuran gambar 320-1280 (default: 640)

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@image.jpg" \
  -F "model_name=yolov12m.pt" \
  -F "conf_threshold=0.25" \
  -F "img_size=640"
```

**Response:**

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

---

### 4. Predict dari URL

**POST** `/predict-url`

Melakukan deteksi objek pada gambar dari URL.

**Parameters:**

- `image_url` (required): URL gambar
- `model_name` (optional): Nama model (default: yolov12m.pt)
- `conf_threshold` (optional): Confidence threshold 0.0-1.0 (default: 0.25)
- `img_size` (optional): Ukuran gambar 320-1280 (default: 640)

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/predict-url?image_url=https://example.com/image.jpg&model_name=yolov12m.pt&conf_threshold=0.25&img_size=640"
```

---

### 5. Switch Model

**POST** `/switch-model`

Mengganti model YOLOv12 yang sedang aktif.

**Parameters:**

- `model_name` (required): Nama model yang ingin diaktifkan

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/switch-model?model_name=yolov12l.pt"
```

**Response:**

```json
{
  "status": "success",
  "message": "Switched to model: yolov12l.pt",
  "loaded_models": ["yolov12m.pt", "yolov12l.pt"]
}
```

---

## Contoh Penggunaan dengan Python

### Menggunakan requests library

```python
import requests

# 1. Predict dari file upload
url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
params = {
    "model_name": "yolov12m.pt",
    "conf_threshold": 0.25,
    "img_size": 640
}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"Detections: {result['detection_count']}")
for detection in result['detections']:
    print(f"  - {detection['class_name']}: {detection['confidence']:.2%}")

# 2. Predict dari URL
url = "http://localhost:8000/predict-url"
params = {
    "image_url": "https://example.com/image.jpg",
    "model_name": "yolov12m.pt",
    "conf_threshold": 0.25,
    "img_size": 640
}

response = requests.post(url, params=params)
result = response.json()

print(f"Detections: {result['detection_count']}")

# 3. Switch model
response = requests.post(
    "http://localhost:8000/switch-model",
    params={"model_name": "yolov12l.pt"}
)
print(response.json())
```

---

## Available Models

- `yolov12n.pt` - Nano (Paling cepat, akurasi terendah)
- `yolov12s.pt` - Small
- `yolov12m.pt` - Medium (Default)
- `yolov12l.pt` - Large
- `yolov12x.pt` - Extra Large (Paling akurat, paling lambat)

## Tips Performa

1. **Model Size**: Gunakan model yang lebih kecil (n, s) untuk inference cepat
2. **Image Size**: Ukuran gambar lebih kecil = inference lebih cepat tapi akurasi bisa berkurang
   - 320: Cepat
   - 640: Balance (default)
   - 1280: Akurat tapi lambat
3. **Confidence Threshold**: Naikkan untuk mengurangi false positives
4. **Model Caching**: Model yang sudah diload akan disimpan, tidak perlu diload ulang

## Troubleshooting

### Model tidak ter-download otomatis

Pastikan model `.pt` sudah ada atau internet connection stabil untuk download dari Ultralytics hub.

### Memory error

Gunakan model yang lebih kecil atau turunkan `img_size`. Untuk GPU, pastikan CUDA sudah terinstall dengan benar.

### Slow inference

- Gunakan model yang lebih kecil
- Gunakan GPU (set CUDA_VISIBLE_DEVICES)
- Turunkan img_size
- Gunakan batch processing untuk multiple images

## Struktur Response Detection

Setiap detection object mengandung:

- `class`: ID class (0, 1, 2, ...)
- `class_name`: Nama class (string)
- `confidence`: Confidence score (0.0-1.0)
- `bbox`: Bounding box dengan koordinat x1, y1, x2, y2
- `width`: Lebar detection box
- `height`: Tinggi detection box
