# � Dokumentasi API Lengkap - Panduan Integrasi
## YOLOv12 Eye Disease Detection API v2.0.0

**Deteksi Otomatis Penyakit Mata Anterior menggunakan YOLOv12 dengan Attention Mechanism**

---

## 📚 Dokumentasi Tersedia

Kami menyediakan **3 file dokumentasi komprehensif**:

### 1. 📖 **INTEGRATION_API_DOCUMENTATION.md** (BACA INI DULU)
Dokumentasi lengkap API dengan:
- ✅ Setup & installation
- ✅ Penjelasan semua 6 endpoints
- ✅ Request/Response format
- ✅ Error handling
- ✅ Authentication options
- ✅ Best practices untuk production
- ✅ Contoh integrasi berbagai bahasa (Python, JavaScript, Java, cURL)

**File path:** `d:\UMNFIX\yolov12\INTEGRATION_API_DOCUMENTATION.md`

### 2. 💻 **INTEGRATION_EXAMPLES.md** (PRAKTIK LANGSUNG)
Contoh implementasi praktis dengan:
- ✅ Python basic & advanced client
- ✅ Flask web application
- ✅ React Native mobile app
- ✅ Electron desktop app
- ✅ Docker & Docker Compose
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Monitoring & metrics

**File path:** `d:\UMNFIX\yolov12\INTEGRATION_EXAMPLES.md`

### 3. 🆘 **FAQ_TROUBLESHOOTING.md** (JIKA ADA MASALAH)
Tanya jawab & solusi masalah:
- ✅ 9 FAQ umum dengan jawaban
- ✅ 9 troubleshooting scenarios
- ✅ Performance optimization
- ✅ Debug checklist

**File path:** `d:\UMNFIX\yolov12\FAQ_TROUBLESHOOTING.md`

---

## 🚀 Quick Start (3 Langkah)

### Step 1: Jalankan API

```bash
cd d:\UMNFIX\yolov12
python api_v2.py
```

**Output yang diharapkan:**
```
INFO:     Started server process
INFO:     Model loaded successfully on device: cuda
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Verifikasi API Berjalan

```bash
# Health check
curl http://localhost:8000/v1/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Step 3: Akses Dokumentasi Interaktif

- **Swagger UI**: http://localhost:8000/v1/docs
- **ReDoc**: http://localhost:8000/v1/redoc

---

## 📋 6 Endpoints Utama

| Endpoint | Method | Fungsi | Doc |
|----------|--------|--------|-----|
| `/v1/health` | GET | Check API status | Health Check |
| `/v1/model/info` | GET | Get model info & classes | Model Info |
| `/v1/detect` | POST | Deteksi single image | Single Detection |
| `/v1/detect/batch` | POST | Batch processing (max 50) | Batch Detection |
| `/v1/supported-formats` | GET | Supported formats | Formats |
| `/` | GET | Root info | Root |

**Penjelasan detail setiap endpoint:** Lihat `INTEGRATION_API_DOCUMENTATION.md`

---

## 💡 Gunakan Dokumentasi Sesuai Kebutuhan

### ❓ Saya ingin...

**...memahami API secara keseluruhan**
→ Baca: `INTEGRATION_API_DOCUMENTATION.md` (Main doc)
→ Mulai dari bagian "Setup & Installation"

**...implementasi di project saya**
→ Baca: `INTEGRATION_EXAMPLES.md`
→ Pilih bahasa/framework sesuai project Anda

**...deploy ke production**
→ Baca: `INTEGRATION_API_DOCUMENTATION.md` (Best Practices)
→ Baca: `INTEGRATION_EXAMPLES.md` (Docker & CI-CD)

**...ada error/masalah**
→ Baca: `FAQ_TROUBLESHOOTING.md`
→ Cari masalah Anda di troubleshooting section

**...integrasi dengan web framework**
→ Baca: `INTEGRATION_EXAMPLES.md` → Flask section

**...integrasi dengan mobile**
→ Baca: `INTEGRATION_EXAMPLES.md` → React Native section

**...integrasi dengan desktop app**
→ Baca: `INTEGRATION_EXAMPLES.md` → Electron section

---

## 🔥 Quick Examples

### Python - Deteksi Satu Gambar

```python
import requests

# 1. Siapkan file gambar
with open("image.jpg", "rb") as f:
    files = {"file": f}
    
    # 2. Kirim ke API
    response = requests.post(
        "http://localhost:8000/v1/detect",
        files=files,
        params={"conf": 0.25}
    )

# 3. Tampilkan hasil
result = response.json()
print(f"Total deteksi: {len(result['detections'])}")

for detection in result['detections']:
    print(f"- {detection['class_name']}: {detection['confidence']:.1%}")
```

### JavaScript - Upload & Detect

```javascript
const file = document.getElementById('imageInput').files[0];
const formData = new FormData();
formData.append('file', file);

fetch('http://localhost:8000/v1/detect', {
    method: 'POST',
    body: formData
})
.then(r => r.json())
.then(data => {
    console.log(`Found ${data.detections.length} diseases`);
    data.detections.forEach(det => {
        console.log(`- ${det.class_name}: ${(det.confidence*100).toFixed(1)}%`);
    });
});
```

### Batch Processing (Python)

```python
import requests
from pathlib import Path

# Siapkan list gambar
image_paths = list(Path("images").glob("*.jpg"))[:50]  # max 50

# Create file list
files = [(
    "files",
    open(path, "rb")
) for path in image_paths]

# Send batch request
response = requests.post(
    "http://localhost:8000/v1/detect/batch",
    files=files
)

result = response.json()
print(f"Processed: {result['processed_images']}/{result['total_images']}")
```

---

## 📊 Response Format

### Health Check Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_info": {
    "num_classes": 8,
    "classes": {
      "0": "Pterygium",
      "1": "Cataracts",
      "2": "Diabetic Retinopathy",
      "3": "Glaucoma",
      "4": "AMD",
      "5": "Corneal Abrasion",
      "6": "Dry Eye",
      "7": "Keratoconus"
    }
  }
}
```

### Detection Response

```json
{
  "success": true,
  "message": "Deteksi selesai: 2 penyakit ditemukan",
  "detections": [
    {
      "class_name": "Pterygium",
      "confidence": 0.95,
      "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 450}
    },
    {
      "class_name": "Cataracts",
      "confidence": 0.87,
      "bbox": {"x1": 150, "y1": 100, "x2": 350, "y2": 400}
    }
  ]
}
```

---

## ⚙️ Configuration

### Parameters untuk Detection

```
conf (float): Confidence threshold 0.1-1.0 (default: 0.25)
  - 0.15 = Sensitive (detect lebih banyak)
  - 0.25 = Default (balanced)
  - 0.50 = Strict (deteksi sedikit, akurat)

iou (float): IOU threshold 0.1-1.0 (default: 0.45)
  - Untuk NMS (menghapus duplicate detection)
  - 0.30 = Aggressive (hapus banyak duplicate)
  - 0.45 = Default (balanced)
  - 0.65 = Conservative (keep more)

return_image (boolean): Include annotated image (default: false)
```

### Rekomendasi

```
Untuk production:  conf=0.35, iou=0.5
Untuk demo:        conf=0.25, iou=0.45
Untuk strict:      conf=0.50, iou=0.6
```

---

## 🔐 Authentication & Security

API ini currently tidak memerlukan authentication. Untuk production, pertimbangkan:

```python
# Option 1: API Key
headers = {"X-API-Key": "your-secret-key"}

# Option 2: Bearer Token
headers = {"Authorization": "Bearer YOUR_TOKEN"}

# Option 3: Basic Auth
auth = ("username", "password")
```

**Full security guide:** Lihat `INTEGRATION_API_DOCUMENTATION.md` → Authentication section

---

## 🐍 Python Integration

### Production-Ready Client

```python
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyeDiseaseAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        try:
            r = self.session.get(f"{self.base_url}/v1/health")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return None
    
    def detect_image(self, image_path, conf=0.25):
        try:
            with open(image_path, "rb") as f:
                r = self.session.post(
                    f"{self.base_url}/v1/detect",
                    files={"file": f},
                    params={"conf": conf}
                )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def detect_batch(self, image_paths, conf=0.25):
        files = [(f"files", open(p, "rb")) for p in image_paths]
        try:
            r = self.session.post(
                f"{self.base_url}/v1/detect/batch",
                files=files,
                params={"conf": conf}
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            raise

# Usage
client = EyeDiseaseAPI()
result = client.detect_image("image.jpg")
print(f"Detections: {len(result['detections'])}")
```

**Lebih banyak contoh Python:** Lihat `INTEGRATION_EXAMPLES.md` → Python section

---

## 🌐 Web Framework Integration

### Flask

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    
    r = requests.post(
        "http://localhost:8000/v1/detect",
        files={"file": file}
    )
    
    return jsonify(r.json())

if __name__ == '__main__':
    app.run()
```

### FastAPI

```python
from fastapi import FastAPI, File, UploadFile
import httpx

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "http://localhost:8000/v1/detect",
            files={"file": await file.read()}
        )
    return r.json()
```

### Django

```python
from django.http import JsonResponse
import requests

def detect_view(request):
    file = request.FILES['image']
    
    r = requests.post(
        "http://localhost:8000/v1/detect",
        files={"file": file}
    )
    
    return JsonResponse(r.json())
```

**Full web integration examples:** Lihat `INTEGRATION_EXAMPLES.md` → Web Application section

---

## 🐳 Docker Deployment

### Quick Docker

```bash
# Build
docker build -t eye-disease-api .

# Run
docker run -p 8000:8000 --gpus all eye-disease-api

# Check
curl http://localhost:8000/v1/health
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
    restart: unless-stopped
```

```bash
docker-compose up -d
```

**Docker guide:** Lihat `INTEGRATION_EXAMPLES.md` → Docker section

---

## ⚠️ Error Handling

### Common Errors

```json
// 400 - Bad Request
{
  "success": false,
  "error": "File format tidak didukung"
}

// 413 - Payload Too Large
{
  "success": false,
  "error": "File terlalu besar (max 50MB)"
}

// 503 - Service Unavailable
{
  "success": false,
  "error": "Model tidak tersedia"
}
```

### Error Handling Best Practice

```python
try:
    response = requests.post(
        "http://localhost:8000/v1/detect",
        files=files,
        timeout=30
    )
    response.raise_for_status()
except requests.exceptions.Timeout:
    print("Request timeout")
except requests.exceptions.ConnectionError:
    print("API tidak tersedia")
except requests.exceptions.HTTPError as e:
    error_detail = e.response.json()
    print(f"Error: {error_detail['error']}")
```

**Error handling guide:** Lihat `INTEGRATION_API_DOCUMENTATION.md` → Error Handling

---

## ✅ Verification Checklist

Sebelum production:

- [ ] API berjalan: `curl http://localhost:8000/v1/health`
- [ ] Model ter-load dengan baik
- [ ] Test dengan sample images
- [ ] Error handling implementasi
- [ ] Logging configured
- [ ] Performance acceptable
- [ ] Security configured
- [ ] Documentation team ready

---

## 🆘 Troubleshooting

### API Connection Error

```bash
# Check if API running
curl http://localhost:8000/

# If not, start
python api_v2.py
```

### Model Not Loading

```bash
# Check model file
ls runs/detect/train4/weights/best.pt

# Restart API
python api_v2.py
```

### Performance Issues

```bash
# Check GPU
nvidia-smi

# Use smaller image
# Increase confidence threshold
# Reduce batch size
```

**Full troubleshooting:** Lihat `FAQ_TROUBLESHOOTING.md`

---

## 📞 Support

### Documentation Files
- 📖 **INTEGRATION_API_DOCUMENTATION.md** - Main API spec
- 💻 **INTEGRATION_EXAMPLES.md** - Code examples
- 🆘 **FAQ_TROUBLESHOOTING.md** - Q&A & fixes

### Online Help
- Swagger UI: http://localhost:8000/v1/docs
- ReDoc: http://localhost:8000/v1/redoc

---

## 🎯 Next Steps

1. **Read Main Doc** (30 min)
   - Open: `INTEGRATION_API_DOCUMENTATION.md`

2. **Try Examples** (15 min)
   - Check: `INTEGRATION_EXAMPLES.md`

3. **Implement** (1-2 hours)
   - Copy relevant example
   - Adapt to your project

4. **Deploy** (Varies)
   - Development: `python api_v2.py`
   - Production: Docker / Gunicorn

---

## 📝 Old Integration Guide (Deprecated)

### Using requests (DEPRECATED)

### Using requests

```python
import requests
from pathlib import Path

def detect_objects(image_path: str, api_url: str = "http://localhost:8000"):
    """Detect objects in image using YOLOv12 API"""

    with open(image_path, "rb") as f:
        response = requests.post(
            f"{api_url}/predict",
            files={"file": f},
            params={
                "model_name": "yolov12m.pt",
                "conf_threshold": 0.25,
                "img_size": 640
            }
        )

    return response.json()

# Usage
result = detect_objects("image.jpg")
print(f"Found {result['detection_count']} objects")
```

### Using async/await (aiohttp)

```python
import aiohttp
import asyncio

async def detect_async(image_url: str):
    """Async detection from URL"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/predict-url",
            params={
                "image_url": image_url,
                "model_name": "yolov12m.pt"
            }
        ) as resp:
            return await resp.json()

# Usage
result = asyncio.run(detect_async("https://example.com/image.jpg"))
```

---

## 📱 Node.js / Express Integration

### Basic Express Middleware

```javascript
const express = require("express");
const fetch = require("node-fetch");
const FormData = require("form-data");
const fs = require("fs");

const app = express();

app.post("/detect", async (req, res) => {
  try {
    const file = req.files.image;
    const form = new FormData();

    form.append("file", fs.createReadStream(file.tempFilePath));
    form.append("model_name", req.body.model || "yolov12m.pt");
    form.append("conf_threshold", req.body.conf || 0.25);

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: form,
    });

    const result = await response.json();
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => console.log("Server running on port 3000"));
```

### Axios Integration

```javascript
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

async function detectFromFile(filePath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(filePath));

  const response = await axios.post("http://localhost:8000/predict", form, {
    headers: form.getHeaders(),
    params: {
      model_name: "yolov12m.pt",
      conf_threshold: 0.25,
      img_size: 640,
    },
  });

  return response.data;
}

// Usage
detectFromFile("image.jpg").then((result) => {
  console.log(`Found ${result.detection_count} objects`);
});
```

---

## ⚛️ React Integration

### Functional Component

```jsx
import React, { useState } from "react";
import axios from "axios";

function YOLODetector() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleDetect = async () => {
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await axios.post(
        "http://localhost:8000/predict",
        formData,
        {
          params: {
            model_name: "yolov12m.pt",
            conf_threshold: 0.25,
          },
        },
      );
      setResults(response.data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
        accept="image/*"
      />
      <button onClick={handleDetect} disabled={loading}>
        {loading ? "Detecting..." : "Detect"}
      </button>

      {results && (
        <div>
          <h3>Results</h3>
          <p>Found {results.detection_count} objects</p>
          <ul>
            {results.detections.map((det, idx) => (
              <li key={idx}>
                {det.class_name} - {(det.confidence * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default YOLODetector;
```

### Using React Query

```jsx
import { useMutation } from "@tanstack/react-query";

function YOLODetector() {
  const detectMutation = useMutation((file) => {
    const formData = new FormData();
    formData.append("file", file);
    return fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    }).then((r) => r.json());
  });

  return (
    <div>
      <input
        type="file"
        onChange={(e) => detectMutation.mutate(e.target.files[0])}
      />
      {detectMutation.isLoading && <p>Detecting...</p>}
      {detectMutation.data && (
        <p>Found {detectMutation.data.detection_count} objects</p>
      )}
    </div>
  );
}
```

---

## 🟦 TypeScript Integration

### Type Definitions

```typescript
interface Detection {
  class: number;
  class_name: string;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  width: number;
  height: number;
}

interface PredictionResult {
  status: string;
  model: string;
  image_size: number;
  confidence_threshold: number;
  detections: Detection[];
  detection_count: number;
}

// Client class
class YOLOv12Client {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async predictFromFile(
    file: File,
    modelName: string = "yolov12m.pt",
    confThreshold: number = 0.25,
  ): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(
      `${this.baseUrl}/predict?model_name=${modelName}&conf_threshold=${confThreshold}`,
      { method: "POST", body: formData },
    );

    return response.json();
  }
}
```

---

## 🔥 Firebase Cloud Functions

```javascript
const functions = require("firebase-functions");
const axios = require("axios");

exports.detectObjects = functions.https.onCall(async (data, context) => {
  try {
    const response = await axios.post(
      "http://localhost:8000/predict-url",
      null,
      {
        params: {
          image_url: data.imageUrl,
          model_name: data.model || "yolov12m.pt",
          conf_threshold: data.confThreshold || 0.25,
        },
      },
    );

    return response.data;
  } catch (error) {
    throw new functions.https.HttpsError(
      "internal",
      "Detection failed: " + error.message,
    );
  }
});
```

---

## 🔷 C# / .NET Integration

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class YOLOv12Client
{
    private readonly string _baseUrl;
    private readonly HttpClient _httpClient;

    public YOLOv12Client(string baseUrl = "http://localhost:8000")
    {
        _baseUrl = baseUrl;
        _httpClient = new HttpClient();
    }

    public async Task<DetectionResult> PredictFromUrlAsync(
        string imageUrl,
        string modelName = "yolov12m.pt",
        float confThreshold = 0.25f,
        int imgSize = 640)
    {
        var url = $"{_baseUrl}/predict-url?" +
                  $"image_url={Uri.EscapeDataString(imageUrl)}&" +
                  $"model_name={modelName}&" +
                  $"conf_threshold={confThreshold}&" +
                  $"img_size={imgSize}";

        var response = await _httpClient.PostAsync(url, null);
        var content = await response.Content.ReadAsStringAsync();

        return JsonConvert.DeserializeObject<DetectionResult>(content);
    }
}

// Usage
var client = new YOLOv12Client();
var result = await client.PredictFromUrlAsync("https://example.com/image.jpg");
Console.WriteLine($"Found {result.detection_count} objects");
```

---

## 🎬 Flask Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
API_URL = "http://localhost:8000"

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    model = request.form.get('model', 'yolov12m.pt')

    # Forward to YOLOv12 API
    response = requests.post(
        f"{API_URL}/predict",
        files={'file': file},
        params={'model_name': model}
    )

    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🔌 Streamlit Integration

```python
import streamlit as st
import requests
from PIL import Image
import io

st.title("🎯 YOLOv12 Detection")

# Sidebar
model = st.sidebar.selectbox(
    "Select Model",
    ["yolov12n.pt", "yolov12s.pt", "yolov12m.pt", "yolov12l.pt", "yolov12x.pt"]
)
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Main
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file and st.button("Detect"):
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Send to API
    with st.spinner("Detecting..."):
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": uploaded_file},
            params={
                "model_name": model,
                "conf_threshold": confidence
            }
        )

    result = response.json()

    # Display results
    st.success(f"✓ Found {result['detection_count']} objects")

    # Display detections
    detections_df = pd.DataFrame(
        [(d['class_name'], d['confidence']) for d in result['detections']],
        columns=['Class', 'Confidence']
    )
    st.dataframe(detections_df)
```

---

## 🐳 Docker Service Integration

### docker-compose.yml (dengan multiple services)

```yaml
version: "3.8"

services:
  yolov12-api:
    build:
      context: .
      dockerfile: Dockerfile-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - yolov12-api
    environment:
      - REACT_APP_API_URL=http://yolov12-api:8000
```

---

## 🔄 Message Queue Integration (Celery)

```python
from celery import Celery
import requests

app = Celery('yolo_tasks')

@app.task
def detect_objects_async(image_url: str, model: str = 'yolov12m.pt'):
    """Async task for detection"""
    response = requests.post(
        'http://localhost:8000/predict-url',
        params={
            'image_url': image_url,
            'model_name': model
        }
    )
    return response.json()

# Usage
task = detect_objects_async.delay('https://example.com/image.jpg')
result = task.get(timeout=300)
```

---

## 📊 Data Pipeline Integration (Airflow)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from datetime import datetime

def detect_in_batch(image_urls: list):
    results = []
    for url in image_urls:
        response = requests.post(
            'http://localhost:8000/predict-url',
            params={'image_url': url}
        )
        results.append(response.json())
    return results

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

dag = DAG('yolo_detection_pipeline', default_args=default_args)

detect_task = PythonOperator(
    task_id='detect_objects',
    python_callable=detect_in_batch,
    op_kwargs={'image_urls': ['url1', 'url2']},
    dag=dag
)
```

---

## 📱 Mobile Integration (Flutter)

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

class YOLOv12Client {
  final String baseUrl;

  YOLOv12Client({this.baseUrl = 'http://localhost:8000'});

  Future<Map<String, dynamic>> predictFromUrl(String imageUrl) async {
    final response = await http.post(
      Uri.parse('$baseUrl/predict-url'),
      queryParameters: {
        'image_url': imageUrl,
        'model_name': 'yolov12m.pt',
      },
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to detect');
    }
  }
}

// Usage
final client = YOLOv12Client();
final result = await client.predictFromUrl('https://example.com/image.jpg');
print('Found ${result['detection_count']} objects');
```

---

Untuk integrasi lebih lanjut, lihat dokumentasi API di [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
