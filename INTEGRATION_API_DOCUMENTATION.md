# 📚 API Integration Documentation

## YOLOv12 Eye Disease Detection API - v2.0.0

**Deteksi Otomatis Penyakit Mata Anterior pada Citra Non-Fundus menggunakan YOLOv12 berbasis Attention dan Explainable AI**

---

## 📑 Daftar Isi

1. [Quick Start](#quick-start)
2. [Setup & Installation](#setup--installation)
3. [API Endpoints](#api-endpoints)
4. [Authentication](#authentication)
5. [Request/Response Format](#requestresponse-format)
6. [Error Handling](#error-handling)
7. [Contoh Integrasi](#contoh-integrasi)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Minimal Setup

```bash
# 1. Install dependencies
pip install -r requirements-api-v2.txt

# 2. Jalankan API
python api_v2.py

# 3. Test endpoint
curl http://localhost:8000/v1/health
```

### Cek API Status

```bash
curl -X GET "http://localhost:8000/v1/health"
```

Expected Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2024-04-20T10:30:00"
}
```

---

## 📦 Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (untuk GPU acceleration - optional)
- 8GB RAM minimum (16GB recommended)

### Step 1: Install Dependencies

```bash
# Navigasi ke folder project
cd d:\UMNFIX\yolov12

# Install requirements
pip install -r requirements-api-v2.txt
```

### Step 2: Download Models (Automatic)

Models akan di-download otomatis saat API pertama kali dijalankan:

- Model checkpoint: `runs/detect/train4/weights/best.pt`
- Model classes akan ter-load otomatis

### Step 3: Konfigurasi Environment (Optional)

Buat file `.env` untuk konfigurasi custom:

```bash
# .env
YOLO_MODEL_PATH=runs/detect/train4/weights/best.pt
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
DEVICE=cuda  # atau 'cpu'
```

### Step 4: Start API Server

```bash
# Development
python api_v2.py

# Production dengan uvicorn
uvicorn api_v2:app --host 0.0.0.0 --port 8000

# Dengan reload
uvicorn api_v2:app --host 0.0.0.0 --port 8000 --reload
```

**Output yang diharapkan:**

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Model loaded successfully on device: cuda
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Verifikasi API

Akses dokumentasi interaktif:

- **Swagger UI**: http://localhost:8000/v1/docs
- **ReDoc**: http://localhost:8000/v1/redoc

---

## 🔌 API Endpoints

### 1. Root Endpoint

```http
GET /
```

Dapatkan informasi umum API

**Response:**

```json
{
  "name": "YOLO v12 Eye Disease Detection API",
  "version": "2.0.0",
  "research": "Deteksi Otomatis Penyakit Mata Anterior...",
  "documentation": "/v1/docs",
  "health": "/v1/health",
  "endpoints": {
    "health_check": "/v1/health",
    "model_info": "/v1/model/info",
    "detect": "/v1/detect",
    "batch_detect": "/v1/detect/batch",
    "supported_formats": "/v1/supported-formats"
  }
}
```

---

### 2. Health Check

```http
GET /v1/health
```

Cek status API dan model yang ter-load

**Headers:** (None required)

**Response (200 - OK):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_info": {
    "model_path": "runs/detect/train4/weights/best.pt",
    "num_classes": 8,
    "classes": {
      "0": "Pterygium",
      "1": "Cataracts",
      "2": "Diabetic Retinopathy",
      "3": "Glaucoma",
      "4": "Age-related Macular Degeneration",
      "5": "Corneal Abrasion",
      "6": "Dry Eye",
      "7": "Keratoconus"
    }
  },
  "timestamp": "2024-04-20T10:30:00.123456"
}
```

**Response (503 - Service Unavailable):**

```json
{
  "success": false,
  "error": "Model tidak tersedia",
  "timestamp": "2024-04-20T10:30:00"
}
```

---

### 3. Model Information

```http
GET /v1/model/info
```

Dapatkan informasi detail model dan penyakit yang dapat dideteksi

**Response (200 - OK):**

```json
{
  "success": true,
  "model": {
    "name": "YOLOv12 with Attention Mechanism",
    "task": "Object Detection",
    "path": "runs/detect/train4/weights/best.pt",
    "device": "cuda",
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
  },
  "research": {
    "title": "DETEKSI OTOMATIS PENYAKIT MATA ANTERIOR PADA CITRA NON-FUNDUS MENGGUNAKAN YOLOV12 BERBASIS ATTENTION DAN EXPLAINABLE AI",
    "description": "Automatic detection of anterior eye diseases from non-fundus images using YOLOv12 with attention mechanisms",
    "framework": "YOLOv12",
    "enhancement": "CBAM (Convolutional Block Attention Module)",
    "interpretability": "Grad-CAM for explainable AI"
  },
  "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
  "api_version": "2.0.0"
}
```

---

### 4. Single Image Detection

```http
POST /v1/detect
```

Deteksi penyakit mata dari satu image

**Headers:**

```
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter    | Type    | Required | Default | Range     | Description                             |
| ------------ | ------- | -------- | ------- | --------- | --------------------------------------- |
| file         | File    | ✅       | -       | -         | Image file (JPEG, PNG, BMP, WebP, TIFF) |
| conf         | float   | ❌       | 0.25    | 0.1 - 1.0 | Confidence threshold untuk deteksi      |
| iou          | float   | ❌       | 0.45    | 0.1 - 1.0 | IOU threshold untuk NMS                 |
| return_image | boolean | ❌       | false   | -         | Return annotated image dalam base64     |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/v1/detect" \
  -H "accept: application/json" \
  -F "file=@image.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45" \
  -F "return_image=true"
```

**Response (200 - OK):**

```json
{
  "success": true,
  "message": "Deteksi selesai: 3 penyakit ditemukan",
  "timestamp": "2024-04-20T10:30:00.123456",
  "image_metadata": {
    "filename": "image.jpg",
    "size": 102400,
    "width": 640,
    "height": 480,
    "channels": 3
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "Pterygium",
      "confidence": 0.95,
      "bbox": {
        "x1": 100.5,
        "y1": 150.2,
        "x2": 300.7,
        "y2": 450.3
      },
      "bbox_normalized": {
        "x1": 0.157,
        "y1": 0.313,
        "x2": 0.47,
        "y2": 0.938
      },
      "area_pixels": 60000
    },
    {
      "class_id": 1,
      "class_name": "Cataracts",
      "confidence": 0.87,
      "bbox": {
        "x1": 150.0,
        "y1": 100.0,
        "x2": 350.0,
        "y2": 400.0
      },
      "bbox_normalized": {
        "x1": 0.234,
        "y1": 0.208,
        "x2": 0.547,
        "y2": 0.833
      },
      "area_pixels": 45000
    }
  ],
  "statistics": {
    "total_detections": 3,
    "avg_confidence": 0.901,
    "max_confidence": 0.95,
    "min_confidence": 0.82,
    "detection_density": 0.003
  },
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
}
```

**Response (400 - Bad Request):**

```json
{
  "success": false,
  "error": "File format tidak didukung",
  "timestamp": "2024-04-20T10:30:00"
}
```

**Response (413 - Payload Too Large):**

```json
{
  "success": false,
  "error": "File terlalu besar (max 50MB)",
  "timestamp": "2024-04-20T10:30:00"
}
```

**Response (503 - Service Unavailable):**

```json
{
  "success": false,
  "error": "Model tidak tersedia",
  "timestamp": "2024-04-20T10:30:00"
}
```

---

### 5. Batch Image Detection

```http
POST /v1/detect/batch
```

Deteksi penyakit pada multiple images (hingga 50 gambar)

**Headers:**

```
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type   | Required | Default | Limit     | Description          |
| --------- | ------ | -------- | ------- | --------- | -------------------- |
| files     | File[] | ✅       | -       | 50        | Multiple image files |
| conf      | float  | ❌       | 0.25    | 0.1 - 1.0 | Confidence threshold |
| iou       | float  | ❌       | 0.45    | 0.1 - 1.0 | IOU threshold        |

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/v1/detect/batch" \
  -H "accept: application/json" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

**Response (200 - OK):**

```json
{
  "success": true,
  "message": "Batch processing selesai: 3/3 berhasil",
  "timestamp": "2024-04-20T10:30:00.123456",
  "total_images": 3,
  "processed_images": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "detections": 5,
      "image_shape": {
        "width": 640,
        "height": 480
      }
    },
    {
      "filename": "image2.jpg",
      "success": true,
      "detections": 2,
      "image_shape": {
        "width": 800,
        "height": 600
      }
    },
    {
      "filename": "image3.jpg",
      "success": false,
      "error": "Format tidak didukung"
    }
  ]
}
```

**Response (413 - Too Many Files):**

```json
{
  "success": false,
  "error": "Maksimal 50 images per batch",
  "timestamp": "2024-04-20T10:30:00"
}
```

---

### 6. Supported Formats

```http
GET /v1/supported-formats
```

Dapatkan daftar format dan batasan

**Response (200 - OK):**

```json
{
  "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
  "max_file_size_mb": 50,
  "max_batch_size": 50
}
```

---

## 🔐 Authentication

API ini tidak memerlukan authentication untuk versi current. Untuk production deployment, pertimbangkan:

### Opsi 1: API Key Authentication

```python
from fastapi import Depends, HTTPException, Header

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key
```

### Opsi 2: Bearer Token

```bash
curl -X GET "http://localhost:8000/v1/health" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Opsi 3: JWT Authentication

Lihat [contoh JWT integration](#jwt-authentication-example) di bawah.

---

## 📋 Request/Response Format

### Request Format

#### 1. Single Image Upload

```python
import requests

url = "http://localhost:8000/v1/detect"
files = {"file": open("image.jpg", "rb")}
params = {
    "conf": 0.25,
    "iou": 0.45,
    "return_image": True
}

response = requests.post(url, files=files, params=params)
```

#### 2. Query Parameters

```python
# URL dengan query parameters
url = "http://localhost:8000/v1/detect/batch"
params = {
    "conf": 0.30,
    "iou": 0.50
}

response = requests.post(url, files=files, params=params)
```

### Response Format

Semua response menggunakan format JSON:

```json
{
  "success": true/false,
  "message": "string",
  "timestamp": "ISO 8601 datetime",
  "data": {}
}
```

---

## ❌ Error Handling

### HTTP Status Codes

| Code | Meaning             | Description                                   |
| ---- | ------------------- | --------------------------------------------- |
| 200  | OK                  | Request berhasil                              |
| 400  | Bad Request         | File format tidak valid, parameter salah      |
| 413  | Payload Too Large   | File terlalu besar atau terlalu banyak images |
| 500  | Server Error        | Internal server error                         |
| 503  | Service Unavailable | Model tidak ter-load                          |

### Error Response Format

```json
{
  "success": false,
  "error": "Deskripsi error",
  "timestamp": "2024-04-20T10:30:00"
}
```

### Common Errors & Solutions

#### Error: "Model tidak tersedia"

**Penyebab:** Model belum ter-load

**Solusi:**

```bash
# 1. Check model file exists
ls runs/detect/train4/weights/best.pt

# 2. Restart API
python api_v2.py

# 3. Check logs untuk error
```

#### Error: "File format tidak didukung"

**Penyebab:** Format file tidak didukung

**Solusi:**

- Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- Konversi gambar ke format yang didukung:

```python
from PIL import Image
img = Image.open("image.gif")
img.save("image.png")
```

#### Error: "File terlalu besar"

**Penyebab:** File melebihi 50MB

**Solusi:**

- Kompres gambar sebelum upload:

```python
from PIL import Image
img = Image.open("large_image.jpg")
img.thumbnail((1920, 1080))
img.save("compressed_image.jpg", quality=85)
```

#### Error: "Maksimal 50 images per batch"

**Penyebab:** Terlalu banyak gambar dalam batch

**Solusi:**

- Bagi menjadi beberapa batch, masing-masing max 50 images

---

## 💻 Contoh Integrasi

### Python - Requests Library

```python
import requests
import json
from pathlib import Path

class EyeDiseaseDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def health_check(self):
        """Check API health"""
        url = f"{self.base_url}/v1/health"
        response = requests.get(url)
        return response.json()

    def get_model_info(self):
        """Get model information and available classes"""
        url = f"{self.base_url}/v1/model/info"
        response = requests.get(url)
        return response.json()

    def detect_single_image(self, image_path, conf=0.25, iou=0.45, return_image=True):
        """Detect diseases in single image"""
        url = f"{self.base_url}/v1/detect"

        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            params = {
                "conf": conf,
                "iou": iou,
                "return_image": return_image
            }

            response = requests.post(url, files=files, params=params)
            return response.json()

    def detect_batch(self, image_paths, conf=0.25, iou=0.45):
        """Detect diseases in multiple images"""
        url = f"{self.base_url}/v1/detect/batch"

        files = []
        for path in image_paths:
            with open(path, "rb") as img_file:
                files.append(("files", img_file.read()))

        params = {
            "conf": conf,
            "iou": iou
        }

        response = requests.post(url, files=files, params=params)
        return response.json()

    def get_supported_formats(self):
        """Get supported image formats"""
        url = f"{self.base_url}/v1/supported-formats"
        response = requests.get(url)
        return response.json()

# Usage Example
if __name__ == "__main__":
    client = EyeDiseaseDetectionClient()

    # 1. Check health
    print("🏥 Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))

    # 2. Get model info
    print("\n📋 Model Information:")
    model_info = client.get_model_info()
    print(f"Model: {model_info['model']['name']}")
    print(f"Classes: {model_info['model']['num_classes']}")

    # 3. Single image detection
    print("\n🔍 Single Image Detection:")
    result = client.detect_single_image("test_image.jpg", return_image=True)
    print(f"Detections: {len(result['detections'])}")
    for det in result['detections']:
        print(f"  • {det['class_name']}: {det['confidence']:.1%}")

    # 4. Batch processing
    print("\n📦 Batch Processing:")
    batch_result = client.detect_batch(["image1.jpg", "image2.jpg"])
    print(f"Processed: {batch_result['processed_images']}/{batch_result['total_images']}")
```

---

### JavaScript/Node.js - Axios

```javascript
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

class EyeDiseaseDetectionClient {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
    this.client = axios.create({ baseURL: baseUrl });
  }

  async healthCheck() {
    try {
      const response = await this.client.get("/v1/health");
      return response.data;
    } catch (error) {
      console.error("Health check failed:", error.message);
      throw error;
    }
  }

  async getModelInfo() {
    try {
      const response = await this.client.get("/v1/model/info");
      return response.data;
    } catch (error) {
      console.error("Get model info failed:", error.message);
      throw error;
    }
  }

  async detectSingleImage(
    imagePath,
    conf = 0.25,
    iou = 0.45,
    returnImage = true,
  ) {
    try {
      const formData = new FormData();
      formData.append("file", fs.createReadStream(imagePath));
      formData.append("conf", conf);
      formData.append("iou", iou);
      formData.append("return_image", returnImage);

      const response = await this.client.post("/v1/detect", formData, {
        headers: formData.getHeaders(),
      });

      return response.data;
    } catch (error) {
      console.error("Detection failed:", error.message);
      throw error;
    }
  }

  async detectBatch(imagePaths, conf = 0.25, iou = 0.45) {
    try {
      const formData = new FormData();

      imagePaths.forEach((imagePath) => {
        formData.append("files", fs.createReadStream(imagePath));
      });

      formData.append("conf", conf);
      formData.append("iou", iou);

      const response = await this.client.post("/v1/detect/batch", formData, {
        headers: formData.getHeaders(),
      });

      return response.data;
    } catch (error) {
      console.error("Batch detection failed:", error.message);
      throw error;
    }
  }

  async getSupportedFormats() {
    try {
      const response = await this.client.get("/v1/supported-formats");
      return response.data;
    } catch (error) {
      console.error("Get supported formats failed:", error.message);
      throw error;
    }
  }
}

// Usage Example
(async () => {
  const client = new EyeDiseaseDetectionClient();

  try {
    // 1. Health check
    console.log("🏥 Health Check:");
    const health = await client.healthCheck();
    console.log(JSON.stringify(health, null, 2));

    // 2. Get model info
    console.log("\n📋 Model Information:");
    const modelInfo = await client.getModelInfo();
    console.log(`Model: ${modelInfo.model.name}`);
    console.log(`Classes: ${modelInfo.model.num_classes}`);

    // 3. Single image detection
    console.log("\n🔍 Single Image Detection:");
    const result = await client.detectSingleImage(
      "test_image.jpg",
      0.25,
      0.45,
      true,
    );
    console.log(`Detections: ${result.detections.length}`);
    result.detections.forEach((det) => {
      console.log(
        `  • ${det.class_name}: ${(det.confidence * 100).toFixed(1)}%`,
      );
    });

    // 4. Batch processing
    console.log("\n📦 Batch Processing:");
    const batchResult = await client.detectBatch(["image1.jpg", "image2.jpg"]);
    console.log(
      `Processed: ${batchResult.processed_images}/${batchResult.total_images}`,
    );
  } catch (error) {
    console.error("Error:", error);
  }
})();
```

---

### cURL - Command Line

```bash
#!/bin/bash

API_URL="http://localhost:8000"

# 1. Health check
echo "🏥 Health Check"
curl -X GET "$API_URL/v1/health"

# 2. Model info
echo -e "\n📋 Model Information"
curl -X GET "$API_URL/v1/model/info"

# 3. Single image detection
echo -e "\n🔍 Single Image Detection"
curl -X POST "$API_URL/v1/detect" \
  -H "accept: application/json" \
  -F "file=@test_image.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45" \
  -F "return_image=true"

# 4. Batch processing
echo -e "\n📦 Batch Processing"
curl -X POST "$API_URL/v1/detect/batch" \
  -H "accept: application/json" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "conf=0.25"

# 5. Supported formats
echo -e "\n📋 Supported Formats"
curl -X GET "$API_URL/v1/supported-formats"
```

---

### JavaScript - Fetch API

```javascript
const API_URL = "http://localhost:8000";

class EyeDiseaseAPI {
  async healthCheck() {
    const response = await fetch(`${API_URL}/v1/health`);
    return response.json();
  }

  async getModelInfo() {
    const response = await fetch(`${API_URL}/v1/model/info`);
    return response.json();
  }

  async detectImage(imageFile, conf = 0.25, iou = 0.45) {
    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("conf", conf);
    formData.append("iou", iou);
    formData.append("return_image", true);

    const response = await fetch(`${API_URL}/v1/detect`, {
      method: "POST",
      body: formData,
    });

    return response.json();
  }

  async detectBatch(imageFiles, conf = 0.25) {
    const formData = new FormData();
    imageFiles.forEach((file) => {
      formData.append("files", file);
    });
    formData.append("conf", conf);

    const response = await fetch(`${API_URL}/v1/detect/batch`, {
      method: "POST",
      body: formData,
    });

    return response.json();
  }
}

// Usage in React/Vue
const api = new EyeDiseaseAPI();

// Example: Detect image from file input
document.getElementById("imageInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  const result = await api.detectImage(file);
  console.log("Detections:", result.detections);
});
```

---

### Java - HTTP Client

```java
import java.io.File;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Paths;
import com.google.gson.Gson;

public class EyeDiseaseDetectionClient {
    private final String baseUrl;
    private final HttpClient httpClient;
    private final Gson gson;

    public EyeDiseaseDetectionClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.httpClient = HttpClient.newHttpClient();
        this.gson = new Gson();
    }

    public String healthCheck() throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(new java.net.URI(baseUrl + "/v1/health"))
            .GET()
            .build();

        HttpResponse<String> response = httpClient.send(request,
            HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public String getModelInfo() throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(new java.net.URI(baseUrl + "/v1/model/info"))
            .GET()
            .build();

        HttpResponse<String> response = httpClient.send(request,
            HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public String detectImage(String imagePath, float conf, float iou) throws Exception {
        byte[] fileContent = Files.readAllBytes(Paths.get(imagePath));

        HttpRequest request = HttpRequest.newBuilder()
            .uri(new java.net.URI(baseUrl + "/v1/detect?conf=" + conf + "&iou=" + iou))
            .POST(HttpRequest.BodyPublishers.ofByteArray(fileContent))
            .header("Content-Type", "image/jpeg")
            .build();

        HttpResponse<String> response = httpClient.send(request,
            HttpResponse.BodyHandlers.ofString());
        return response.body();
    }
}

// Usage
public class Main {
    public static void main(String[] args) throws Exception {
        EyeDiseaseDetectionClient client =
            new EyeDiseaseDetectionClient("http://localhost:8000");

        String health = client.healthCheck();
        System.out.println(health);
    }
}
```

---

## ✅ Best Practices

### 1. Error Handling

```python
import requests
from requests.exceptions import RequestException, Timeout

def detect_with_retry(image_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/v1/detect",
                files={"file": open(image_path, "rb")},
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except Timeout:
            print(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                raise

        except RequestException as e:
            print(f"Request error: {e}")
            if attempt == max_retries - 1:
                raise
```

### 2. Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        backoff_factor=1
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

session = create_session()
response = session.post("http://localhost:8000/v1/detect", files=files)
```

### 3. Image Compression

```python
from PIL import Image
import io

def compress_image(image_path, max_width=1920, quality=85):
    """Compress image before sending"""
    img = Image.open(image_path)

    # Resize if too large
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)

    return buffer

# Usage
compressed = compress_image("large_image.jpg")
response = requests.post(
    "http://localhost:8000/v1/detect",
    files={"file": compressed}
)
```

### 4. Performance Optimization

```python
import concurrent.futures
from pathlib import Path

def detect_images_parallel(image_dir, num_workers=4):
    """Process multiple images in parallel"""
    images = list(Path(image_dir).glob("*.jpg"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(detect_single_image, img): img
            for img in images
        }

        results = {}
        for future in concurrent.futures.as_completed(futures):
            img = futures[future]
            try:
                results[img.name] = future.result()
            except Exception as e:
                print(f"Error processing {img.name}: {e}")

    return results
```

### 5. Logging

```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_client.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def detect_with_logging(image_path):
    logger.info(f"Starting detection for {image_path}")

    try:
        response = requests.post(
            "http://localhost:8000/v1/detect",
            files={"file": open(image_path, "rb")}
        )

        logger.info(f"Detection successful. Detections: {len(response.json()['detections'])}")
        return response.json()

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise
```

---

## 🔧 Troubleshooting

### Issue: Connection Refused

```bash
# Check if server is running
curl http://localhost:8000/

# If not, start the server
python api_v2.py

# Check port availability
netstat -an | grep 8000
```

### Issue: Slow Detection

```python
# 1. Check current confidence threshold
# Try with higher threshold
response = requests.post(
    "http://localhost:8000/v1/detect",
    files=files,
    params={"conf": 0.5}  # Higher threshold = faster
)

# 2. Use smaller batch size
# Instead of 50, use 10-20 images per batch

# 3. Monitor GPU usage
# nvidia-smi
```

### Issue: Out of Memory

```bash
# Run with CPU instead of GPU
DEVICE=cpu python api_v2.py

# Or reduce batch size
# Modify MAX_BATCH_SIZE in api_v2.py
```

### Issue: Model Not Found

```bash
# Check model path
ls -la runs/detect/train4/weights/

# If missing, the model will auto-download
# Make sure internet connection is available

# Manually download if needed
python -c "from ultralytics import YOLO; YOLO('best.pt')"
```

---

## 📞 Support & Resources

### Documentation

- **Interactive Docs**: http://localhost:8000/v1/docs
- **API Specification**: http://localhost:8000/v1/openapi.json

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Check memory
free -h

# Check CPU usage
top
```

---

## 📝 Version History

### v2.0.0 (Current)

- ✅ Single image detection
- ✅ Batch processing (up to 50 images)
- ✅ Confidence & IOU thresholds
- ✅ Annotated image output
- ✅ Comprehensive error handling
- ✅ CBAM attention mechanism
- ✅ Explainable AI support

### v1.0.0

- Initial release with basic detection

---

## 📄 License

This API is part of the research project:
"DETEKSI OTOMATIS PENYAKIT MATA ANTERIOR PADA CITRA NON-FUNDUS MENGGUNAKAN YOLOV12 BERBASIS ATTENTION DAN EXPLAINABLE AI"

---

**Last Updated**: April 20, 2024 | **API Version**: 2.0.0
