# 🔨 API Integration Examples

## YOLOv12 Eye Disease Detection API - Practical Examples

---

## 📂 Daftar Contoh

1. [Python - Basic Usage](#python---basic-usage)
2. [Python - Advanced Integration](#python---advanced-integration)
3. [Web Application Integration](#web-application-integration)
4. [Mobile Integration](#mobile-integration)
5. [Desktop Application](#desktop-application)
6. [Docker Integration](#docker-integration)
7. [CI/CD Pipeline](#cicd-pipeline)

---

## 🐍 Python - Basic Usage

### Simple Detection Script

**File: `detect_simple.py`**

```python
import requests
import json

# Configuration
API_URL = "http://localhost:8000"
IMAGE_PATH = "test_image.jpg"

def main():
    # Check API health
    health_response = requests.get(f"{API_URL}/v1/health")
    if health_response.status_code != 200:
        print("❌ API tidak tersedia")
        return

    print("✅ API healthy")

    # Load and detect
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": f}
        params = {
            "conf": 0.25,
            "iou": 0.45,
            "return_image": False
        }

        response = requests.post(f"{API_URL}/v1/detect", files=files, params=params)

    if response.status_code == 200:
        result = response.json()
        print(f"\n🔍 Deteksi Hasil:")
        print(f"Total Deteksi: {len(result['detections'])}")

        for i, det in enumerate(result['detections'], 1):
            print(f"\n  {i}. {det['class_name']}")
            print(f"     Confidence: {det['confidence']:.2%}")
            print(f"     Lokasi: ({det['bbox']['x1']:.0f}, {det['bbox']['y1']:.0f})")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    main()
```

### Batch Processing Script

**File: `detect_batch.py`**

```python
import requests
from pathlib import Path
import json

API_URL = "http://localhost:8000"
IMAGE_DIR = "images"

def detect_batch(image_dir, conf=0.25):
    """Detect diseases in batch of images"""

    # Get all image files
    image_paths = list(Path(image_dir).glob("*.jpg"))
    if not image_paths:
        print(f"❌ Tidak ada gambar ditemukan di {image_dir}")
        return

    print(f"📦 Memproses {len(image_paths)} gambar...")

    # Prepare files
    files = []
    for path in image_paths:
        files.append(("files", open(path, "rb")))

    # Send batch request
    response = requests.post(
        f"{API_URL}/v1/detect/batch",
        files=files,
        params={"conf": conf}
    )

    if response.status_code == 200:
        result = response.json()

        print(f"\n✅ Selesai!")
        print(f"Total: {result['total_images']}")
        print(f"Berhasil: {result['processed_images']}")

        # Print results
        print("\n📋 Detail Hasil:")
        for item in result['results']:
            if item['success']:
                print(f"  ✅ {item['filename']}: {item['detections']} deteksi")
            else:
                print(f"  ❌ {item['filename']}: {item.get('error', 'Unknown error')}")
    else:
        print(f"❌ Error: {response.status_code}")

if __name__ == "__main__":
    detect_batch(IMAGE_DIR)
```

---

## 🐍 Python - Advanced Integration

### Production-Ready Client

**File: `api_client.py`**

```python
import requests
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EyeDiseaseAPI:
    """Production-ready client untuk Eye Disease Detection API"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self):
        """Create requests session dengan retry strategy"""
        session = requests.Session()

        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

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

    def health_check(self) -> bool:
        """Check API health"""
        try:
            response = self.session.get(
                f"{self.base_url}/v1/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("API health check passed")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            response = self.session.get(
                f"{self.base_url}/v1/model/info",
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("Model info retrieved successfully")
            return response.json()
        except Exception as e:
            logger.error(f"Get model info failed: {e}")
            raise

    def detect_image(self, image_path: str, conf: float = 0.25,
                    iou: float = 0.45, return_image: bool = False) -> Dict:
        """Detect disease in single image"""

        try:
            # Validate image file
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            logger.info(f"Processing image: {image_path}")

            with open(image_path, "rb") as f:
                files = {"file": f}
                params = {
                    "conf": conf,
                    "iou": iou,
                    "return_image": return_image
                }

                response = self.session.post(
                    f"{self.base_url}/v1/detect",
                    files=files,
                    params=params,
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()

                logger.info(f"Detection completed. Found {len(result['detections'])} diseases")
                return result

        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            raise

    def detect_batch(self, image_paths: List[str], conf: float = 0.25,
                    iou: float = 0.45) -> Dict:
        """Detect disease in multiple images"""

        try:
            # Validate images
            image_list = []
            for path in image_paths:
                img_file = Path(path)
                if not img_file.exists():
                    logger.warning(f"Image not found, skipping: {path}")
                    continue
                image_list.append(img_file)

            if not image_list:
                raise ValueError("No valid images provided")

            if len(image_list) > 50:
                logger.warning("Batch size exceeds 50, splitting...")
                results = []
                for i in range(0, len(image_list), 50):
                    batch = [str(p) for p in image_list[i:i+50]]
                    results.append(self.detect_batch(batch, conf, iou))
                return self._merge_batch_results(results)

            logger.info(f"Processing batch of {len(image_list)} images")

            files = []
            for path in image_list:
                files.append(("files", open(str(path), "rb")))

            params = {
                "conf": conf,
                "iou": iou
            }

            response = self.session.post(
                f"{self.base_url}/v1/detect/batch",
                files=files,
                params=params,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            logger.info(f"Batch processing completed. Processed: {result['processed_images']}/{result['total_images']}")
            return result

        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            raise

    def _merge_batch_results(self, batch_results: List[Dict]) -> Dict:
        """Merge multiple batch results"""
        merged = {
            "success": True,
            "message": "Merged batch results",
            "total_images": 0,
            "processed_images": 0,
            "results": []
        }

        for result in batch_results:
            merged["total_images"] += result["total_images"]
            merged["processed_images"] += result["processed_images"]
            merged["results"].extend(result["results"])

        return merged

    def save_results(self, detections: Dict, output_path: str):
        """Save detection results to JSON"""
        try:
            with open(output_path, "w") as f:
                json.dump(detections, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

# Usage Example
if __name__ == "__main__":
    client = EyeDiseaseAPI()

    # Check health
    if not client.health_check():
        print("API is not available")
        exit(1)

    # Get model info
    model_info = client.get_model_info()
    print(f"Model: {model_info['model']['name']}")
    print(f"Classes: {list(model_info['model']['classes'].values())}")

    # Single image
    result = client.detect_image("test_image.jpg")
    print(f"\nDetections: {len(result['detections'])}")

    # Batch processing
    images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    batch_result = client.detect_batch(images)
    print(f"Batch result: {batch_result['processed_images']}/{batch_result['total_images']}")

    # Save results
    client.save_results(result, "detections.json")
```

---

## 🌐 Web Application Integration

### Flask Application

**File: `flask_app.py`**

```python
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
import os
from pathlib import Path
import base64
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'uploads'

API_BASE_URL = "http://localhost:8000"

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/v1/health")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/v1/model/info")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def detect():
    """Detect disease from uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get parameters
        conf = float(request.form.get('conf', 0.25))
        iou = float(request.form.get('iou', 0.45))
        return_image = request.form.get('return_image', 'true').lower() == 'true'

        # Send to detection API
        with open(filepath, "rb") as f:
            files = {"file": f}
            params = {
                "conf": conf,
                "iou": iou,
                "return_image": return_image
            }

            response = requests.post(
                f"{API_BASE_URL}/v1/detect",
                files=files,
                params=params
            )

        # Clean up
        os.remove(filepath)

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect-batch', methods=['POST'])
def detect_batch():
    """Batch detection"""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files selected"}), 400

        conf = float(request.form.get('conf', 0.25))

        # Prepare multipart data
        file_data = []
        temp_paths = []

        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                temp_paths.append(filepath)
                file_data.append(("files", open(filepath, "rb")))

        # Send batch request
        response = requests.post(
            f"{API_BASE_URL}/v1/detect/batch",
            files=file_data,
            params={"conf": conf}
        )

        # Clean up
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**HTML Template: `templates/index.html`**

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Eye Disease Detection</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
      }

      .container {
        background: #f5f5f5;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
      }

      .section {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
      }

      button {
        background: #007bff;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
      }

      button:hover {
        background: #0056b3;
      }

      .results {
        background: #e8f5e9;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
      }

      .error {
        background: #ffebee;
        padding: 15px;
        border-radius: 4px;
        color: #c62828;
      }

      input[type="file"] {
        margin-bottom: 10px;
      }

      .detection-item {
        background: #f9f9f9;
        padding: 10px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
      }
    </style>
  </head>
  <body>
    <h1>🏥 Eye Disease Detection API</h1>

    <div class="section">
      <h2>API Status</h2>
      <button onclick="checkHealth()">Check API Health</button>
      <div id="healthStatus"></div>
    </div>

    <div class="section">
      <h2>Single Image Detection</h2>
      <input type="file" id="imageInput" accept="image/*" />
      <div>
        <label>Confidence Threshold (0-1):</label>
        <input
          type="number"
          id="conf"
          value="0.25"
          min="0.1"
          max="1.0"
          step="0.05"
        />
      </div>
      <button onclick="detectImage()">Detect</button>
      <div id="singleResult"></div>
    </div>

    <div class="section">
      <h2>Batch Detection</h2>
      <input type="file" id="batchInput" accept="image/*" multiple />
      <button onclick="detectBatch()">Detect Batch</button>
      <div id="batchResult"></div>
    </div>

    <script>
      const API_BASE = "http://localhost:8000";

      async function checkHealth() {
        try {
          const response = await fetch(`${API_BASE}/v1/health`);
          const data = await response.json();

          const html = `
                    <div class="results">
                        <p>Status: ${data.status}</p>
                        <p>Model Loaded: ${data.model_loaded ? "✅" : "❌"}</p>
                        <p>Device: ${data.device}</p>
                    </div>
                `;
          document.getElementById("healthStatus").innerHTML = html;
        } catch (error) {
          document.getElementById("healthStatus").innerHTML =
            `<div class="error">Error: ${error.message}</div>`;
        }
      }

      async function detectImage() {
        const file = document.getElementById("imageInput").files[0];
        if (!file) {
          alert("Please select an image");
          return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("conf", document.getElementById("conf").value);
        formData.append("return_image", "true");

        try {
          const response = await fetch(`${API_BASE}/v1/detect`, {
            method: "POST",
            body: formData,
          });

          const data = await response.json();

          let html = `<div class="results">
                    <p><strong>Detections: ${data.detections.length}</strong></p>`;

          data.detections.forEach((det) => {
            html += `
                        <div class="detection-item">
                            <strong>${det.class_name}</strong><br>
                            Confidence: ${(det.confidence * 100).toFixed(1)}%<br>
                            Location: (${det.bbox.x1.toFixed(0)}, ${det.bbox.y1.toFixed(0)})
                        </div>
                    `;
          });

          html += "</div>";
          document.getElementById("singleResult").innerHTML = html;
        } catch (error) {
          document.getElementById("singleResult").innerHTML =
            `<div class="error">Error: ${error.message}</div>`;
        }
      }

      async function detectBatch() {
        const files = document.getElementById("batchInput").files;
        if (files.length === 0) {
          alert("Please select images");
          return;
        }

        const formData = new FormData();
        for (let file of files) {
          formData.append("files", file);
        }

        try {
          const response = await fetch(`${API_BASE}/v1/detect/batch`, {
            method: "POST",
            body: formData,
          });

          const data = await response.json();

          let html = `<div class="results">
                    <p><strong>Processed: ${data.processed_images}/${data.total_images}</strong></p>`;

          data.results.forEach((item) => {
            if (item.success) {
              html += `<div class="detection-item">✅ ${item.filename}: ${item.detections} detections</div>`;
            } else {
              html += `<div class="detection-item" style="border-color: red;">❌ ${item.filename}: ${item.error}</div>`;
            }
          });

          html += "</div>";
          document.getElementById("batchResult").innerHTML = html;
        } catch (error) {
          document.getElementById("batchResult").innerHTML =
            `<div class="error">Error: ${error.message}</div>`;
        }
      }
    </script>
  </body>
</html>
```

---

## 📱 Mobile Integration

### React Native Example

**File: `MobileApp.tsx`**

```typescript
import React, { useState } from 'react';
import {
  View,
  Text,
  Button,
  Image,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';

const API_BASE = 'http://192.168.1.100:8000';

interface Detection {
  class_name: string;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

interface DetectionResult {
  success: boolean;
  detections: Detection[];
  message: string;
}

export default function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const detectDisease = async () => {
    if (!selectedImage) return;

    setLoading(true);
    try {
      const fileInfo = await FileSystem.getInfoAsync(selectedImage);
      const base64 = await FileSystem.readAsStringAsync(selectedImage, {
        encoding: FileSystem.EncodingType.Base64,
      });

      const formData = new FormData();
      formData.append('file', {
        uri: selectedImage,
        name: 'photo.jpg',
        type: 'image/jpeg',
      } as any);

      const response = await fetch(`${API_BASE}/v1/detect`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Detection error:', error);
      alert('Error detecting disease');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Eye Disease Detection</Text>

      <Button title="Pick Image" onPress={pickImage} />

      {selectedImage && (
        <>
          <Image source={{ uri: selectedImage }} style={styles.image} />
          <Button title="Detect Disease" onPress={detectDisease} />
        </>
      )}

      {loading && <ActivityIndicator size="large" />}

      {result && (
        <View style={styles.results}>
          <Text style={styles.resultTitle}>
            Detections: {result.detections.length}
          </Text>

          {result.detections.map((det, idx) => (
            <View key={idx} style={styles.detectionItem}>
              <Text style={styles.className}>{det.class_name}</Text>
              <Text>Confidence: {(det.confidence * 100).toFixed(1)}%</Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  image: {
    width: '100%',
    height: 300,
    resizeMode: 'contain',
    marginVertical: 20,
  },
  results: {
    marginTop: 20,
    padding: 15,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  detectionItem: {
    padding: 10,
    backgroundColor: '#fff',
    marginBottom: 10,
    borderRadius: 5,
    borderLeftWidth: 3,
    borderLeftColor: '#007bff',
  },
  className: {
    fontSize: 16,
    fontWeight: 'bold',
  },
});
```

---

## 🖥️ Desktop Application

### Electron + Python Backend

**File: `main.js` (Electron)**

```javascript
const { app, BrowserWindow, Menu, ipcMain, dialog } = require("electron");
const path = require("path");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

const API_BASE = "http://localhost:8000";

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile("src/index.html");
}

app.on("ready", createWindow);

// IPC Handlers
ipcMain.handle("check-health", async () => {
  try {
    const response = await axios.get(`${API_BASE}/v1/health`);
    return response.data;
  } catch (error) {
    throw new Error("API not available");
  }
});

ipcMain.handle("detect-image", async (event, imagePath, conf, iou) => {
  try {
    const formData = new FormData();
    formData.append("file", fs.createReadStream(imagePath));
    formData.append("conf", conf);
    formData.append("iou", iou);
    formData.append("return_image", true);

    const response = await axios.post(`${API_BASE}/v1/detect`, formData, {
      headers: formData.getHeaders(),
    });

    return response.data;
  } catch (error) {
    throw error;
  }
});

ipcMain.handle("select-image", async () => {
  const { filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile"],
    filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png", "bmp"] }],
  });

  return filePaths[0];
});
```

---

## 🐳 Docker Integration

### Docker Compose

**File: `docker-compose.yml`**

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
      - YOLO_MODEL_PATH=/app/runs/detect/train4/weights/best.pt
    volumes:
      - ./runs:/app/runs
    restart: unless-stopped

  web:
    build:
      context: .
      dockerfile: Dockerfile.web
    ports:
      - "5000:5000"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000
    restart: unless-stopped

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
      - web
    restart: unless-stopped
```

**File: `Dockerfile`**

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3-pip

# Copy files
COPY requirements-api-v2.txt .
RUN pip install --no-cache-dir -r requirements-api-v2.txt

COPY . .

EXPOSE 8000

CMD ["python", "api_v2.py"]
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions

**File: `.github/workflows/api-tests.yml`**

```yaml
name: API Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          pip install -r requirements-api-v2.txt
          pip install pytest requests

      - name: Start API
        run: python api_v2.py &

      - name: Wait for API
        run: sleep 10

      - name: Run tests
        run: pytest tests/

      - name: Health check
        run: |
          curl -X GET "http://localhost:8000/v1/health"

  build:
    needs: test
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: docker build -t eye-disease-api:latest .
```

**File: `tests/test_api.py`**

```python
import requests
import pytest

API_URL = "http://localhost:8000"

class TestAPI:
    def test_health_check(self):
        response = requests.get(f"{API_URL}/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_model_info(self):
        response = requests.get(f"{API_URL}/v1/model/info")
        assert response.status_code == 200
        assert "model" in response.json()

    def test_supported_formats(self):
        response = requests.get(f"{API_URL}/v1/supported-formats")
        assert response.status_code == 200
        assert "supported_formats" in response.json()
```

---

## 📊 Monitoring & Analytics

### Prometheus Metrics

**File: `metrics.py`**

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# Metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])
detection_count = Counter('detections_total', 'Total detections')

def track_request(endpoint):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                request_count.labels(method='POST', endpoint=endpoint).inc()
                request_duration.labels(endpoint=endpoint).observe(duration)
        return wrapper
    return decorator

# Start metrics server
start_http_server(8001)
```

---

**Last Updated**: April 20, 2024
