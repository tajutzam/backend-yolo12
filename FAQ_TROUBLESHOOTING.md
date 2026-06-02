# 🆘 FAQ & Troubleshooting Guide

## YOLOv12 Eye Disease Detection API - v2.0.0

---

## ❓ Frequently Asked Questions

### 1. Bagaimana cara memulai API?

**Q: Saya baru pertama kali, langkah apa yang harus saya lakukan?**

A: Ikuti langkah ini:

```bash
# 1. Navigasi ke folder project
cd d:\UMNFIX\yolov12

# 2. Install dependencies (jika belum)
pip install -r requirements-api-v2.txt

# 3. Jalankan API
python api_v2.py

# 4. Cek apakah API berjalan
curl http://localhost:8000/v1/health
```

Jika output menunjukkan `"status": "healthy"`, API Anda sudah siap!

---

### 2. Format gambar apa saja yang didukung?

**Q: Gambar dalam format apa saja yang bisa dideteksi?**

A: API mendukung format berikut:

- ✅ JPEG (.jpg, .jpeg)
- ✅ PNG (.png)
- ✅ BMP (.bmp)
- ✅ TIFF (.tiff)
- ✅ WebP (.webp)

Ukuran maksimal file: **50 MB**

Jika gambar Anda dalam format lain, konversi terlebih dahulu:

```python
from PIL import Image

# Konversi GIF ke PNG
img = Image.open("image.gif")
img.save("image.png")

# Konversi dengan kompresi
img = Image.open("large_image.jpg")
img.save("compressed.jpg", quality=85, optimize=True)
```

---

### 3. Perbedaan antara confidence dan IOU threshold?

**Q: Apa perbedaan parameter `conf` dan `iou`?**

A:

| Parameter | Fungsi                                                                         | Range     | Default | Contoh                                      |
| --------- | ------------------------------------------------------------------------------ | --------- | ------- | ------------------------------------------- |
| `conf`    | Confidence Threshold - Minimum confidence score untuk menganggap deteksi valid | 0.1 - 1.0 | 0.25    | 0.5 = hanya deteksi dengan confidence ≥ 50% |
| `iou`     | IOU Threshold - Untuk Non-Maximum Suppression (menghapus duplicate detection)  | 0.1 - 1.0 | 0.45    | 0.7 = deteksi yang overlap > 70% dihapus    |

**Saran:**

- `conf=0.25, iou=0.45` - Default (balanced)
- `conf=0.5, iou=0.5` - Lebih strict (fewer false positives)
- `conf=0.15, iou=0.3` - Lebih sensitive (lebih banyak deteksi)

---

### 4. Bagaimana cara menggunakan API dari bahasa pemrograman lain?

**Q: Saya menggunakan Java/C#/Go, bagaimana cara mengakses API?**

A: Semua bahasa pemrograman bisa mengakses API melalui HTTP requests. Contoh untuk berbagai bahasa:

**Java:**

```java
import java.net.http.*;

var client = HttpClient.newHttpClient();
var request = HttpRequest.newBuilder()
    .uri(URI.create("http://localhost:8000/v1/health"))
    .GET()
    .build();

var response = client.send(request, HttpResponse.BodyHandlers.ofString());
System.out.println(response.body());
```

**C# (.NET):**

```csharp
using (var client = new HttpClient())
{
    var response = await client.GetAsync("http://localhost:8000/v1/health");
    var content = await response.Content.ReadAsStringAsync();
    Console.WriteLine(content);
}
```

**Go:**

```go
package main

import (
    "fmt"
    "net/http"
    "io/ioutil"
)

func main() {
    resp, _ := http.Get("http://localhost:8000/v1/health")
    defer resp.Body.Close()
    body, _ := ioutil.ReadAll(resp.Body)
    fmt.Println(string(body))
}
```

---

### 5. Bagaimana cara menggunakan API dari frontend (JavaScript)?

**Q: Saya ingin memanggil API dari web browser, apa yang harus diperhatikan?**

A: Pastikan API mengizinkan CORS. Caranya:

```python
# Sudah dikonfigurasi di api_v2.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua origin (untuk development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Contoh JavaScript:

```javascript
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch("http://localhost:8000/v1/detect", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result);
```

---

### 6. Penyakit mata apa saja yang bisa dideteksi?

**Q: Model dapat mendeteksi penyakit apa saja?**

A: Saat ini model dapat mendeteksi 8 penyakit anterior mata:

1. **Pterygium** - Jaringan fibrovaskular yang tumbuh ke kornea
2. **Cataracts** - Kekeruhan pada lensa mata
3. **Diabetic Retinopathy** - Komplikasi diabetes pada retina
4. **Glaucoma** - Peningkatan tekanan intraokular
5. **AMD (Age-related Macular Degeneration)** - Degenerasi makula
6. **Corneal Abrasion** - Gesekan pada kornea
7. **Dry Eye** - Mata kering
8. **Keratoconus** - Kornea berbentuk kerucut

Untuk mendapatkan daftar lengkap dari API:

```bash
curl http://localhost:8000/v1/model/info | grep -A 20 "classes"
```

---

### 7. Berapa lama waktu deteksi?

**Q: Berapa lama detection process?**

A: Tergantung beberapa faktor:

| Factor               | Impact                                         | Typical Time                            |
| -------------------- | ---------------------------------------------- | --------------------------------------- |
| GPU                  | GPU ≈ 10x lebih cepat dari CPU                 | GPU: 0.1-0.5s, CPU: 1-5s                |
| Ukuran image         | Image lebih besar = lebih lambat               | 640x480: 0.2s, 1920x1080: 1s            |
| Model                | Model lebih besar = lebih lambat               | YOLOv12n: 0.1s, YOLOv12x: 0.5s          |
| Number of detections | Lebih banyak detection = processing lebih lama | 1 detection: 0.1s, 10+ detections: 0.3s |

**Untuk performa optimal:**

```python
# 1. Gunakan GPU
# Pastikan CUDA terinstall:
# nvidia-smi

# 2. Resize image sebelum upload (max 1920x1080)
from PIL import Image
img = Image.open("image.jpg")
img.thumbnail((1920, 1080))
img.save("resized.jpg")

# 3. Tingkatkan confidence threshold
# Confidence 0.5 lebih cepat dari 0.25
```

---

### 8. Bagaimana cara deploy ke production?

**Q: Bagaimana cara deploy API ke server production?**

A: Ada beberapa pilihan:

**Opsi 1: Uvicorn dengan Gunicorn**

```bash
# Install
pip install gunicorn

# Run dengan 4 worker processes
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 api_v2:app
```

**Opsi 2: Docker**

```bash
# Build
docker build -t eye-disease-api .

# Run
docker run -p 8000:8000 eye-disease-api
```

**Opsi 3: Sistem Service (Linux)**

```bash
# Create service file
sudo nano /etc/systemd/system/eye-disease-api.service

# Content:
[Unit]
Description=Eye Disease Detection API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/yolov12
ExecStart=/usr/bin/python3 /home/ubuntu/yolov12/api_v2.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable & start
sudo systemctl enable eye-disease-api
sudo systemctl start eye-disease-api
```

---

## 🐛 Troubleshooting

### Problem 1: "Connection refused" / "Cannot connect to localhost:8000"

**Error:**

```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8000):
Max retries exceeded
```

**Penyebab Umum:**

1. API belum dijalankan
2. API berjalan di port berbeda
3. Firewall memblokir port

**Solusi:**

```bash
# 1. Check apakah API berjalan
ps aux | grep api_v2

# 2. Check port 8000 apakah sedang digunakan
netstat -an | grep 8000

# 3. Jika port sudah digunakan, jalankan di port berbeda
python api_v2.py --port 8001

# 4. Test dengan curl
curl http://localhost:8000/v1/health

# 5. Jika menggunakan Windows, coba localhost
# Bukan 127.0.0.1
```

---

### Problem 2: "Model tidak tersedia" / Model Loading Error

**Error:**

```json
{
  "success": false,
  "error": "Model tidak tersedia"
}
```

**Penyebab:**

1. Model file tidak ditemukan
2. Model belum di-download
3. CUDA/GPU issue

**Solusi:**

```bash
# 1. Check model file ada atau tidak
ls -la runs/detect/train4/weights/

# 2. Jika tidak ada, download manual
python -c "from ultralytics import YOLO; YOLO('best.pt')"

# 3. Restart API dengan verbose logging
python api_v2.py --verbose

# 4. Jika GPU error, jalankan dengan CPU
export DEVICE=cpu
python api_v2.py

# 5. Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Problem 3: "File format tidak didukung"

**Error:**

```json
{
  "success": false,
  "error": "File format tidak didukung"
}
```

**Penyebab:**

1. Format file tidak didukung
2. File corrupted
3. Wrong file extension

**Solusi:**

```python
# 1. Check format
import os
SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
filename = "image.gif"
if os.path.splitext(filename)[1].lower() not in SUPPORTED:
    print("Format tidak didukung!")

# 2. Konversi ke format yang didukung
from PIL import Image
img = Image.open("image.gif")
img.save("image.png")

# 3. Verify file bukan corrupted
try:
    img = Image.open("image.jpg")
    img.verify()
    print("File OK")
except Exception as e:
    print(f"File corrupted: {e}")
```

---

### Problem 4: "File terlalu besar" / Timeout

**Error:**

```json
{
  "success": false,
  "error": "File terlalu besar (max 50MB)"
}
```

**Penyebab:**

1. File lebih dari 50MB
2. Request timeout

**Solusi:**

```python
import os
from PIL import Image

# 1. Check file size
filename = "image.jpg"
file_size_mb = os.path.getsize(filename) / (1024 * 1024)
print(f"File size: {file_size_mb:.1f} MB")

# 2. Compress jika terlalu besar
if file_size_mb > 10:
    img = Image.open(filename)

    # Reduce resolution
    max_width = 1920
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # Reduce quality
    img.save("compressed.jpg", quality=75, optimize=True)
    print("File compressed!")

# 3. Increase timeout pada client
import requests
response = requests.post(
    "http://localhost:8000/v1/detect",
    files=files,
    timeout=300  # 5 minutes
)
```

---

### Problem 5: Memory Error / Out of Memory

**Error:**

```
CUDA out of memory. Tried to allocate X.XX GiB...
```

**Penyebab:**

1. Model terlalu besar untuk GPU
2. Batch size terlalu besar
3. Image resolution terlalu tinggi

**Solusi:**

```python
# 1. Run dengan CPU (lebih lambat tapi tidak ada memory limit)
import os
os.environ['DEVICE'] = 'cpu'

# 2. Gunakan model yang lebih kecil
# Ganti dari YOLOv12x ke YOLOv12n (lebih kecil)

# 3. Reduce batch size
# Dari 50 menjadi 10 images per batch

# 4. Monitor GPU
# nvidia-smi

# 5. Clear GPU cache sebelum run
import torch
torch.cuda.empty_cache()
```

---

### Problem 6: Slow Performance / High Latency

**Q: Deteksi sangat lambat, bagaimana cara mempercepat?**

**Solusi:**

```python
# 1. Use GPU (jika belum)
import torch
if not torch.cuda.is_available():
    print("GPU tidak tersedia, install CUDA")

# 2. Optimize input size
# Jangan use 1920x1080, gunakan 640x480
from PIL import Image
img = Image.open("image.jpg")
img.thumbnail((640, 480))

# 3. Batch processing bukan sequential
# Jangan process 1 per 1, batch 10-20

# 4. Connection pooling
import requests
session = requests.Session()
for image in images:
    session.post("http://localhost:8000/v1/detect", files={"file": image})

# 5. Monitor
# python -c "import time; start = time.time(); requests.get('http://localhost:8000/v1/health'); print(time.time()-start)"
```

---

### Problem 7: API Crash / Unexpected Exit

**Error:**

```
API crashed suddenly
Error traceback...
```

**Penyebab:**

1. Memory issue
2. Unhandled exception
3. Incompatible library version

**Solusi:**

```bash
# 1. Check logs
tail -f api.log

# 2. Run dengan debug mode
python -u api_v2.py 2>&1 | tee debug.log

# 3. Verify dependencies version
pip list | grep -E "torch|ultralytics|fastapi|opencv"

# 4. Upgrade packages
pip install --upgrade -r requirements-api-v2.txt

# 5. Use process manager (untuk auto-restart)
# Install supervisor
pip install supervisor

# supervisor config
cat > /etc/supervisor/conf.d/api.conf << EOF
[program:eye-disease-api]
command=python /path/to/api_v2.py
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/api.log
EOF

supervisorctl reread
supervisorctl update
```

---

### Problem 8: CORS Error di Browser

**Error:**

```
Access to XMLHttpRequest at 'http://localhost:8000/v1/detect'
from origin 'http://localhost:3000' has been blocked by CORS policy
```

**Penyebab:**

1. CORS tidak dikonfigurasi
2. Origin tidak diizinkan

**Solusi:**

CORS sudah dikonfigurasi di api_v2.py untuk allow all origins. Jika tetap error:

```python
# Verify CORS middleware ada di api_v2.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jika masih error, restart API
```

---

### Problem 9: Deteksi Tidak Akurat / False Positives

**Q: Hasil deteksi banyak false positives atau miss detections**

**Solusi:**

```python
# 1. Adjust confidence threshold
# Higher threshold = fewer false positives tapi lebih banyak miss detections
response = requests.post(
    "http://localhost:8000/v1/detect",
    files=files,
    params={"conf": 0.5}  # Coba 0.5 instead of default 0.25
)

# 2. Adjust IOU threshold
# Higher IOU = lebih ketat dalam menghapus duplicate detections
params={"iou": 0.6}

# 3. Pre-process image
from PIL import Image
import cv2
import numpy as np

# Enhance contrast
img = cv2.imread("image.jpg")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l = clahe.apply(l)
lab = cv2.merge([l,a,b])
img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
cv2.imwrite("enhanced.jpg", img_enhanced)

# 4. Try different input sizes
# Some images work better at different resolutions
for size in [512, 640, 768, 1024]:
    # Resize and detect
```

---

## 📞 Getting Help

### Resources

1. **API Documentation**: `/v1/docs` (Swagger UI)
2. **Logs**: Check `api.log` file
3. **GitHub Issues**: [Project Repository]
4. **Stack Overflow**: Tag with `yolov12`, `fastapi`, `eye-disease`

### Debug Checklist

Sebelum report bug, pastikan sudah check:

- [ ] API berjalan? (`curl http://localhost:8000/v1/health`)
- [ ] Model ter-load? (Check health response)
- [ ] File format didukung? (JPEG, PNG, BMP, WebP, TIFF)
- [ ] File size OK? (Max 50MB)
- [ ] Network connection OK? (Tidak behind proxy)
- [ ] Dependencies up-to-date? (`pip list`)
- [ ] Coba dengan contoh script yang disediakan?

### Report Bug

Jika masih ada masalah, report dengan informasi:

```
API Version: 2.0.0
Python Version: 3.x
OS: Windows/Linux/MacOS
GPU: CUDA 11.8 / CPU
Error Message: [Paste full error]
Steps to Reproduce: [Detailed steps]
Expected Behavior: [What should happen]
Actual Behavior: [What actually happened]
```

---

## 📚 Additional Resources

### Learning Materials

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics YOLOv12 Docs](https://docs.ultralytics.com/)
- [REST API Best Practices](https://restfulapi.net/)

### Tools

- [Postman](https://www.postman.com/) - API testing
- [Swagger Editor](https://editor.swagger.io/) - API documentation
- [curl](https://curl.se/) - Command line testing

---

**Last Updated**: April 20, 2024 | **API Version**: 2.0.0
