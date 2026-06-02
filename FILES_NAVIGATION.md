# 🗂️ API Files Navigation & Quick Reference

## 📁 Project Structure

```
yolov12/
├── 📄 QUICKSTART_API.md                   ⭐ START HERE - 5 minute setup
├── 📄 README_API_V2.md                    📚 Complete API documentation
├── 📄 SETUP_DEPLOYMENT_GUIDE.md           🚀 Production deployment guide
├── 📄 API_IMPLEMENTATION_SUMMARY.md       📋 This implementation overview
│
├── 🔧 API Core Files
│   ├── api_v2.py                          (800+ lines) Main FastAPI application
│   ├── client_v2.py                       (300+ lines) Python client library
│   └── requirements-api-v2.txt            Python dependencies
│
├── 🐳 Docker & Deployment
│   ├── Dockerfile-api-v2                  Multi-stage Docker build
│   ├── docker-compose-api-v2.yml          Service orchestration
│   ├── start_api.sh                       Linux startup script
│   ├── start_api.bat                      Windows startup script
│   ├── test_api_v2.sh                     Automated test script
│   └── nginx.conf                         Reverse proxy config
│
├── ⚙️ Configuration
│   └── .env.example                       Environment template
│
├── 📚 Examples & Tests
│   ├── example_api_notebook.py            Python usage examples
│   └── test_api_v2.sh                     Test automation
│
└── 📁 Data
    └── runs/detect/train4/weights/best.pt Trained YOLO v12 model
```

---

## 🎯 Getting Started by User Type

### 👤 I'm a Developer/Researcher

**Goal**: Setup API locally and test it

**Steps**:

1. Read: `QUICKSTART_API.md`
2. Run: `start_api.bat` (Windows) or `./start_api.sh` (Linux)
3. Test: Open http://localhost:8000/v1/docs
4. Code: Check `example_api_notebook.py`

**Time**: 10-15 minutes

---

### 👤 I'm a DevOps/System Admin

**Goal**: Deploy to production

**Steps**:

1. Read: `SETUP_DEPLOYMENT_GUIDE.md`
2. Choose: Docker or Linux server setup
3. Configure: Environment variables in `.env`
4. Deploy: Using docker-compose or systemd
5. Monitor: Check logs and health

**Time**: 30-60 minutes

---

### 👤 I'm an End User / Researcher

**Goal**: Use the API for inference

**Steps**:

1. Read: `README_API_V2.md` (Section: API Endpoints)
2. Setup: Use docker-compose
3. Upload: Images via Swagger UI
4. Analyze: Results with statistics

**Time**: 15-20 minutes

---

## 📖 Documentation Map

### For Quick Setup

- ✅ **QUICKSTART_API.md** - 5-minute setup
- ✅ **start_api.sh** / **start_api.bat** - Run these

### For API Usage

- ✅ **README_API_V2.md** - Complete reference
- ✅ **Swagger UI** - Interactive docs at `/v1/docs`
- ✅ **example_api_notebook.py** - Code examples

### For Deployment

- ✅ **SETUP_DEPLOYMENT_GUIDE.md** - All deployment options
- ✅ **docker-compose-api-v2.yml** - Docker compose config
- ✅ **nginx.conf** - Reverse proxy setup

### For Development

- ✅ **api_v2.py** - Main application code
- ✅ **client_v2.py** - Client implementation
- ✅ **requirements-api-v2.txt** - Dependencies

### For Troubleshooting

- ✅ **README_API_V2.md** (Section: Troubleshooting)
- ✅ **SETUP_DEPLOYMENT_GUIDE.md** (Section: Troubleshooting)
- ✅ **test_api_v2.sh** - Run automated tests

---

## 🚀 Command Quick Reference

### Start API Locally

#### Windows

```bash
cd yolov12
python -m venv venv
venv\Scripts\activate
pip install -r requirements-api-v2.txt
start_api.bat
```

#### Linux/Mac

```bash
cd yolov12
python -m venv venv
source venv/bin/activate
pip install -r requirements-api-v2.txt
chmod +x start_api.sh && ./start_api.sh
```

### Docker

```bash
# Build
docker build -f Dockerfile-api-v2 -t yolov12-eye-api .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/runs:/app/runs:ro \
  yolov12-eye-api

# Or with Docker Compose
docker-compose -f docker-compose-api-v2.yml up
```

### Test API

```bash
# Health check
curl http://localhost:8000/v1/health | jq

# Full test
bash test_api_v2.sh

# Python client
python client_v2.py --health
python client_v2.py --image test.jpg
```

---

## 📊 API Endpoints Reference

| Endpoint                | Method | Purpose                   |
| ----------------------- | ------ | ------------------------- |
| `/v1/health`            | GET    | Check if API is running   |
| `/v1/model/info`        | GET    | Get model information     |
| `/v1/detect`            | POST   | Detect in single image    |
| `/v1/detect/batch`      | POST   | Detect in multiple images |
| `/v1/supported-formats` | GET    | List supported file types |
| `/v1/docs`              | GET    | Swagger UI documentation  |
| `/v1/redoc`             | GET    | ReDoc documentation       |

---

## 🔑 Key Files Explained

### `api_v2.py`

- Main FastAPI application
- 800+ lines of code
- Handles all endpoints
- Loads YOLO model
- Processes images
- Returns JSON responses

### `client_v2.py`

- Python client library
- 300+ lines
- Test API from Python
- Batch processing
- Image saving
- CLI support

### `Dockerfile-api-v2`

- Multi-stage Docker build
- Optimized image size
- GPU support ready
- Production-ready

### `docker-compose-api-v2.yml`

- Orchestrate services
- API container
- Optional Nginx
- Health checks
- Volume mounts

### Requirements Files

- `requirements-api-v2.txt` - FastAPI + dependencies (13 packages)
- `requirements.txt` - Full YOLOv12 setup
- `requirements-api.txt` - Gradio version (legacy)

---

## ⚙️ Configuration Guide

### Environment Variables (`.env`)

```bash
HOST=0.0.0.0                    # API host
PORT=8000                       # API port
DEBUG=False                     # Debug mode
WORKERS=4                       # Number of workers
DEVICE=cuda                     # cuda or cpu
CONFIDENCE_THRESHOLD=0.25       # Detection confidence
IOU_THRESHOLD=0.45              # NMS threshold
MAX_IMAGE_SIZE_MB=50            # Max file size
MAX_BATCH_SIZE=50               # Max images per batch
```

### Model Configuration

- **Path**: `runs/detect/train4/weights/best.pt`
- **Size**: ~300-400 MB
- **Task**: Object detection
- **Classes**: 5 (Cataracts, Pterygium, etc.)

---

## 🧪 Testing Checklist

Before deployment, verify:

- [ ] Model file exists at correct path
- [ ] Health endpoint returns 200
- [ ] Model loads without errors
- [ ] Single image detection works
- [ ] Batch detection works
- [ ] API responds in <1 second
- [ ] GPU/CPU detected correctly
- [ ] All dependencies installed

```bash
# Automated check
bash test_api_v2.sh
```

---

## 📈 Performance Tuning

| Scenario           | Recommendation                                       |
| ------------------ | ---------------------------------------------------- |
| **Slow Detection** | Increase workers, use GPU, reduce image size         |
| **High Load**      | Use batch API, implement caching, scale horizontally |
| **Memory Issues**  | Reduce batch size, use CPU, limit max workers        |
| **Large Files**    | Compress images before upload, increase timeout      |

---

## 🔐 Security Checklist

- [ ] Use HTTPS/SSL in production
- [ ] Implement API authentication
- [ ] Add rate limiting
- [ ] Validate all inputs
- [ ] Sanitize error messages
- [ ] Keep dependencies updated
- [ ] Monitor access logs
- [ ] Regular security scans

---

## 📞 Troubleshooting Quick Links

| Problem                 | Solution                                       |
| ----------------------- | ---------------------------------------------- |
| **Port already in use** | Change port: `--port 8001`                     |
| **Model not found**     | Check: `ls runs/detect/train4/weights/best.pt` |
| **Out of memory**       | Reduce batch size or use CPU                   |
| **Connection refused**  | Ensure API is running on correct port          |
| **Slow response**       | Check GPU usage: `nvidia-smi`                  |

---

## 📚 Reading Order

1. **First**: `API_IMPLEMENTATION_SUMMARY.md` (this file - overview)
2. **Second**: `QUICKSTART_API.md` (quick setup)
3. **Third**: `README_API_V2.md` (detailed docs)
4. **Fourth**: `SETUP_DEPLOYMENT_GUIDE.md` (production)

---

## 🎯 Common Tasks

### Run API in Development

```bash
python -m uvicorn api_v2:app --reload --host 0.0.0.0 --port 8000
```

### Run API in Production

```bash
python -m uvicorn api_v2:app --host 0.0.0.0 --port 8000 --workers 4
```

### Test Single Image

```bash
python client_v2.py --image test.jpg --save-image output.png
```

### Test Batch

```bash
python client_v2.py --batch img1.jpg img2.jpg img3.jpg
```

### View Logs

```bash
tail -f logs/api.log
docker logs yolov12-eye-disease-api
```

### Check Health

```bash
curl http://localhost:8000/v1/health | jq
```

---

## 📞 Support & Help

### Logs Location

```
logs/
├── api.log          - Main API logs
└── inference.log    - Model inference logs
```

### Debug Mode

```bash
# Enable debug logging
python -m uvicorn api_v2:app --log-level debug
```

### Health Endpoint

```bash
# Always working (if API runs)
curl http://localhost:8000/v1/health
```

---

## 📝 Version Information

| Component | Version | Status         |
| --------- | ------- | -------------- |
| API       | 2.0.0   | ✅ Latest      |
| FastAPI   | 0.104.1 | ✅ Stable      |
| YOLOv12   | Latest  | ✅ Integrated  |
| PyTorch   | 2.2.2   | ✅ Optimized   |
| Python    | 3.11    | ✅ Recommended |

---

## 🎓 Research Details

**Penelitian:**

```
DETEKSI OTOMATIS PENYAKIT MATA ANTERIOR
PADA CITRA NON-FUNDUS MENGGUNAKAN YOLOV12
BERBASIS ATTENTION DAN EXPLAINABLE AI
```

**Model:** YOLOv12 with CBAM Attention Module  
**XAI Method:** Grad-CAM for visual explanations  
**Input:** Anterior eye disease images (non-fundus)  
**Output:** Disease detection with confidence scores

---

## ✨ Final Checklist

- [ ] Baca `QUICKSTART_API.md`
- [ ] Setup environment (virtual env)
- [ ] Install dependencies
- [ ] Verify model exists
- [ ] Start API server
- [ ] Test health endpoint
- [ ] Access Swagger UI
- [ ] Run test script
- [ ] Try single image detection
- [ ] Try batch detection

---

**Total Time to Production**: 1-2 hours  
**Difficulty Level**: Beginner-friendly  
**Support**: Full documentation provided

**Status**: ✅ **READY TO DEPLOY**

---

_Created: April 2026_  
_API Version: 2.0.0_  
_Document Version: 1.0.0_  
_Last Updated: April 19, 2026_
