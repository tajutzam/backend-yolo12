# API Setup & Deployment Guide

## 📋 Overview

Panduan lengkap untuk setup, deployment, dan maintenance YOLO v12 Eye Disease Detection API.

---

## 1️⃣ Local Development Setup (Windows/Mac/Linux)

### Step 1: Clone/Prepare Repository

```bash
cd d:\UMNFIX\yolov12  # Windows
# or
cd ~/UMNFIX/yolov12   # Mac/Linux
```

### Step 2: Create Virtual Environment

```bash
# Python 3.9+ recommended
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements-api-v2.txt

# Verify installation
python -c "import fastapi, ultralytics; print('✓ Installation complete')"
```

### Step 4: Verify Model Location

```bash
# Windows
dir runs\detect\train4\weights\best.pt

# Mac/Linux
ls runs/detect/train4/weights/best.pt
```

### Step 5: Start Development Server

```bash
# Windows
start_api.bat

# Mac/Linux
chmod +x start_api.sh && ./start_api.sh

# Or direct command
python -m uvicorn api_v2:app --reload --host 0.0.0.0 --port 8000
```

### Step 6: Verify API is Running

```bash
# In new terminal
curl http://localhost:8000/v1/health

# Expected output:
# {"status": "healthy", "model_loaded": true, ...}
```

### Step 7: Access Web Interface

- Open: http://localhost:8000/v1/docs
- Test endpoints directly in Swagger UI

---

## 2️⃣ Docker Development Setup

### Option A: Docker Build & Run

```bash
# Build image (one-time)
docker build -f Dockerfile-api-v2 -t yolov12-eye-api:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/runs:/app/runs:ro \
  -e PYTHONUNBUFFERED=1 \
  yolov12-eye-api:latest

# Access: http://localhost:8000/v1/docs
```

### Option B: Docker Compose (Recommended)

```bash
# Start all services
docker-compose -f docker-compose-api-v2.yml up -d

# View logs
docker-compose -f docker-compose-api-v2.yml logs -f api

# Stop services
docker-compose -f docker-compose-api-v2.yml down

# Access: http://localhost:8000/v1/docs
```

### Option C: Docker with GPU Support

```bash
# Edit docker-compose-api-v2.yml - uncomment GPU section
# Then:
docker-compose -f docker-compose-api-v2.yml up -d

# Verify GPU
docker exec yolov12-eye-disease-api python -c "import torch; print(torch.cuda.is_available())"
```

---

## 3️⃣ Production Deployment

### Using Ubuntu/Linux Server

#### Step 1: Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo apt install -y git curl wget
sudo apt install -y docker.io docker-compose-v2 nvidia-docker2  # if GPU available

# Verify Docker
docker --version
docker run hello-world
```

#### Step 2: Deploy Application

```bash
# Clone/transfer project to server
cd /opt/yolov12

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-api-v2.txt

# Run in background with systemd
sudo systemctl start yolov12-api
```

#### Step 3: Setup Systemd Service

Create `/etc/systemd/system/yolov12-api.service`:

```ini
[Unit]
Description=YOLO v12 Eye Disease Detection API
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/yolov12
Environment="PATH=/opt/yolov12/venv/bin"
ExecStart=/opt/yolov12/venv/bin/python -m uvicorn api_v2:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable yolov12-api
sudo systemctl start yolov12-api
sudo systemctl status yolov12-api
```

#### Step 4: Setup Nginx Reverse Proxy

```bash
sudo apt install -y nginx

# Copy nginx config
sudo cp nginx.conf /etc/nginx/sites-available/yolov12

# Enable site
sudo ln -s /etc/nginx/sites-available/yolov12 /etc/nginx/sites-enabled/

# Test config
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

#### Step 5: Setup SSL Certificate (Let's Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx

# Generate certificate
sudo certbot certonly --nginx -d yourdomain.com

# Auto-renew
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

#### Step 6: Monitor Service

```bash
# View logs
sudo journalctl -u yolov12-api -f

# Check status
sudo systemctl status yolov12-api

# View nginx logs
sudo tail -f /var/log/nginx/access.log
```

---

## 4️⃣ AWS/Cloud Deployment

### EC2 Instance Setup

```bash
# Launch EC2 instance
# - AMI: Ubuntu 22.04 LTS
# - Instance: p3.2xlarge (GPU) or t3.large (CPU)
# - Security Group: Allow port 80, 443, 22

# SSH into instance
ssh -i key.pem ubuntu@your-instance-ip

# Follow Linux deployment steps above
```

### AWS Lambda + API Gateway (Serverless)

```bash
# Requires: Lambda layer with PyTorch (large file size limitation)
# Not recommended for real-time detection (cold start issues)
# Use EC2 or ECS instead
```

### AWS ECS + Fargate (Container Orchestration)

```bash
# Push Docker image to ECR
aws ecr create-repository --repository-name yolov12-eye-api

# Build and push
docker build -t yolov12-eye-api .
docker tag yolov12-eye-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/yolov12-eye-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/yolov12-eye-api:latest

# Create ECS task definition and service
# (Use AWS console or CloudFormation)
```

---

## 5️⃣ Testing & Validation

### Health Check

```bash
curl http://localhost:8000/v1/health | jq

# Should return:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cuda" or "cpu"
# }
```

### Model Information

```bash
curl http://localhost:8000/v1/model/info | jq
```

### Test Detection

```bash
# Single image
curl -X POST http://localhost:8000/v1/detect \
  -F "file=@test_image.jpg" | jq '.statistics'

# Batch
curl -X POST http://localhost:8000/v1/detect/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" | jq '.results'
```

### Load Testing

```bash
# Install Apache Bench
sudo apt install apache2-utils

# Simple load test
ab -n 100 -c 10 http://localhost:8000/v1/health

# More realistic test with file uploads
# (Requires custom script with wrk or locust)
```

---

## 6️⃣ Monitoring & Maintenance

### Log Management

```bash
# View application logs
tail -f logs/api.log

# Filter errors
grep ERROR logs/api.log

# Archive old logs
find logs/ -name "*.log" -mtime +30 -delete
```

### Performance Monitoring

```bash
# Monitor GPU usage
nvidia-smi

# Monitor system resources
htop

# Check disk space
df -h

# Check memory usage
free -h
```

### Backup Model

```bash
# Backup trained model
cp runs/detect/train4/weights/best.pt best_backup_$(date +%Y%m%d).pt

# Backup database/results (if applicable)
tar -czf backup_$(date +%Y%m%d).tar.gz results/
```

### Update Dependencies

```bash
# Check for updates
pip list --outdated

# Update all packages safely
pip install --upgrade -r requirements-api-v2.txt

# Test after update
python -m uvicorn api_v2:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 7️⃣ Scaling Recommendations

### Horizontal Scaling

- Use load balancer (Nginx, HAProxy)
- Multiple API instances on different ports
- Shared model cache (if needed)

### Vertical Scaling

- Use more powerful GPU
- Increase worker processes
- Upgrade server RAM

### Batch Processing

- Process images in batches
- Queue system (Celery, RQ)
- Scheduled jobs

---

## 8️⃣ Security Checklist

- [ ] Update all dependencies to latest versions
- [ ] Enable HTTPS/SSL (Let's Encrypt)
- [ ] Implement API authentication (JWT, API keys)
- [ ] Add rate limiting
- [ ] Setup firewall rules
- [ ] Regular security updates
- [ ] Monitor access logs
- [ ] Backup critical data
- [ ] Setup error alerts
- [ ] Regular penetration testing

---

## 9️⃣ Troubleshooting

### Issue: Model not loading

```bash
# Check model file
ls -lh runs/detect/train4/weights/best.pt

# Test model independently
python -c "from ultralytics import YOLO; m = YOLO('runs/detect/train4/weights/best.pt')"
```

### Issue: Out of memory

```bash
# Reduce batch size
# Use CPU instead of GPU
# Reduce image size

# Or monitor memory
watch nvidia-smi
```

### Issue: Slow responses

```bash
# Check GPU utilization
nvidia-smi

# Increase workers
python -m uvicorn api_v2:app --workers 8

# Use batch processing API
```

### Issue: Port already in use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 PID

# Or use different port
python -m uvicorn api_v2:app --port 8001
```

---

## 🔟 Support & Contact

**Documentation:**

- Full API Docs: `README_API_V2.md`
- Quick Start: `QUICKSTART_API.md`
- Swagger UI: http://localhost:8000/v1/docs

**Logs Location:**

```
logs/
├── api.log
├── inference.log
└── error.log
```

**Issue Resolution:**

1. Check logs
2. Verify configuration
3. Test health endpoint
4. Run test script

---

**Last Updated**: April 2026  
**API Version**: 2.0.0  
**Document Version**: 1.0.0
