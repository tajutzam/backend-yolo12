#!/usr/bin/env python
"""Test YOLOv12 Detection API"""

import requests
import json

API_URL = 'http://localhost:8000'

print('🧪 TESTING YOLOV12 DETECTION API')
print('=' * 70)

# Test 1: Health Check
print('\n1️⃣  HEALTH CHECK')
print('-' * 70)
r = requests.get(f'{API_URL}/health')
print(f'Status Code: {r.status_code}')
print(f'Response: {json.dumps(r.json(), indent=2)}')

# Test 2: Get Available Models
print('\n2️⃣  AVAILABLE MODELS')
print('-' * 70)
r = requests.get(f'{API_URL}/models')
print(f'Status Code: {r.status_code}')
data = r.json()
print(f'Available Models: {data.get("available_models")}')
print(f'Default Model: {data.get("default_model")}')
print(f'Currently Loaded: {data.get("currently_loaded")}')

# Test 3: API Info
print('\n3️⃣  API INFORMATION')
print('-' * 70)
r = requests.get(f'{API_URL}/')
print(f'Status Code: {r.status_code}')
data = r.json()
print(f'API Name: {data.get("message")}')
print(f'Version: {data.get("version")}')
print(f'Endpoints: {list(data.get("endpoints", {}).keys())}')

# Test 4: Predict from URL
print('\n4️⃣  PREDICTION FROM URL')
print('-' * 70)
image_url = 'https://ultralytics.com/images/bus.jpg'
print(f'Testing with: {image_url}')
print('Loading local model best.pt and running inference...')

r = requests.post(
    f'{API_URL}/predict-url',
    params={
        'image_url': image_url,
        'model_name': 'best',  # Use local best.pt model
        'conf_threshold': 0.25,
        'img_size': 640
    },
    timeout=120
)

print(f'Status Code: {r.status_code}')
result = r.json()

if r.status_code == 200:
    print(f'\nResults:')
    print(f'  Model: {result.get("model")}')
    print(f'  Image Size: {result.get("image_size")}')
    print(f'  Confidence Threshold: {result.get("confidence_threshold")}')
    print(f'  Total Detections: {result.get("detection_count")}')
    
    if result.get('detection_count', 0) > 0:
        print(f'\n  Detected Objects:')
        for i, det in enumerate(result.get('detections', []), 1):
            conf_pct = det['confidence'] * 100
            print(f'    {i}. {det["class_name"]:15s} | Confidence: {conf_pct:6.2f}% | BBox: ({det["bbox"]["x1"]:.0f}, {det["bbox"]["y1"]:.0f})')
    else:
        print('  No objects detected')
else:
    print(f'Error: {result.get("detail", "Unknown error")}')

print('\n' + '=' * 70)
print('✅ API TEST COMPLETED SUCCESSFULLY!')
print('=' * 70)
