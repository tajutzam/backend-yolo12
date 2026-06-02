#!/usr/bin/env python
"""Test YOLOv12 model loading without making predictions"""

import requests
import json

API_URL = 'http://localhost:8000'

print('=' * 70)
print('✅ YOLOv12 MODEL LOADING TEST')
print('=' * 70)

# Test 1: Health Check
print('\n1️⃣  HEALTH CHECK')
print('-' * 70)
try:
    r = requests.get(f'{API_URL}/health', timeout=10)
    print(f'Status Code: {r.status_code}')
    data = r.json()
    print(f'API Status: {data.get("status")}')
    print(f'Model Loaded: {data.get("model_loaded")}')
    if r.status_code == 200 and data.get("model_loaded"):
        print('✅ API is running and model loaded successfully!')
    else:
        print('❌ Model loading failed')
except Exception as e:
    print(f'❌ Error: {e}')

# Test 2: Get Available Models
print('\n2️⃣  AVAILABLE MODELS')
print('-' * 70)
try:
    r = requests.get(f'{API_URL}/models', timeout=10)
    print(f'Status Code: {r.status_code}')
    data = r.json()
    print(f'Available Models: {data.get("available_models")}')
    print(f'Default Model: {data.get("default_model")}')
    print(f'Currently Loaded: {data.get("currently_loaded")}')
    if data.get('currently_loaded'):
        print('✅ Model(s) successfully loaded in memory')
except Exception as e:
    print(f'❌ Error: {e}')

# Test 3: API Info
print('\n3️⃣  API INFORMATION')
print('-' * 70)
try:
    r = requests.get(f'{API_URL}/', timeout=10)
    print(f'Status Code: {r.status_code}')
    data = r.json()
    print(f'API Name: {data.get("message")}')
    print(f'Version: {data.get("version")}')
    print(f'Available Endpoints:')
    for endpoint, info in data.get("endpoints", {}).items():
        print(f'  - {endpoint}')
except Exception as e:
    print(f'❌ Error: {e}')

print('\n' + '=' * 70)
print('✅ MODEL LOADING TEST COMPLETE')
print('=' * 70)
print('\nNote: This test verifies that the model loads successfully.')
print('Actual prediction functionality may require additional configuration.')
