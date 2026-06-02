import requests
import json

with open('assets/test_image.jpg', 'rb') as f:
    files = {'file': f}
    r = requests.post('http://127.0.0.1:8001/v1/detect', files=files)
    print(f'Status: {r.status_code}')
    try:
        data = r.json()
        print(f'Response: {json.dumps(data, indent=2)}')
    except:
        print(f'Raw response: {r.text[:200]}')
