#!/usr/bin/env python
"""Simple test script to check if ultralytics can be imported"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    print("\n1. Testing ultralytics import...")
    from ultralytics import YOLO
    print("✓ ultralytics imported successfully")
    
    print("\n2. Testing YOLO model load...")
    # Try to load default model
    model = YOLO('yolov8n.pt')
    print("✓ YOLO model loaded successfully")
    
    print("\n3. API is ready to use!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
