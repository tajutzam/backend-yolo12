"""
Simple startup script for YOLO v12 Eye Disease API
"""
import sys
import os
from pathlib import Path

# Add yolov12 directory to path
yolov12_dir = Path(__file__).parent
sys.path.insert(0, str(yolov12_dir))

# Change to yolov12 directory
os.chdir(yolov12_dir)

# Now import and run
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting YOLO v12 Eye Disease Detection API...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🔧 Python path includes: {yolov12_dir}")
    
    uvicorn.run(
        "api_v2:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info"
    )
