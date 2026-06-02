"""
Example Notebook untuk Testing YOLO v12 Eye Disease Detection API
Deteksi Otomatis Penyakit Mata Anterior pada Citra Non-Fundus

Gunakan notebook ini untuk:
1. Testing API dalam development mode
2. Visualisasi hasil deteksi
3. Analisis statistik deteksi
4. Export hasil untuk publikasi
"""

# ============================================================================
# Setup & Imports
# ============================================================================

import os
import json
import requests
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# API Configuration
API_BASE_URL = "http://localhost:8000"
EXPECTED_CLASSES = ['Cataracts', 'Pterygium', 'Diabetes_Retinopathy', 'Keratoconus', 'Normal']

print("✓ Imports completed")

# ============================================================================
# 1. API Health Check & Information
# ============================================================================

def test_api_health():
    """Test API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ API Health Check:")
            print(f"  - Status: {data['status']}")
            print(f"  - Model Loaded: {data['model_loaded']}")
            print(f"  - Device: {data['device']}")
            print(f"  - Classes: {len(data['model_info']['classes'])}")
            return True
        else:
            print(f"✗ API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Connection error: {str(e)}")
        return False

def get_model_info():
    """Get model information"""
    response = requests.get(f"{API_BASE_URL}/v1/model/info", timeout=5)
    data = response.json()
    
    print("\n✓ Model Information:")
    print(f"  - Name: {data['model']['name']}")
    print(f"  - Task: {data['model']['task']}")
    print(f"  - Device: {data['model']['device']}")
    print(f"  - Classes: {data['model']['num_classes']}")
    print(f"\n✓ Research Info:")
    print(f"  - Title: {data['research']['title']}")
    print(f"  - Framework: {data['research']['framework']}")
    print(f"  - Attention: {data['research']['enhancement']}")
    print(f"  - XAI Method: {data['research']['interpretability']}")
    
    return data

# ============================================================================
# 2. Single Image Detection
# ============================================================================

def detect_image(image_path: str, conf: float = 0.25, return_image: bool = True):
    """
    Detect diseases in single image
    
    Args:
        image_path: Path to image
        conf: Confidence threshold
        return_image: Include annotated image in response
    
    Returns:
        Detection results
    """
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return None
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        params = {
            'conf': conf,
            'return_image': return_image
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/v1/detect",
                files=files,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"✗ API error: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None

def visualize_detection(result: dict, figsize: tuple = (14, 8)):
    """Visualize detection results"""
    
    if result is None or not result.get('success'):
        print("✗ Invalid result")
        return
    
    # Extract data
    detections = result['detections']
    stats = result['statistics']
    image_meta = result['image_metadata']
    
    print(f"\n✓ Detection Results for: {image_meta['filename']}")
    print(f"  - Image Size: {image_meta['width']}x{image_meta['height']}")
    print(f"  - Total Detections: {stats['total_detections']}")
    
    if stats['total_detections'] > 0:
        print(f"  - Average Confidence: {stats['average_confidence']:.4f}")
        print(f"  - Confidence Range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Annotated Image
    if result.get('image_base64'):
        import base64
        from io import BytesIO
        
        image_data = base64.b64decode(result['image_base64'])
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(image_rgb)
        axes[0].set_title(f"Detections: {stats['total_detections']}", fontsize=14, fontweight='bold')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Annotated image not available\n(set return_image=True)',
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].axis('off')
    
    # Plot 2: Detection Statistics
    if detections:
        # Prepare data for visualization
        classes = [d['class_name'] for d in detections]
        confidences = [d['confidence'] for d in detections]
        areas = [d['area_pixels'] for d in detections]
        
        # Count by class
        class_counts = {}
        for cls in classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Plot detection summary
        y_pos = np.arange(len(detections))
        colors = plt.cm.RdYlGn(np.array(confidences))
        
        bars = axes[1].barh(y_pos, confidences, color=colors)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels([f"{d['class_name']}" for d in detections])
        axes[1].set_xlabel('Confidence Score', fontsize=12)
        axes[1].set_title('Detection Confidence Scores', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            axes[1].text(conf + 0.02, i, f'{conf:.3f}', va='center', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No detections found',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    if detections:
        print("\n✓ Detected Diseases:")
        for idx, det in enumerate(detections, 1):
            print(f"\n  [{idx}] {det['class_name']}")
            print(f"      - Confidence: {det['confidence']:.4f}")
            print(f"      - Bounding Box: ({det['bbox']['x1']:.1f}, {det['bbox']['y1']:.1f}) → "
                  f"({det['bbox']['x2']:.1f}, {det['bbox']['y2']:.1f})")
            print(f"      - Area: {det['area_pixels']} pixels")
            print(f"      - Normalized Box: ({det['bbox_normalized']['x1']:.3f}, {det['bbox_normalized']['y1']:.3f}) → "
                  f"({det['bbox_normalized']['x2']:.3f}, {det['bbox_normalized']['y2']:.3f})")

# ============================================================================
# 3. Batch Processing
# ============================================================================

def batch_detect(image_paths: list, conf: float = 0.25):
    """
    Batch detection for multiple images
    
    Args:
        image_paths: List of image paths
        conf: Confidence threshold
    
    Returns:
        Batch results
    """
    
    files = []
    for img_path in image_paths:
        if not Path(img_path).exists():
            print(f"⚠ Skipping: {img_path} (not found)")
            continue
        
        with open(img_path, 'rb') as f:
            files.append(('files', (Path(img_path).name, f.read())))
    
    if not files:
        print("✗ No valid images found")
        return None
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/detect/batch",
            files=files,
            params={'conf': conf},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None

def create_results_dataframe(batch_result: dict):
    """Create DataFrame from batch results"""
    
    if batch_result is None or not batch_result.get('success'):
        return None
    
    data = []
    for result in batch_result['results']:
        data.append({
            'Filename': result['filename'],
            'Success': result['success'],
            'Detections': result.get('detections', 0) if result['success'] else 0,
            'Error': result.get('error', '') if not result['success'] else ''
        })
    
    df = pd.DataFrame(data)
    return df

# ============================================================================
# 4. Analysis & Statistics
# ============================================================================

def analyze_detections(detections: list):
    """Analyze detection statistics"""
    
    if not detections:
        print("No detections to analyze")
        return
    
    # Extract data
    classes = [d['class_name'] for d in detections]
    confidences = [d['confidence'] for d in detections]
    areas = [d['area_pixels'] for d in detections]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': classes,
        'Confidence': confidences,
        'Area': areas
    })
    
    print("\n✓ Detection Analysis:")
    print(f"\n  Summary Statistics:")
    print(df.describe().to_string())
    
    print(f"\n  Class Distribution:")
    print(df['Class'].value_counts().to_string())
    
    print(f"\n  Confidence by Class:")
    conf_by_class = df.groupby('Class')['Confidence'].agg(['mean', 'min', 'max', 'count'])
    print(conf_by_class.to_string())
    
    return df

# ============================================================================
# EXAMPLE USAGE - Uncomment to test
# ============================================================================

if __name__ == "__main__":
    # Test API
    if test_api_health():
        
        # Get model info
        model_info = get_model_info()
        
        # Test single image (if available)
        # Uncomment and modify path as needed
        # if Path("eye_image.jpg").exists():
        #     result = detect_image("eye_image.jpg", conf=0.25, return_image=True)
        #     if result:
        #         visualize_detection(result)
        
        # Batch processing example
        # image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]
        # batch_result = batch_detect(image_files, conf=0.25)
        # if batch_result:
        #     df_results = create_results_dataframe(batch_result)
        #     print(df_results)
    
    print("\n✓ Setup complete! Use the functions above for detection and analysis.")

"""
NOTES:
1. Ensure API is running: python -m uvicorn api_v2:app --reload
2. Update image paths according to your test images
3. Adjust confidence threshold as needed
4. For batch processing, prepare multiple images
5. Export results to DataFrame for analysis
"""
