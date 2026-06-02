"""
Contoh script untuk menggunakan YOLOv12 Detection API
"""

import requests
import json
from pathlib import Path
import time

# API Base URL
API_BASE_URL = "http://localhost:8000"


class YOLOv12APIClient:
    """Client untuk YOLOv12 Detection API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_available_models(self) -> dict:
        """Get list of available models"""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def predict_from_file(
        self,
        image_path: str,
        model_name: str = "yolov12m.pt",
        conf_threshold: float = 0.25,
        img_size: int = 640
    ) -> dict:
        """
        Predict dari file image
        
        Args:
            image_path: Path ke file gambar
            model_name: Nama model YOLOv12
            conf_threshold: Confidence threshold
            img_size: Ukuran gambar
        """
        files = {"file": open(image_path, "rb")}
        params = {
            "model_name": model_name,
            "conf_threshold": conf_threshold,
            "img_size": img_size
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            files=files,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def predict_from_url(
        self,
        image_url: str,
        model_name: str = "yolov12m.pt",
        conf_threshold: float = 0.25,
        img_size: int = 640
    ) -> dict:
        """
        Predict dari URL image
        
        Args:
            image_url: URL gambar
            model_name: Nama model YOLOv12
            conf_threshold: Confidence threshold
            img_size: Ukuran gambar
        """
        params = {
            "image_url": image_url,
            "model_name": model_name,
            "conf_threshold": conf_threshold,
            "img_size": img_size
        }
        
        response = requests.post(
            f"{self.base_url}/predict-url",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def switch_model(self, model_name: str) -> dict:
        """Switch ke model berbeda"""
        params = {"model_name": model_name}
        response = requests.post(
            f"{self.base_url}/switch-model",
            params=params
        )
        response.raise_for_status()
        return response.json()


def example_1_health_check():
    """Contoh 1: Check API health"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Health Check")
    print("="*60)
    
    client = YOLOv12APIClient()
    
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"Error: {e}")


def example_2_get_models():
    """Contoh 2: Get available models"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Get Available Models")
    print("="*60)
    
    client = YOLOv12APIClient()
    
    try:
        models_info = client.get_available_models()
        print("Available Models:")
        for model in models_info['available_models']:
            print(f"  - {model}")
        
        print(f"\nCurrently Loaded: {models_info['currently_loaded']}")
    except Exception as e:
        print(f"Error: {e}")


def example_3_predict_from_file():
    """Contoh 3: Predict dari file"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Predict from File")
    print("="*60)
    
    client = YOLOv12APIClient()
    
    # Pastikan file image ada
    image_path = "test_image.jpg"  # Ganti dengan path gambar Anda
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    try:
        result = client.predict_from_file(
            image_path=image_path,
            model_name="yolov12m.pt",
            conf_threshold=0.25,
            img_size=640
        )
        
        print(f"Status: {result['status']}")
        print(f"Model: {result['model']}")
        print(f"Detections: {result['detection_count']}")
        
        if result['detections']:
            print("\nDetections:")
            for i, det in enumerate(result['detections'], 1):
                print(f"  {i}. {det['class_name']}")
                print(f"     Confidence: {det['confidence']:.2%}")
                print(f"     BBox: ({det['bbox']['x1']:.0f}, {det['bbox']['y1']:.0f})")
                print(f"            ({det['bbox']['x2']:.0f}, {det['bbox']['y2']:.0f})")
        else:
            print("No objects detected")
    
    except Exception as e:
        print(f"Error: {e}")


def example_4_predict_from_url():
    """Contoh 4: Predict dari URL"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Predict from URL")
    print("="*60)
    
    client = YOLOv12APIClient()
    
    # Contoh URL gambar (Anda bisa ganti dengan URL gambar sebenarnya)
    image_url = "https://ultralytics.com/images/bus.jpg"
    
    try:
        result = client.predict_from_url(
            image_url=image_url,
            model_name="yolov12m.pt",
            conf_threshold=0.25,
            img_size=640
        )
        
        print(f"Status: {result['status']}")
        print(f"Image URL: {result['image_url']}")
        print(f"Detections: {result['detection_count']}")
        
        if result['detections']:
            print("\nDetections:")
            for i, det in enumerate(result['detections'], 1):
                print(f"  {i}. {det['class_name']} ({det['confidence']:.2%})")
        else:
            print("No objects detected")
    
    except Exception as e:
        print(f"Error: {e}")


def example_5_switch_model():
    """Contoh 5: Switch model"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Switch Model")
    print("="*60)
    
    client = YOLOv12APIClient()
    
    try:
        print("Switching from yolov12m.pt to yolov12l.pt...")
        result = client.switch_model("yolov12l.pt")
        
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        print(f"Loaded Models: {result['loaded_models']}")
    
    except Exception as e:
        print(f"Error: {e}")


def example_6_batch_prediction():
    """Contoh 6: Batch prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Batch Prediction")
    print("="*60)
    
    client = YOLOv12APIClient()
    
    # List of image files
    image_files = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    ]
    
    results = []
    
    for image_path in image_files:
        if not Path(image_path).exists():
            print(f"Skipping {image_path} (not found)")
            continue
        
        try:
            print(f"Processing {image_path}...")
            result = client.predict_from_file(
                image_path=image_path,
                model_name="yolov12m.pt"
            )
            
            results.append({
                "image": image_path,
                "detections": result['detection_count'],
                "classes": [det['class_name'] for det in result['detections']]
            })
            
            print(f"  -> Found {result['detection_count']} objects")
        
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # Summary
    print("\n" + "="*40)
    print("BATCH RESULTS SUMMARY")
    print("="*40)
    for r in results:
        print(f"{r['image']}: {r['detections']} detections")
        if r['classes']:
            print(f"  Classes: {', '.join(set(r['classes']))}")


def example_7_performance_comparison():
    """Contoh 7: Perbandingan performa antar model"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Model Performance Comparison")
    print("="*60)
    
    client = YOLOv12APIClient()
    
    image_path = "test_image.jpg"  # Ganti dengan path gambar Anda
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    models = ["yolov12n.pt", "yolov12s.pt", "yolov12m.pt", "yolov12l.pt"]
    
    print(f"Testing with: {image_path}\n")
    
    for model in models:
        try:
            print(f"Testing {model}...")
            
            start_time = time.time()
            result = client.predict_from_file(
                image_path=image_path,
                model_name=model
            )
            inference_time = time.time() - start_time
            
            print(f"  Inference Time: {inference_time:.3f}s")
            print(f"  Detections: {result['detection_count']}")
            
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    """
    Jalankan example dengan menjalankan function yang sesuai:
    
    python example_client.py
    """
    
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║     YOLOv12 Detection API - Example Client            ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    # Uncomment example yang ingin dijalankan:
    
    example_1_health_check()
    example_2_get_models()
    # example_3_predict_from_file()
    # example_4_predict_from_url()
    # example_5_switch_model()
    # example_6_batch_prediction()
    # example_7_performance_comparison()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60 + "\n")
