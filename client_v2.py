"""
Client untuk testing YOLO v12 Eye Disease Detection API
"""

import requests
import json
import base64
from pathlib import Path
from typing import Dict, List, Any
import cv2
import argparse
from datetime import datetime

class EyeDiseaseAPIClient:
    """Client untuk Eye Disease Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EyeDiseaseAPIClient/2.0"
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/v1/health")
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/v1/model/info")
        return response.json()
    
    def detect(self, image_path: str, conf: float = 0.25, iou: float = 0.45, return_image: bool = False) -> Dict[str, Any]:
        """
        Detect diseases in an image
        
        Args:
            image_path: Path to image file
            conf: Confidence threshold
            iou: IOU threshold
            return_image: Return annotated image
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {
                'conf': conf,
                'iou': iou,
                'return_image': return_image
            }
            response = self.session.post(
                f"{self.base_url}/v1/detect",
                files=files,
                params=params
            )
        
        return response.json()
    
    def batch_detect(self, image_paths: List[str], conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
        """
        Batch detection for multiple images
        
        Args:
            image_paths: List of image file paths
            conf: Confidence threshold
            iou: IOU threshold
        """
        files = []
        for idx, img_path in enumerate(image_paths):
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            with open(img_path, 'rb') as f:
                files.append(('files', (Path(img_path).name, f.read())))
        
        params = {
            'conf': conf,
            'iou': iou
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/detect/batch",
            files=files,
            params=params
        )
        
        return response.json()
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get supported image formats"""
        response = self.session.get(f"{self.base_url}/v1/supported-formats")
        return response.json()
    
    def save_annotated_image(self, response: Dict, output_path: str):
        """Save annotated image from response"""
        if 'image_base64' not in response or response['image_base64'] is None:
            print("No image data in response")
            return
        
        image_data = base64.b64decode(response['image_base64'])
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f"Annotated image saved to: {output_path}")


def print_detection_results(response: Dict):
    """Pretty print detection results"""
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)
    
    if not response.get('success'):
        print(f"❌ Error: {response.get('error', 'Unknown error')}")
        return
    
    print(f"✓ Status: {response['message']}")
    print(f"✓ Timestamp: {response['timestamp']}")
    
    # Image metadata
    metadata = response.get('image_metadata', {})
    print(f"\nImage Information:")
    print(f"  - File: {metadata.get('filename')}")
    print(f"  - Size: {metadata.get('size')} bytes")
    print(f"  - Dimensions: {metadata.get('width')}x{metadata.get('height')}")
    
    # Statistics
    stats = response.get('statistics', {})
    print(f"\nDetection Statistics:")
    print(f"  - Total Detections: {stats.get('total_detections', 0)}")
    if stats.get('total_detections', 0) > 0:
        print(f"  - Average Confidence: {stats.get('average_confidence', 0):.4f}")
        print(f"  - Min Confidence: {stats.get('min_confidence', 0):.4f}")
        print(f"  - Max Confidence: {stats.get('max_confidence', 0):.4f}")
        print(f"  - Detection Density: {stats.get('detection_density', 0):.4f}")
    
    # Detections
    detections = response.get('detections', [])
    if detections:
        print(f"\nDetected Diseases:")
        for idx, det in enumerate(detections, 1):
            print(f"\n  [{idx}] {det['class_name']}")
            print(f"      - Confidence: {det['confidence']:.4f}")
            print(f"      - Bounding Box: ({det['bbox']['x1']:.1f}, {det['bbox']['y1']:.1f}) → "
                  f"({det['bbox']['x2']:.1f}, {det['bbox']['y2']:.1f})")
            print(f"      - Area: {det['area_pixels']} pixels")
    else:
        print("\n✓ No diseases detected in the image")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Client untuk YOLO v12 Eye Disease Detection API"
    )
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--image', help='Image file path for detection')
    parser.add_argument('--batch', nargs='+', help='Multiple image paths for batch detection')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--save-image', help='Save annotated image to path')
    parser.add_argument('--info', action='store_true', help='Get model info')
    parser.add_argument('--health', action='store_true', help='Health check')
    
    args = parser.parse_args()
    
    client = EyeDiseaseAPIClient(args.url)
    
    # Health check
    if args.health or args.info:
        try:
            print("\n📋 Health Check:")
            health = client.health_check()
            print(json.dumps(health, indent=2))
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return
    
    # Model info
    if args.info:
        try:
            print("\n📋 Model Information:")
            info = client.get_model_info()
            print(json.dumps(info, indent=2))
            return
        except Exception as e:
            print(f"❌ Failed to get model info: {e}")
            return
    
    # Single image detection
    if args.image:
        try:
            print(f"\n🔍 Processing: {args.image}")
            response = client.detect(
                args.image,
                conf=args.conf,
                iou=args.iou,
                return_image=bool(args.save_image)
            )
            print_detection_results(response)
            
            if args.save_image:
                client.save_annotated_image(response, args.save_image)
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    # Batch detection
    elif args.batch:
        try:
            print(f"\n🔍 Batch processing {len(args.batch)} images...")
            response = client.batch_detect(args.batch, conf=args.conf, iou=args.iou)
            
            print("\n" + "=" * 80)
            print("BATCH RESULTS")
            print("=" * 80)
            print(f"✓ Total: {response['total_images']} | Processed: {response['processed_images']}")
            print(f"✓ {response['message']}")
            
            for result in response['results']:
                if result['success']:
                    print(f"  ✓ {result['filename']}: {result.get('detections', 0)} detections")
                else:
                    print(f"  ❌ {result['filename']}: {result.get('error', 'Unknown error')}")
            
            print("=" * 80)
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    else:
        print("Silakan gunakan --image atau --batch untuk menjalankan deteksi")
        print("Atau gunakan --health untuk health check")
        print("Atau gunakan --info untuk model information")


if __name__ == "__main__":
    main()
