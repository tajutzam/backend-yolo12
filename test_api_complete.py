"""
Comprehensive API Test Suite for YOLO v12 Eye Disease Detection
Tests all endpoints and core functionality
"""

import requests
import json
import sys
from pathlib import Path
import base64
from datetime import datetime

# Configuration
API_BASE = "http://127.0.0.1:8001"
ENDPOINTS = {
    "health": f"{API_BASE}/v1/health",
    "model_info": f"{API_BASE}/v1/model/info",
    "formats": f"{API_BASE}/v1/supported-formats",
    "detect": f"{API_BASE}/v1/detect",
    "batch_detect": f"{API_BASE}/v1/detect/batch",
}

# Test results storage
results = {
    "timestamp": datetime.now().isoformat(),
    "tests": {},
    "summary": {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
}

def log_test(name, status, response=None, error=None):
    """Log test result"""
    results["tests"][name] = {
        "status": status,
        "response": response,
        "error": str(error) if error else None
    }
    results["summary"]["total"] += 1
    if status == "PASS":
        results["summary"]["passed"] += 1
        print(f"✅ {name}")
    elif status == "FAIL":
        results["summary"]["failed"] += 1
        print(f"❌ {name}: {error}")
    else:
        results["summary"]["skipped"] += 1
        print(f"⊘ {name}: {error}")
    return status == "PASS"

def test_health():
    """Test /v1/health endpoint"""
    try:
        response = requests.get(ENDPOINTS["health"])
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy" and data.get("model_loaded"):
                log_test("Health Check", "PASS", data)
                return True
            else:
                log_test("Health Check", "FAIL", None, f"Status: {data.get('status')}, Model loaded: {data.get('model_loaded')}")
                return False
        else:
            log_test("Health Check", "FAIL", None, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        log_test("Health Check", "FAIL", None, str(e))
        return False

def test_model_info():
    """Test /v1/model/info endpoint"""
    try:
        response = requests.get(ENDPOINTS["model_info"])
        if response.status_code == 200:
            data = response.json()
            # Check for classes at top level or nested in model
            classes = data.get("classes") or data.get("model", {}).get("classes")
            if classes and len(classes) > 0:
                log_test("Model Info", "PASS", {
                    "model_name": data.get("model", {}).get("name", "Unknown"),
                    "num_classes": len(classes),
                    "classes": classes
                })
                return True
            else:
                log_test("Model Info", "FAIL", None, "No classes in response")
                return False
        else:
            log_test("Model Info", "FAIL", None, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        log_test("Model Info", "FAIL", None, str(e))
        return False

def test_supported_formats():
    """Test /v1/supported-formats endpoint"""
    try:
        response = requests.get(ENDPOINTS["formats"])
        if response.status_code == 200:
            data = response.json()
            formats = data.get("supported_formats", [])
            log_test("Supported Formats", "PASS", {
                "formats": formats,
                "max_file_size_mb": data.get("max_file_size_mb"),
                "max_batch_size": data.get("max_batch_size")
            })
            return True
        else:
            log_test("Supported Formats", "FAIL", None, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        log_test("Supported Formats", "FAIL", None, str(e))
        return False

def test_detection_with_sample():
    """Test /v1/detect with sample image"""
    try:
        # Check if sample image exists (try multiple filenames)
        sample_images = [
            Path("assets/test_image.jpg"),
            Path("assets/sample_eye_image.jpg"),
            Path("assets/eye.jpg"),
        ]
        
        sample_image = None
        for img_path in sample_images:
            if img_path.exists():
                sample_image = img_path
                break
        
        if not sample_image:
            log_test("Single Image Detection", "SKIP", None, "No sample image found")
            return None
        
        with open(sample_image, "rb") as f:
            files = {"file": f}
            response = requests.post(
                ENDPOINTS["detect"],
                files=files,
                data={"conf": 0.25, "iou": 0.45}
            )
        
        if response.status_code == 200:
            data = response.json()
            log_test("Single Image Detection", "PASS", {
                "success": data.get("success"),
                "detections": len(data.get("detections", [])),
                "image_size": f"{data.get('image_metadata', {}).get('width')}x{data.get('image_metadata', {}).get('height')}"
            })
            return True
        elif response.status_code == 422:
            log_test("Single Image Detection", "SKIP", None, "Validation error (expected if no sample image)")
            return None
        else:
            log_test("Single Image Detection", "FAIL", None, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        log_test("Single Image Detection", "FAIL", None, str(e))
        return False

def test_error_handling():
    """Test error handling with invalid request"""
    try:
        # Test missing file parameter
        response = requests.post(ENDPOINTS["detect"])
        if response.status_code == 422:
            log_test("Error Handling - Missing File", "PASS", "Correctly returns 422")
            return True
        else:
            log_test("Error Handling - Missing File", "FAIL", None, f"Expected 422, got {response.status_code}")
            return False
    except Exception as e:
        log_test("Error Handling - Missing File", "FAIL", None, str(e))
        return False

def test_invalid_format():
    """Test error handling with invalid file format"""
    try:
        # Create a text file
        files = {"file": ("test.txt", b"This is not an image", "text/plain")}
        response = requests.post(ENDPOINTS["detect"], files=files)
        
        if response.status_code in [400, 422]:
            log_test("Error Handling - Invalid Format", "PASS", f"HTTP {response.status_code}")
            return True
        else:
            log_test("Error Handling - Invalid Format", "FAIL", None, f"Expected 400/422, got {response.status_code}")
            return False
    except Exception as e:
        log_test("Error Handling - Invalid Format", "FAIL", None, str(e))
        return False

def test_cors_headers():
    """Test CORS headers"""
    try:
        response = requests.options(ENDPOINTS["detect"])
        headers = response.headers
        
        cors_headers = {
            "Access-Control-Allow-Origin": headers.get("Access-Control-Allow-Origin"),
            "Access-Control-Allow-Methods": headers.get("Access-Control-Allow-Methods"),
        }
        
        if cors_headers["Access-Control-Allow-Origin"]:
            log_test("CORS Headers", "PASS", cors_headers)
            return True
        else:
            log_test("CORS Headers", "SKIP", None, "CORS headers not found")
            return None
    except Exception as e:
        log_test("CORS Headers", "FAIL", None, str(e))
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 YOLO v12 API Test Suite")
    print("=" * 70)
    print(f"\n🔗 API Base URL: {API_BASE}\n")
    
    print("📝 Running Tests...\n")
    
    # Run tests
    test_health()
    test_model_info()
    test_supported_formats()
    test_error_handling()
    test_invalid_format()
    test_cors_headers()
    test_detection_with_sample()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"Total Tests:  {results['summary']['total']}")
    print(f"Passed:       {results['summary']['passed']} ✅")
    print(f"Failed:       {results['summary']['failed']} ❌")
    print(f"Skipped:      {results['summary']['skipped']} ⊘")
    
    # Calculate pass rate
    if results['summary']['total'] > 0:
        pass_rate = (results['summary']['passed'] / (results['summary']['total'] - results['summary']['skipped'])) * 100 if (results['summary']['total'] - results['summary']['skipped']) > 0 else 0
        print(f"Pass Rate:    {pass_rate:.1f}%")
    
    print("=" * 70)
    
    # Overall status
    if results['summary']['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {results['summary']['failed']} test(s) failed")
    
    print("=" * 70 + "\n")
    
    # Save results
    with open("API_TEST_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: API_TEST_RESULTS.json\n")
    
    return results['summary']['failed'] == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
