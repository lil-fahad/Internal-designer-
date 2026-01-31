"""
Example: Test the API endpoints
"""
import sys
import os
import requests
import json
import numpy as np
from PIL import Image
import io
import base64

# API base URL
BASE_URL = 'http://localhost:5000'


def create_test_floor_plan():
    """Create a test floor plan image"""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Draw walls
    img[0:10, :] = 0
    img[-10:, :] = 0
    img[:, 0:10] = 0
    img[:, -10:] = 0
    img[250:260, :] = 0
    img[:, 250:260] = 0
    
    return img


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    response = requests.get(f'{BASE_URL}/api/health')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_generate_design():
    """Test design generation endpoint"""
    print("\n" + "="*60)
    print("Testing Design Generation")
    print("="*60)
    
    # Create test floor plan
    floor_plan = create_test_floor_plan()
    img = Image.fromarray(floor_plan)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Send request
    files = {'file': ('floor_plan.png', img_bytes, 'image/png')}
    data = {'style': 'modern', 'variations': 2}
    
    response = requests.post(
        f'{BASE_URL}/api/design/generate',
        files=files,
        data=data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Style: {result['style']}")
        print(f"Number of variations: {result['num_variations']}")
        print(f"Design images received: {len(result['designs'])}")
        
        # Optionally save designs
        for idx, design_b64 in enumerate(result['designs']):
            img_data = base64.b64decode(design_b64)
            img = Image.open(io.BytesIO(img_data))
            filename = f'test_design_{idx+1}.png'
            img.save(filename)
            print(f"  Saved: {filename}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_analyze_floor_plan():
    """Test floor plan analysis endpoint"""
    print("\n" + "="*60)
    print("Testing Floor Plan Analysis")
    print("="*60)
    
    # Create test floor plan
    floor_plan = create_test_floor_plan()
    img = Image.fromarray(floor_plan)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Send request
    files = {'file': ('floor_plan.png', img_bytes, 'image/png')}
    
    response = requests.post(
        f'{BASE_URL}/api/design/analyze',
        files=files
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Analysis Results:")
        print(json.dumps(result['analysis'], indent=2))
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_furniture_suggestions():
    """Test furniture suggestions endpoint"""
    print("\n" + "="*60)
    print("Testing Furniture Suggestions")
    print("="*60)
    
    data = {
        'room_type': 'bedroom',
        'dimensions': [5, 3, 4],
        'style': 'modern'
    }
    
    response = requests.post(
        f'{BASE_URL}/api/design/suggest',
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Room Type: {result['room_type']}")
        print(f"Style: {result['style']}")
        print(f"Suggestions:")
        for item in result['suggestions']:
            print(f"  - {item['item']} ({item.get('size', 'standard')})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_training_status():
    """Test training status endpoint"""
    print("\n" + "="*60)
    print("Testing Training Status")
    print("="*60)
    
    response = requests.get(f'{BASE_URL}/api/train/status')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def main():
    """Run all API tests"""
    print("="*60)
    print("AI Interior Designer - API Testing")
    print("="*60)
    print(f"\nBase URL: {BASE_URL}")
    print("\nMake sure the API server is running!")
    print("(Run: python app.py)")
    
    input("\nPress Enter to start tests...")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Design Generation", test_generate_design),
        ("Floor Plan Analysis", test_analyze_floor_plan),
        ("Furniture Suggestions", test_furniture_suggestions),
        ("Training Status", test_training_status),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except requests.exceptions.ConnectionError:
            print(f"\n✗ Error: Could not connect to API server at {BASE_URL}")
            print("Please make sure the server is running (python app.py)")
            return 1
        except Exception as e:
            print(f"\n✗ Error during test: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
