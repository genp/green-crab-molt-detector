#!/usr/bin/env python3
"""
Test the app with a real crab image.
"""

import requests
import sys
from pathlib import Path

def test_app_with_real_image():
    """Test the app with a real crab image."""
    
    # Find a crab image
    image_path = "/Users/gen/green_crabs/NH Green Crab Project 2016/Crabs Aug 26 - Oct 4/F1 (molted 9:23)/9:23/IMG_3324.JPG"
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return False
        
    print(f"Testing with image: {image_path}")
    
    # Make request to the app
    url = "http://localhost:5001/predict"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('crab.jpg', f, 'image/jpeg')}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print("\n✅ Prediction successful!")
                print(f"  Days until molt: {result['days_until_molt']:.1f}")
                print(f"  Phase: {result['phase']}")
                print(f"  Harvest ready: {result['harvest_ready']}")
                print(f"  Recommendation: {result['recommendation']}")
                print(f"  Confidence: {result['confidence']}")
                print(f"  Feature type: {result['feature_type']}")
                return True
            else:
                print(f"\n❌ Prediction failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to app. Make sure the app is running on port 5001")
        print("Start the app with: python app.py")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_app_with_real_image()
    sys.exit(0 if success else 1)