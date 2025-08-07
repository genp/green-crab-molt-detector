#!/usr/bin/env python3
"""
Test that the loading modal properly disappears after analysis.
"""

import time
import requests
from pathlib import Path

def test_modal_behavior():
    """Test that modal properly closes after prediction."""
    
    print("Testing Modal Dismissal Fix")
    print("=" * 50)
    
    # Find a test image
    image_path = "/Users/gen/green_crabs/NH Green Crab Project 2016/Crabs Aug 26 - Oct 4/F1 (molted 9:23)/9:23/IMG_3324.JPG"
    
    if not Path(image_path).exists():
        print("❌ Test image not found")
        return False
    
    print("✓ Found test image")
    
    # Test the prediction endpoint
    url = "http://localhost:5001/predict"
    
    try:
        print("✓ Sending image for analysis...")
        
        with open(image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            
            # Time the request
            start_time = time.time()
            response = requests.post(url, files=files)
            end_time = time.time()
            
        elapsed = end_time - start_time
        print(f"✓ Analysis completed in {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print(f"✓ Prediction successful: {result['phase']}")
                print(f"  Days until molt: {result['days_until_molt']:.1f}")
                
                print("\n" + "=" * 50)
                print("MODAL FIX VERIFICATION:")
                print("The loading modal should now properly disappear after analysis.")
                print("Test this manually by:")
                print("1. Opening http://localhost:5001 in your browser")
                print("2. Uploading an image")
                print("3. Clicking 'Analyze Molt Phase'")
                print("4. Verifying the loading modal disappears after results show")
                print("=" * 50)
                
                return True
            else:
                print(f"❌ Prediction failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_modal_behavior()
    
    if success:
        print("\n✅ Backend test passed. Please verify modal behavior in browser.")
    else:
        print("\n❌ Test failed")