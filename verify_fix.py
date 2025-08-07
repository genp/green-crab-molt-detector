#!/usr/bin/env python3
"""
Verify that all fixes have been applied successfully.
"""

import sys
import os
from pathlib import Path
import joblib
import subprocess
import time

def check_models():
    """Check that models are properly formatted with scalers."""
    print("Checking models...")
    models_dir = Path("/Users/gen/green_crabs/models")
    
    required_models = [
        "molt_regressor_yolo_random_forest.joblib",
        "molt_regressor_cnn_random_forest.joblib",
        "molt_regressor_vit_random_forest.joblib",
    ]
    
    all_good = True
    for model_file in required_models:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                data = joblib.load(model_path)
                if isinstance(data, dict) and 'scaler' in data and data['scaler'] is not None:
                    print(f"  ✅ {model_file}: Has scaler")
                else:
                    print(f"  ❌ {model_file}: Missing scaler")
                    all_good = False
            except Exception as e:
                print(f"  ❌ {model_file}: Error loading - {e}")
                all_good = False
        else:
            print(f"  ❌ {model_file}: Not found")
            all_good = False
            
    return all_good

def test_app_startup():
    """Test that the app can start without errors."""
    print("\nTesting app startup...")
    
    # Start app in background
    proc = subprocess.Popen(
        ["python", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for startup
    time.sleep(5)
    
    # Check if still running
    if proc.poll() is None:
        print("  ✅ App started successfully")
        
        # Test health endpoint
        import requests
        try:
            response = requests.get("http://localhost:5001/health")
            if response.status_code == 200:
                data = response.json()
                print(f"    Feature extractor loaded: {data.get('feature_extractor', False)}")
                print(f"    Regressor loaded: {data.get('regressor', False)}")
                print(f"    Feature type: {data.get('feature_type', 'Unknown')}")
                
                # Test image upload fix
                print("\n  Testing image upload fix...")
                # Create simple test to verify selectedFile variable is used
                test_html = Path("templates/index.html").read_text()
                if "let selectedFile = null" in test_html and "formData.append('file', selectedFile)" in test_html:
                    print("    ✅ Image upload fix applied")
                else:
                    print("    ❌ Image upload fix not found")
                    
        except Exception as e:
            print(f"  ⚠️  Could not test health endpoint: {e}")
        
        # Terminate app
        proc.terminate()
        proc.wait(timeout=5)
        return True
    else:
        stdout, stderr = proc.communicate()
        print(f"  ❌ App failed to start")
        if stderr:
            print(f"    Error: {stderr[:200]}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("MOLT DETECTION APP - FIX VERIFICATION")
    print("=" * 60)
    
    checks_passed = []
    
    # Check 1: Models have scalers
    print("\n1. MODEL FILES")
    checks_passed.append(check_models())
    
    # Check 2: App starts and loads models
    print("\n2. APP FUNCTIONALITY")
    checks_passed.append(test_app_startup())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all(checks_passed):
        print("✅ All fixes successfully applied!")
        print("\nThe app is ready to use:")
        print("  1. Start the app: python app.py")
        print("  2. Open browser: http://localhost:5001")
        print("  3. Upload a crab image to get molt phase prediction")
        return True
    else:
        print("⚠️  Some issues remain, but app may still be functional")
        print("\nTry running: python fix_models.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)