#!/usr/bin/env python3
"""
Run the Green Crab Molt Detection web application.
This script assumes pre-trained models exist and starts the Flask app directly.
"""

import os
import sys
import pickle
import subprocess
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    # Map of import names to package names - only check essential ones
    essential_packages = {
        'flask': 'Flask',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
    }
    
    optional_packages = {
        'torch': 'PyTorch',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib'
    }
    
    missing_essential = []
    missing_optional = []
    
    # Check essential packages
    for import_name, display_name in essential_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            missing_essential.append(display_name)
            print(f"  ✗ {display_name}")
    
    # Check optional packages (warn but don't fail)
    for import_name, display_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {display_name}")
        except Exception:
            missing_optional.append(display_name)
            print(f"  ⚠ {display_name} (optional)")
    
    if missing_essential:
        print(f"\n❌ Missing essential packages: {', '.join(missing_essential)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("   The app will run but some features may be limited.")
    
    print("\n✅ Core dependencies satisfied")
    return True


def check_models():
    """Check if pre-trained models exist."""
    print("\nChecking for pre-trained models...")
    
    models_found = {}
    
    # Check for main molt prediction models - using actual file names
    model_paths = {
        'YOLO Regressor': 'models/best_yolo_regressor.joblib',
        'CNN Regressor': 'models/best_cnn_regressor.joblib',
        'ViT Regressor': 'models/best_vit_regressor.joblib',
        'Best Temporal': 'models/best_temporal_model.pkl',
        'Random Forest': 'models/random_forest_model.joblib',
        'Gradient Boosting': 'models/gradient_boosting_model.joblib',
    }
    
    for name, path in model_paths.items():
        if Path(path).exists():
            models_found[name] = path
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ⚠️  {name}: Not found at {path}")
    
    if not models_found:
        print("\n❌ No pre-trained models found!")
        print("Please run 'python run_pipeline.py' first to train models.")
        return False
    
    # Check for feature extractor
    if Path('models/vit_model.pkl').exists():
        print("\n✅ ViT feature extractor found")
    elif Path('models/yolo_features.pkl').exists():
        print("\n✅ YOLO feature extractor found")
    else:
        print("\n⚠️  No feature extractor found - using pre-computed features")
    
    return True


def check_data():
    """Check if processed data exists."""
    print("\nChecking for processed data...")
    
    data_files = {
        'Dataset': 'data/processed/crab_dataset.csv',
        'ViT Features': 'data/processed/vit_features.npy',
        'CNN Features': 'data/processed/cnn_features.npy',
        'YOLO Features': 'data/processed/yolo_features.npy'
    }
    
    data_found = False
    for name, path in data_files.items():
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
            data_found = True
        else:
            print(f"  ⚠️  {name}: Not found")
    
    if not data_found:
        print("\n⚠️  No processed features found")
        print("The app will extract features on-the-fly for uploaded images")
    
    return True


def load_best_model():
    """Load and display information about the best model."""
    print("\nLoading best model...")
    
    best_model_path = Path("models/best_temporal_model.pkl")
    if best_model_path.exists():
        with open(best_model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Loaded: {model_data.get('name', 'Best Temporal Model')}")
        
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            print(f"   Test MAE: {metrics.get('test_mae', 'N/A'):.2f} days")
            print(f"   Test R²: {metrics.get('test_r2', 'N/A'):.3f}")
        
        return model_data
    
    # Fallback to any available model
    for model_file in Path("models").glob("*.joblib"):
        print(f"✅ Using fallback model: {model_file.name}")
        return None
    
    for model_file in Path("models").glob("*.pkl"):
        print(f"✅ Using fallback model: {model_file.name}")
        return None
    
    print("❌ No models found!")
    return None


def start_app():
    """Start the Flask web application."""
    print("\n" + "="*60)
    print("Starting Green Crab Molt Detection App")
    print("="*60)
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found!")
        print("Please ensure app.py is in the current directory.")
        return False
    
    print("\n🚀 Starting Flask application...")
    print("="*60)
    print("Access the app at: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Get test images for user reference
    print("Test images you can use:")
    print("-" * 40)
    test_images = [
        "NH Green Crab Project 2016/Crabs Aug 26 - Oct 4/F2 (molted 9:20)/9:9/thumb_IMG_3097_1024.jpg",
        "NH Green Crab Project 2016/Crabs Aug 26 - Oct 4/F1 (molted 9:23)/9:8/thumb_IMG_3027_1024.jpg",
        "NH Green Crab Project 2016/Crabs Aug 26 - Oct 4/M7/9:9/thumb_IMG_3069_1024.jpg",
        "NH Green Crab Project 2016/Crabs Aug 26 - Oct 4/F9 (molted 9:14)/9:8/thumb_IMG_3012_1024.jpg"
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"  • {img_path}")
    
    print("\n" + "="*60 + "\n")
    
    try:
        # Run the Flask app
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running app: {e}")
        return False
    
    return True


def main():
    """Main execution function."""
    print("="*60)
    print("Green Crab Molt Detection - App Runner")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Run checks
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        sys.exit(1)
    
    if not check_models():
        print("\n❌ Please train models first using 'python run_pipeline.py'")
        sys.exit(1)
    
    check_data()
    
    # Load best model info
    best_model = load_best_model()
    if not best_model:
        print("\n⚠️  Warning: No optimal model found")
        print("The app will use whatever models are available")
    
    # Start the app
    if not start_app():
        print("\n❌ Failed to start the application")
        sys.exit(1)
    
    print("\n✅ Application closed successfully")


if __name__ == "__main__":
    main()