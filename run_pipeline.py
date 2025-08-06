#!/usr/bin/env python
"""
Complete pipeline runner for green crab molt detection system.

This script runs the entire pipeline with smart caching:
1. Data loading and preprocessing
2. Feature extraction (cached)
3. t-SNE visualization  
4. Model training (cached)
5. Web app preparation

Features:
- Caches extracted features to avoid re-extraction
- Caches trained models to avoid re-training
- Smart cache checking based on file timestamps
"""

import subprocess
import sys
from pathlib import Path
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def check_cache_validity(cache_files: list, source_files: list) -> bool:
    """
    Check if cached files are valid (exist and newer than source files).
    
    Args:
        cache_files: List of cache file paths to check
        source_files: List of source file paths to compare against
        
    Returns:
        True if all cache files exist and are newer than all source files
    """
    if not cache_files or not source_files:
        return False
        
    # Check if all cache files exist
    for cache_file in cache_files:
        if not Path(cache_file).exists():
            return False
    
    # Find newest source file timestamp
    newest_source = 0
    for source_file in source_files:
        if Path(source_file).exists():
            newest_source = max(newest_source, Path(source_file).stat().st_mtime)
    
    # Find oldest cache file timestamp
    oldest_cache = float('inf')
    for cache_file in cache_files:
        oldest_cache = min(oldest_cache, Path(cache_file).stat().st_mtime)
    
    # Cache is valid if oldest cache file is newer than newest source file
    return oldest_cache > newest_source


def check_features_cache() -> dict:
    """Check which feature files are cached and valid."""
    cache_dir = Path("data/processed")
    
    # Define cache files
    cache_files = {
        'yolo': cache_dir / "yolo_features.npy",
        'cnn': cache_dir / "cnn_features.npy", 
        'vit': cache_dir / "vit_features.npy",
        'dataset': cache_dir / "crab_dataset.csv",
        'tsne': cache_dir / "yolo_tsne_embedding.npy"
    }
    
    # Define source files (if any change, we need to re-extract)
    source_files = [
        "src/feature_extractor.py",
        "src/data_loader.py", 
        "run_feature_analysis.py"
    ]
    
    # Check individual cache validity
    cache_status = {}
    for feature_type, cache_file in cache_files.items():
        cache_status[feature_type] = cache_file.exists()
        if cache_status[feature_type]:
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            logger.info(f"✓ {feature_type} features cached ({mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    # Overall cache validity
    all_cached = all(cache_status.values())
    cache_valid = check_cache_validity(list(cache_files.values()), source_files)
    
    return {
        'all_cached': all_cached,
        'cache_valid': cache_valid,
        'individual_status': cache_status,
        'cache_files': cache_files
    }


def check_models_cache() -> dict:
    """Check which models are cached and valid."""
    models_dir = Path("models")
    
    # Define model files for each feature type
    feature_types = ['yolo', 'cnn', 'vit', 'combined']
    model_files = {}
    scaler_files = {}
    result_files = {}
    
    for ft in feature_types:
        model_files[ft] = models_dir / f"best_{ft}_regressor.joblib"
        scaler_files[ft] = models_dir / f"{ft}_scaler.joblib" 
        result_files[ft] = models_dir / f"{ft}_results.csv"
    
    # Define source files (if any change, we need to re-train)
    source_files = [
        "src/model.py" if Path("src/model.py").exists() else None,
        "train_model.py",
        "data/processed/yolo_features.npy",
        "data/processed/cnn_features.npy",
        "data/processed/vit_features.npy",
        "data/processed/crab_dataset.csv"
    ]
    source_files = [f for f in source_files if f and Path(f).exists()]
    
    # Check cache status
    cache_status = {}
    for ft in feature_types:
        model_exists = model_files[ft].exists()
        scaler_exists = scaler_files[ft].exists()
        results_exist = result_files[ft].exists()
        
        cache_status[ft] = model_exists and scaler_exists and results_exist
        
        if cache_status[ft]:
            mod_time = datetime.fromtimestamp(model_files[ft].stat().st_mtime)
            logger.info(f"✓ {ft} model cached ({mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    # Overall cache validity
    all_files = list(model_files.values()) + list(scaler_files.values()) + list(result_files.values())
    existing_files = [f for f in all_files if f.exists()]
    cache_valid = check_cache_validity(existing_files, source_files) if existing_files else False
    
    return {
        'any_cached': any(cache_status.values()),
        'cache_valid': cache_valid,
        'individual_status': cache_status
    }


def run_command(command: str, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    logger.info('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error code {e.returncode}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False


def check_requirements():
    """Check if required files and directories exist."""
    logger.info("Checking requirements...")
    
    issues = []
    
    # Check for YOLO model
    yolo_path = Path("/Users/gen/BarderryAppliedResearch/FathomNet/qscp/jupyter_notebooks/fathomverse_detector/fathomverse-only-imgs_update_to_FathomNet-NoGameLabels-2024-09-28-model_yolo8_epochs_10_2024-10-22.pt")
    if not yolo_path.exists():
        issues.append(f"YOLO model not found at: {yolo_path}")
        logger.warning("YOLO model not found - will use CNN features instead")
    
    # Check for image data
    data_path = Path("NH Green Crab Project 2016")
    if not data_path.exists():
        issues.append(f"Image data directory not found: {data_path}")
    
    if issues:
        logger.warning("Some requirements are missing:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
            
    return True


def main():
    """Run the complete pipeline with smart caching."""
    logger.info("Green Crab Molt Detection Pipeline (with Smart Caching)")
    logger.info("=" * 65)
    
    # Check requirements
    if not check_requirements():
        logger.error("Exiting due to missing requirements")
        return 1
    
    # Create necessary directories
    for dir_name in ['data/processed', 'models', 'plots', 'temp_uploads']:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Check feature cache status
    logger.info("\n🗂️  Checking feature cache...")
    feature_cache = check_features_cache()
    
    # Step 1: Feature extraction and visualization
    if feature_cache['all_cached'] and feature_cache['cache_valid']:
        logger.info("✅ All features cached and valid - skipping extraction")
        logger.info("   Cached features: YOLO, CNN, ViT, Dataset, t-SNE embedding")
    else:
        logger.info("🔄 Running feature extraction...")
        if feature_cache['all_cached'] and not feature_cache['cache_valid']:
            logger.info("   (Cache exists but source files newer - re-extracting)")
        
        # Set environment variable for MPS fallback
        import os
        env = os.environ.copy()
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        if not run_command(
            "PYTORCH_ENABLE_MPS_FALLBACK=1 python run_feature_analysis.py",
            "Feature extraction and t-SNE visualization"
        ):
            logger.error("Feature extraction failed")
            return 1
    
    # Check model cache status
    logger.info("\n🤖 Checking model cache...")
    model_cache = check_models_cache()
    
    # Step 2: Model training
    if model_cache['any_cached'] and model_cache['cache_valid']:
        logger.info("✅ Models cached and valid - skipping training")
        cached_models = [k for k, v in model_cache['individual_status'].items() if v]
        logger.info(f"   Cached models: {', '.join(cached_models)}")
    else:
        logger.info("🔄 Running model training...")
        if model_cache['any_cached'] and not model_cache['cache_valid']:
            logger.info("   (Cache exists but features/code newer - re-training)")
        
        if not run_command(
            "python train_model.py",
            "Training molt phase regression models"
        ):
            logger.error("Model training failed")
            return 1
    
    # Step 3: Prepare deployment (always run - lightweight)
    if not run_command(
        "python deploy.py",
        "Creating deployment files"
    ):
        logger.error("Deployment preparation failed")
        return 1
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)
    
    logger.info("\nGenerated files:")
    logger.info("- data/processed/: Dataset and extracted features")
    logger.info("- plots/: t-SNE visualizations and model comparisons")
    logger.info("- models/: Trained regression models")
    logger.info("- Deployment files: Dockerfile, DEPLOYMENT.md, etc.")
    
    logger.info("\nNext steps:")
    logger.info("1. Review the plots in the plots/ directory")
    logger.info("2. Test the web app locally: python app.py")
    logger.info("3. Deploy using instructions in DEPLOYMENT.md")
    
    # Check if models were successfully trained
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.joblib"))
    if model_files:
        logger.info(f"\nTrained models found: {len(model_files)}")
        for model_file in model_files:
            logger.info(f"  - {model_file.name}")
    else:
        logger.warning("\nNo trained models found - check training logs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
