#!/usr/bin/env python3
"""
Fix model files and ensure they have proper scalers.
"""

import joblib
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_fix_models():
    """Check all model files and fix scaler issues."""
    models_dir = Path("/Users/gen/green_crabs/models")
    
    # Check for model files with their corresponding scalers
    model_scaler_pairs = [
        ("best_yolo_regressor.joblib", "yolo_scaler.joblib"),
        ("best_cnn_regressor.joblib", "cnn_scaler.joblib"),
        ("best_vit_regressor.joblib", "vit_scaler.joblib"),
    ]
    
    fixed_models = []
    
    for model_file, scaler_file in model_scaler_pairs:
        model_path = models_dir / model_file
        scaler_path = models_dir / scaler_file
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            continue
            
        if not scaler_path.exists():
            logger.warning(f"Scaler file not found: {scaler_path}")
            continue
            
        logger.info(f"Checking {model_file}...")
        
        try:
            # Load model
            model_data = joblib.load(model_path)
            
            # Check if it's in the old format (just the model object)
            if not isinstance(model_data, dict):
                logger.info(f"  Converting {model_file} to new format...")
                
                # Load the corresponding scaler
                scaler = joblib.load(scaler_path)
                
                # Create new format
                new_model_data = {
                    'algorithm': 'random_forest',  # Most of our models are RF
                    'model': model_data,
                    'scaler': scaler,
                    'is_fitted': True
                }
                
                # Save with new format
                output_name = model_file.replace('best_', 'molt_regressor_').replace('_regressor', '')
                if 'yolo' in model_file:
                    output_name = 'molt_regressor_yolo_random_forest.joblib'
                elif 'cnn' in model_file:
                    output_name = 'molt_regressor_cnn_random_forest.joblib'
                elif 'vit' in model_file:
                    output_name = 'molt_regressor_vit_random_forest.joblib'
                    
                output_path = models_dir / output_name
                joblib.dump(new_model_data, output_path)
                logger.info(f"  Saved fixed model to {output_path}")
                fixed_models.append(output_path)
                
            else:
                # Check if scaler is included
                if 'scaler' not in model_data or model_data['scaler'] is None:
                    logger.info(f"  Adding scaler to {model_file}...")
                    scaler = joblib.load(scaler_path)
                    model_data['scaler'] = scaler
                    joblib.dump(model_data, model_path)
                    logger.info(f"  Updated {model_path} with scaler")
                    fixed_models.append(model_path)
                else:
                    logger.info(f"  {model_file} already has scaler")
                    
        except Exception as e:
            logger.error(f"Error processing {model_file}: {e}")
            
    # Create expected files for the app
    logger.info("\nCreating expected model files for app...")
    
    # The app looks for molt_regressor_{feature_type}_*.joblib
    # Let's make sure we have them
    expected_files = [
        ("molt_regressor_yolo_random_forest.joblib", "best_yolo_regressor.joblib", "yolo_scaler.joblib"),
        ("molt_regressor_cnn_random_forest.joblib", "best_cnn_regressor.joblib", "cnn_scaler.joblib"),
        ("molt_regressor_vit_random_forest.joblib", "best_vit_regressor.joblib", "vit_scaler.joblib"),
    ]
    
    for expected_file, source_model, source_scaler in expected_files:
        expected_path = models_dir / expected_file
        
        if not expected_path.exists():
            source_model_path = models_dir / source_model
            source_scaler_path = models_dir / source_scaler
            
            if source_model_path.exists() and source_scaler_path.exists():
                logger.info(f"Creating {expected_file}...")
                
                model = joblib.load(source_model_path)
                scaler = joblib.load(source_scaler_path)
                
                # Create proper format
                model_data = {
                    'algorithm': 'random_forest',
                    'model': model,
                    'scaler': scaler,
                    'is_fitted': True
                }
                
                joblib.dump(model_data, expected_path)
                logger.info(f"  Created {expected_path}")
                
    return fixed_models

if __name__ == "__main__":
    logger.info("Fixing model files...")
    fixed = check_and_fix_models()
    logger.info(f"\nFixed {len(fixed)} model files")
    logger.info("\nDone!")