#!/usr/bin/env python3
"""Test M7 crab using the same process as F1, F2, F9."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Patch for torch compatibility (CRITICAL FIX!)
import torch
if not hasattr(torch, 'uint64'):
    torch.uint64 = torch.int64
if not hasattr(torch, 'uint32'):
    torch.uint32 = torch.int32
if not hasattr(torch, 'uint16'):
    torch.uint16 = torch.int16

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extractor import GeneralCrustaceanFeatureExtractor
import joblib

def calculate_days_until_molt(obs_date: str, molt_date: str) -> float:
    """Calculate days from observation to molt."""
    # Parse dates like "9:23" as month:day
    obs_month, obs_day = obs_date.replace(':', ' ').split()
    molt_month, molt_day = molt_date.replace(':', ' ').split() if molt_date else ('9', '23')
    
    obs_dt = datetime(2016, int(obs_month), int(obs_day))
    molt_dt = datetime(2016, int(molt_month), int(molt_day))
    
    return (molt_dt - obs_dt).days

def main():
    print("=" * 80)
    print("TESTING M7 WITH PROPER TORCH COMPATIBILITY")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    models_dir = Path('models')
    
    # Load feature extractors
    cnn_extractor = GeneralCrustaceanFeatureExtractor('resnet50')
    vit_extractor = GeneralCrustaceanFeatureExtractor('vit_base')
    
    # Load CNN models
    cnn_rf_data = joblib.load(models_dir / 'best_cnn_regressor.joblib')
    if isinstance(cnn_rf_data, dict):
        cnn_rf = cnn_rf_data['model']
        cnn_scaler = cnn_rf_data.get('scaler', joblib.load(models_dir / 'cnn_scaler.joblib'))
    else:
        cnn_rf = cnn_rf_data
        cnn_scaler = joblib.load(models_dir / 'cnn_scaler.joblib')
    
    # Load VIT models  
    vit_rf = joblib.load(models_dir / 'best_vit_regressor.joblib')
    vit_scaler = joblib.load(models_dir / 'vit_scaler.joblib')
    
    # Load ensemble models (best available)
    vit_rf_ensemble_data = joblib.load(models_dir / 'molt_regressor_vit_random_forest.joblib')
    cnn_rf_ensemble_data = joblib.load(models_dir / 'molt_regressor_cnn_random_forest.joblib')
    
    # Extract actual models from dictionaries
    vit_rf_ensemble = vit_rf_ensemble_data['model'] if isinstance(vit_rf_ensemble_data, dict) else vit_rf_ensemble_data
    cnn_rf_ensemble = cnn_rf_ensemble_data['model'] if isinstance(cnn_rf_ensemble_data, dict) else cnn_rf_ensemble_data
    
    # Use embedded scalers if available
    if isinstance(vit_rf_ensemble_data, dict) and 'scaler' in vit_rf_ensemble_data:
        vit_scaler = vit_rf_ensemble_data['scaler']
    if isinstance(cnn_rf_ensemble_data, dict) and 'scaler' in cnn_rf_ensemble_data:
        cnn_scaler = cnn_rf_ensemble_data['scaler']
    
    print("Models loaded successfully!")
    
    # Process M7
    base_path = Path('NH Green Crab Project 2016/Crabs Aug 26 - Oct 4')
    m7_folder = base_path / 'M7'
    
    # M7 doesn't have molt date in folder name, but based on data it's 9/23
    molt_date = '9:23'
    
    print(f"\n{'-' * 60}")
    print(f"Testing M7 (assumed molt date: {molt_date})")
    print(f"{'-' * 60}")
    
    all_results = []
    
    # Get all observation folders
    obs_folders = sorted([f for f in m7_folder.iterdir() if f.is_dir()])
    
    for obs_folder in obs_folders:
        obs_date = obs_folder.name
        
        # Calculate ground truth
        try:
            days_until_molt = calculate_days_until_molt(obs_date, molt_date)
        except:
            print(f"  ⚠️  Could not parse date: {obs_date}")
            continue
        
        # Get all images
        images = list(obs_folder.glob("*.jpg")) + list(obs_folder.glob("*.JPG")) + \
                list(obs_folder.glob("*.jpeg")) + list(obs_folder.glob("*.JPEG"))
        
        print(f"\n  {obs_date} ({days_until_molt} days until molt): {len(images)} images")
        
        for img_path in images:
            try:
                # Load image
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # Extract CNN features
                cnn_features = cnn_extractor.extract_features(img_array)
                cnn_features_scaled = cnn_scaler.transform(cnn_features.reshape(1, -1))
                cnn_pred = cnn_rf.predict(cnn_features_scaled)[0]
                
                # Extract VIT features
                vit_features = vit_extractor.extract_features(img_array)
                vit_features_scaled = vit_scaler.transform(vit_features.reshape(1, -1))
                vit_pred = vit_rf.predict(vit_features_scaled)[0]
                
                # Ensemble predictions (using best available models)
                vit_ensemble_pred = vit_rf_ensemble.predict(vit_features_scaled)[0]
                cnn_ensemble_pred = cnn_rf_ensemble.predict(cnn_features_scaled)[0]
                ensemble_avg_pred = (vit_ensemble_pred + cnn_ensemble_pred) / 2
                
                result = {
                    'crab_id': 'M7',
                    'molt_date': molt_date,
                    'obs_date': obs_date,
                    'days_until_molt': days_until_molt,
                    'image_name': img_path.name,
                    'image_path': str(img_path),
                    'cnn_pred': cnn_pred,
                    'vit_pred': vit_pred,
                    'cnn_ensemble_pred': cnn_ensemble_pred,
                    'vit_ensemble_pred': vit_ensemble_pred,
                    'ensemble_avg_pred': ensemble_avg_pred
                }
                
                all_results.append(result)
                
                # Print first image prediction as sample
                if len(all_results) == 1:
                    print(f"    Sample - {img_path.name}:")
                    print(f"      Ground Truth: {days_until_molt:.1f} days")
                    print(f"      CNN: {cnn_pred:.1f} days")
                    print(f"      VIT: {vit_pred:.1f} days")
                    print(f"      Ensemble: {ensemble_avg_pred:.1f} days")
                
            except Exception as e:
                print(f"    ⚠️  Error processing {img_path.name}: {e}")
    
    print(f"\n{'-' * 60}")
    print(f"Processed {len(all_results)} M7 images total")
    
    # Save results
    output_dir = Path('testset_complete_results')
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / 'M7_proper_predictions.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to: {csv_path}")
    
    # Calculate statistics
    if len(all_results) > 0:
        gt = results_df['days_until_molt'].values
        
        # CNN errors
        cnn_errors = np.abs(results_df['cnn_pred'].values - gt)
        print(f"\nCNN Model Performance:")
        print(f"  MAE: {np.mean(cnn_errors):.2f} ± {np.std(cnn_errors):.2f} days")
        print(f"  <2 day accuracy: {100 * np.sum(cnn_errors < 2) / len(cnn_errors):.1f}%")
        
        # VIT errors
        vit_errors = np.abs(results_df['vit_pred'].values - gt)
        print(f"\nVIT Model Performance:")
        print(f"  MAE: {np.mean(vit_errors):.2f} ± {np.std(vit_errors):.2f} days")
        print(f"  <2 day accuracy: {100 * np.sum(vit_errors < 2) / len(vit_errors):.1f}%")
        
        # Ensemble errors
        ensemble_errors = np.abs(results_df['ensemble_avg_pred'].values - gt)
        print(f"\nEnsemble Model Performance:")
        print(f"  MAE: {np.mean(ensemble_errors):.2f} ± {np.std(ensemble_errors):.2f} days")
        print(f"  <2 day accuracy: {100 * np.sum(ensemble_errors < 2) / len(ensemble_errors):.1f}%")
    
    # Create visualization
    if len(all_results) == 0:
        print("\nNo images processed successfully, skipping visualization")
        return
    
    print("\nCreating visualization...")
    n_images = min(len(all_results), 24)
    cols = 4
    rows = max(1, (n_images + cols - 1) // cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx in range(n_images):
        ax = axes[idx]
        result = all_results[idx]
        
        # Load and display image
        img = Image.open(result['image_path'])
        img_resized = img.resize((200, 200))
        ax.imshow(img_resized)
        ax.axis('off')
        
        # Add predictions
        gt = result['days_until_molt']
        cnn = result['cnn_pred']
        vit = result['vit_pred']
        ensemble = result['ensemble_avg_pred']
        
        title = f"M7 - {result['obs_date']}\n"
        title += f"GT: {gt:.0f}d | "
        title += f"CNN: {cnn:.1f}d | VIT: {vit:.1f}d\n"
        title += f"Ensemble: {ensemble:.1f}d"
        
        # Color based on ensemble prediction accuracy
        error = abs(ensemble - gt)
        if error < 2:
            color = 'green'
        elif error < 5:
            color = 'orange'
        else:
            color = 'red'
        
        ax.set_title(title, fontsize=9, color=color)
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('M7 Complete Test Results - ALL Models (Proper)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_path = output_dir / 'testset_M7_complete.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {viz_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()