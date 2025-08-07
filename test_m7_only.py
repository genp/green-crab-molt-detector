#!/usr/bin/env python3
"""Test M7 crab images with temporal models."""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import feature extractor (assumes src is in PYTHONPATH)
from feature_extractor import GeneralCrustaceanFeatureExtractor
import joblib

def main():
    print("=" * 80)
    print("TESTING M7 CRAB IMAGES")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    feature_extractor = GeneralCrustaceanFeatureExtractor('vit_base')
    
    # Load temporal models
    models_dir = 'models'
    temporal_rf = joblib.load(os.path.join(models_dir, 'temporal_vit_rf_model.joblib'))
    temporal_gb = joblib.load(os.path.join(models_dir, 'temporal_vit_gb_model.joblib'))
    temporal_ensemble_model = joblib.load(os.path.join(models_dir, 'temporal_vit_ensemble_model.joblib'))
    temporal_scaler = joblib.load(os.path.join(models_dir, 'temporal_vit_features_scaler.joblib'))
    
    # Process M7
    base_dir = 'NH Green Crab Project 2016/Crabs Aug 26 - Oct 4'
    crab_folder = os.path.join(base_dir, 'M7')
    
    print(f"\nProcessing M7 from: {crab_folder}")
    
    # Collect all images
    all_images = []
    for date_folder in sorted(os.listdir(crab_folder)):
        date_path = os.path.join(crab_folder, date_folder)
        if os.path.isdir(date_path):
            for img_file in os.listdir(date_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(date_path, img_file)
                    all_images.append({
                        'crab_id': 'M7',
                        'date': date_folder,
                        'image_path': img_path,
                        'image_name': img_file
                    })
    
    print(f"Found {len(all_images)} images for M7")
    
    if not all_images:
        print("No images found for M7!")
        return
    
    # Extract features
    print("\nExtracting features...")
    features_list = []
    for i, img_info in enumerate(all_images):
        if i % 10 == 0:
            print(f"  Processing image {i+1}/{len(all_images)}")
        
        img = Image.open(img_info['image_path'])
        img_array = np.array(img)
        features = feature_extractor.extract_features(img_array)
        features_list.append(features.flatten())
    
    features_array = np.array(features_list)
    
    # Scale features
    print("\nMaking predictions...")
    features_scaled = temporal_scaler.transform(features_array)
    
    # Make predictions
    rf_preds = temporal_rf.predict(features_scaled)
    gb_preds = temporal_gb.predict(features_scaled)
    ensemble_preds = temporal_ensemble_model.predict(features_scaled)
    
    # Save results
    results_df = pd.DataFrame(all_images)
    results_df['temporal_rf_pred'] = rf_preds
    results_df['temporal_gb_pred'] = gb_preds
    results_df['temporal_ensemble_pred'] = ensemble_preds
    
    # Save CSV
    output_dir = 'testset_complete_results'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'M7_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to: {csv_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    n_images = min(len(all_images), 20)  # Show first 20 images
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx in range(n_images):
        ax = axes[idx]
        img_info = all_images[idx]
        
        img = Image.open(img_info['image_path'])
        ax.imshow(img)
        ax.axis('off')
        
        # Add predictions
        title = f"M7 - {img_info['date']}\n"
        title += f"RF: {rf_preds[idx]:.1f}d, GB: {gb_preds[idx]:.1f}d\n"
        title += f"Ensemble: {ensemble_preds[idx]:.1f} days"
        
        ax.set_title(title, fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('M7 Complete Test Results - TEMPORAL Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, 'testset_M7_complete.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {viz_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("M7 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal images processed: {len(all_images)}")
    print(f"\nTemporal RF predictions:")
    print(f"  Mean: {np.mean(rf_preds):.2f} days")
    print(f"  Std: {np.std(rf_preds):.2f} days")
    print(f"  Min: {np.min(rf_preds):.2f} days")
    print(f"  Max: {np.max(rf_preds):.2f} days")
    
    print(f"\nTemporal GB predictions:")
    print(f"  Mean: {np.mean(gb_preds):.2f} days")
    print(f"  Std: {np.std(gb_preds):.2f} days")
    print(f"  Min: {np.min(gb_preds):.2f} days")
    print(f"  Max: {np.max(gb_preds):.2f} days")
    
    print(f"\nTemporal Ensemble predictions:")
    print(f"  Mean: {np.mean(ensemble_preds):.2f} days")
    print(f"  Std: {np.std(ensemble_preds):.2f} days")
    print(f"  Min: {np.min(ensemble_preds):.2f} days")
    print(f"  Max: {np.max(ensemble_preds):.2f} days")
    
    print("\nDone!")

if __name__ == "__main__":
    main()