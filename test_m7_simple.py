#!/usr/bin/env python3
"""Test M7 crab images with all models."""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 80)
    print("TESTING M7 CRAB IMAGES")
    print("=" * 80)
    
    # Add src to path
    sys.path.insert(0, '/Users/gen/green_crabs/src')
    from feature_extractor import GeneralCrustaceanFeatureExtractor
    
    # Load feature extractors
    print("\nLoading feature extractors...")
    cnn_extractor = GeneralCrustaceanFeatureExtractor('resnet50')
    vit_extractor = GeneralCrustaceanFeatureExtractor('vit_base')
    
    # Load all models
    print("Loading models...")
    models_dir = '/Users/gen/green_crabs/models'
    
    # Single-shot models
    cnn_rf = joblib.load(os.path.join(models_dir, 'cnn_rf_model.joblib'))
    cnn_gb = joblib.load(os.path.join(models_dir, 'cnn_gb_model.joblib'))
    cnn_ensemble = joblib.load(os.path.join(models_dir, 'cnn_ensemble_model.joblib'))
    cnn_scaler = joblib.load(os.path.join(models_dir, 'cnn_features_scaler.joblib'))
    
    vit_rf = joblib.load(os.path.join(models_dir, 'vit_rf_model.joblib'))
    vit_gb = joblib.load(os.path.join(models_dir, 'vit_gb_model.joblib'))
    vit_ensemble = joblib.load(os.path.join(models_dir, 'vit_ensemble_model.joblib'))
    vit_scaler = joblib.load(os.path.join(models_dir, 'vit_features_scaler.joblib'))
    
    # Temporal models
    temporal_rf = joblib.load(os.path.join(models_dir, 'temporal_vit_rf_model.joblib'))
    temporal_gb = joblib.load(os.path.join(models_dir, 'temporal_vit_gb_model.joblib'))
    temporal_ensemble = joblib.load(os.path.join(models_dir, 'temporal_vit_ensemble_model.joblib'))
    temporal_scaler = joblib.load(os.path.join(models_dir, 'temporal_vit_features_scaler.joblib'))
    
    # Process M7
    base_dir = '/Users/gen/green_crabs/NH Green Crab Project 2016/Crabs Aug 26 - Oct 4'
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
                    
                    # Parse ground truth from folder name if available
                    gt_days = None
                    if date_folder == '9:21':  # 2 days before 9/23
                        gt_days = 2.0
                    elif date_folder == '9:20':  # 3 days before 9/23
                        gt_days = 3.0
                    elif date_folder == '9:19':  # 4 days before 9/23
                        gt_days = 4.0
                    elif date_folder == '9:14':  # 9 days before 9/23
                        gt_days = 9.0
                    elif date_folder == '9:9':  # 14 days before 9/23
                        gt_days = 14.0
                    elif date_folder == '9:8':  # 15 days before 9/23
                        gt_days = 15.0
                    elif date_folder == '9:1':  # 22 days before 9/23
                        gt_days = 22.0
                    elif date_folder == '9:23':  # molt day
                        gt_days = 0.0
                    elif date_folder == '10:4':  # post-molt
                        gt_days = -11.0
                    
                    all_images.append({
                        'crab_id': 'M7',
                        'date': date_folder,
                        'image_path': img_path,
                        'image_name': img_file,
                        'ground_truth': gt_days
                    })
    
    print(f"Found {len(all_images)} images for M7")
    
    if not all_images:
        print("No images found for M7!")
        return
    
    # Extract features
    print("\nExtracting features...")
    cnn_features_list = []
    vit_features_list = []
    
    for i, img_info in enumerate(all_images):
        if i % 5 == 0:
            print(f"  Processing image {i+1}/{len(all_images)}")
        
        img = Image.open(img_info['image_path'])
        img_array = np.array(img)
        
        # Extract CNN features
        cnn_features = cnn_extractor.extract_features(img_array)
        cnn_features_list.append(cnn_features.flatten())
        
        # Extract VIT features
        vit_features = vit_extractor.extract_features(img_array)
        vit_features_list.append(vit_features.flatten())
    
    cnn_features_array = np.array(cnn_features_list)
    vit_features_array = np.array(vit_features_list)
    
    # Scale features
    print("\nMaking predictions...")
    cnn_features_scaled = cnn_scaler.transform(cnn_features_array)
    vit_features_scaled = vit_scaler.transform(vit_features_array)
    temporal_features_scaled = temporal_scaler.transform(vit_features_array)
    
    # Make predictions with all models
    cnn_rf_preds = cnn_rf.predict(cnn_features_scaled)
    cnn_gb_preds = cnn_gb.predict(cnn_features_scaled)
    cnn_ensemble_preds = cnn_ensemble.predict(cnn_features_scaled)
    
    vit_rf_preds = vit_rf.predict(vit_features_scaled)
    vit_gb_preds = vit_gb.predict(vit_features_scaled)
    vit_ensemble_preds = vit_ensemble.predict(vit_features_scaled)
    
    temporal_rf_preds = temporal_rf.predict(temporal_features_scaled)
    temporal_gb_preds = temporal_gb.predict(temporal_features_scaled)
    temporal_ensemble_preds = temporal_ensemble.predict(temporal_features_scaled)
    
    # Save results
    results_df = pd.DataFrame(all_images)
    results_df['cnn_rf_pred'] = cnn_rf_preds
    results_df['cnn_gb_pred'] = cnn_gb_preds
    results_df['cnn_ensemble_pred'] = cnn_ensemble_preds
    results_df['vit_rf_pred'] = vit_rf_preds
    results_df['vit_gb_pred'] = vit_gb_preds
    results_df['vit_ensemble_pred'] = vit_ensemble_preds
    results_df['temporal_rf_pred'] = temporal_rf_preds
    results_df['temporal_gb_pred'] = temporal_gb_preds
    results_df['temporal_ensemble_pred'] = temporal_ensemble_preds
    
    # Save CSV
    output_dir = '/Users/gen/green_crabs/testset_complete_results'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'M7_complete_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to: {csv_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    n_images = len(all_images)
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx in range(n_images):
        if idx < len(axes):
            ax = axes[idx]
            img_info = all_images[idx]
            
            # Load and display image
            img = Image.open(img_info['image_path'])
            img_resized = img.resize((150, 150))
            ax.imshow(img_resized)
            ax.axis('off')
            
            # Determine color based on predictions
            avg_pred = (temporal_rf_preds[idx] + temporal_gb_preds[idx] + temporal_ensemble_preds[idx]) / 3
            if avg_pred <= 3:
                color = 'red'  # Peeler
            elif avg_pred <= 14:
                color = 'orange'  # Pre-molt
            else:
                color = 'green'  # Inter-molt
            
            # Add predictions as title
            title = f"M7 - {img_info['date']}"
            if img_info['ground_truth'] is not None:
                title += f"\nGT: {img_info['ground_truth']:.0f}d"
            title += f"\nCNN: {cnn_ensemble_preds[idx]:.1f}d"
            title += f"\nVIT: {vit_ensemble_preds[idx]:.1f}d"  
            title += f"\nTemporal: {temporal_ensemble_preds[idx]:.1f}d"
            
            ax.set_title(title, fontsize=8, color=color, fontweight='bold' if avg_pred <= 3 else 'normal')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('M7 Complete Test Results - ALL Models', fontsize=14, fontweight='bold')
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
    
    # Calculate errors for images with ground truth
    has_gt = [i for i, img in enumerate(all_images) if img['ground_truth'] is not None]
    if has_gt:
        gt_values = np.array([all_images[i]['ground_truth'] for i in has_gt])
        
        print(f"\nImages with ground truth: {len(has_gt)}")
        
        # CNN ensemble errors
        cnn_errors = np.abs(cnn_ensemble_preds[has_gt] - gt_values)
        print(f"\nCNN Ensemble Model:")
        print(f"  MAE: {np.mean(cnn_errors):.2f} days")
        print(f"  Max Error: {np.max(cnn_errors):.2f} days")
        print(f"  <2 day accuracy: {100 * np.sum(cnn_errors < 2) / len(cnn_errors):.1f}%")
        
        # VIT ensemble errors
        vit_errors = np.abs(vit_ensemble_preds[has_gt] - gt_values)
        print(f"\nVIT Ensemble Model:")
        print(f"  MAE: {np.mean(vit_errors):.2f} days")
        print(f"  Max Error: {np.max(vit_errors):.2f} days")
        print(f"  <2 day accuracy: {100 * np.sum(vit_errors < 2) / len(vit_errors):.1f}%")
        
        # Temporal ensemble errors
        temporal_errors = np.abs(temporal_ensemble_preds[has_gt] - gt_values)
        print(f"\nTemporal Ensemble Model:")
        print(f"  MAE: {np.mean(temporal_errors):.2f} days")
        print(f"  Max Error: {np.max(temporal_errors):.2f} days")
        print(f"  <2 day accuracy: {100 * np.sum(temporal_errors < 2) / len(temporal_errors):.1f}%")
    
    print("\n" + "=" * 80)
    print("Phase Distribution (Temporal Ensemble):")
    peeler = np.sum(temporal_ensemble_preds <= 3)
    premolt = np.sum((temporal_ensemble_preds > 3) & (temporal_ensemble_preds <= 14))
    intermolt = np.sum(temporal_ensemble_preds > 14)
    postmolt = np.sum(temporal_ensemble_preds < 0)
    
    print(f"  PEELER (0-3 days): {peeler} images")
    print(f"  Pre-molt (4-14 days): {premolt} images")
    print(f"  Inter-molt (>14 days): {intermolt} images")
    print(f"  Post-molt (<0 days): {postmolt} images")
    
    print("\nDone!")

if __name__ == "__main__":
    main()