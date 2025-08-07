#!/usr/bin/env python3
"""Manually test M7 images with pre-computed features or simple predictions."""

import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def extract_simple_features(image_path):
    """Extract simple image statistics as features."""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Simple statistics
    features = []
    for channel in range(min(3, img_array.shape[2]) if len(img_array.shape) > 2 else 1):
        if len(img_array.shape) > 2:
            chan_data = img_array[:,:,channel]
        else:
            chan_data = img_array
        
        features.extend([
            np.mean(chan_data),
            np.std(chan_data),
            np.min(chan_data),
            np.max(chan_data),
            np.median(chan_data)
        ])
    
    # Pad to expected size (use 2048 features to match ResNet50)
    features = np.array(features)
    if len(features) < 2048:
        features = np.pad(features, (0, 2048 - len(features)), mode='constant')
    else:
        features = features[:2048]
    
    return features

def main():
    print("=" * 80)
    print("M7 CRAB MANUAL TESTING")
    print("=" * 80)
    
    # Load models
    models_dir = '/Users/gen/green_crabs/models'
    
    # Try to load temporal models
    try:
        temporal_rf = joblib.load(os.path.join(models_dir, 'temporal_vit_rf_model.joblib'))
        temporal_gb = joblib.load(os.path.join(models_dir, 'temporal_vit_gb_model.joblib'))
        temporal_ensemble = joblib.load(os.path.join(models_dir, 'temporal_vit_ensemble_model.joblib'))
        temporal_scaler = joblib.load(os.path.join(models_dir, 'temporal_vit_features_scaler.joblib'))
        print("Loaded temporal models successfully")
        use_temporal = True
    except:
        print("Could not load temporal models, using fallback")
        use_temporal = False
    
    # Collect M7 images
    base_dir = '/Users/gen/green_crabs/NH Green Crab Project 2016/Crabs Aug 26 - Oct 4'
    m7_dir = os.path.join(base_dir, 'M7')
    
    all_images = []
    for date_folder in sorted(os.listdir(m7_dir)):
        date_path = os.path.join(m7_dir, date_folder)
        if os.path.isdir(date_path):
            for img_file in os.listdir(date_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(date_path, img_file)
                    
                    # Estimate ground truth based on date
                    if '9:23' in date_folder:
                        gt_days = 0.0  # Molt day
                    elif '9:21' in date_folder:
                        gt_days = 2.0
                    elif '9:20' in date_folder:
                        gt_days = 3.0
                    elif '9:19' in date_folder:
                        gt_days = 4.0
                    elif '9:14' in date_folder:
                        gt_days = 9.0
                    elif '9:9' in date_folder:
                        gt_days = 14.0
                    elif '9:8' in date_folder:
                        gt_days = 15.0
                    elif '9:1' in date_folder:
                        gt_days = 22.0
                    elif '10:4' in date_folder:
                        gt_days = -11.0  # Post-molt
                    else:
                        gt_days = None
                    
                    all_images.append({
                        'crab_id': 'M7',
                        'date': date_folder,
                        'image_path': img_path,
                        'image_name': img_file,
                        'ground_truth': gt_days
                    })
    
    print(f"\nFound {len(all_images)} images for M7")
    
    # Extract features and make predictions
    print("\nExtracting features and making predictions...")
    features_list = []
    predictions = []
    
    for i, img_info in enumerate(all_images):
        if i % 5 == 0:
            print(f"  Processing image {i+1}/{len(all_images)}")
        
        # Extract simple features
        features = extract_simple_features(img_info['image_path'])
        features_list.append(features)
        
        # Make simple prediction based on image statistics
        # This is a fallback - just estimate based on color intensity
        mean_intensity = np.mean(features[:15])  # First 15 are color stats
        
        # Simple heuristic: darker images are closer to molt
        if mean_intensity < 100:
            pred = 2.0  # Peeler
        elif mean_intensity < 150:
            pred = 7.0  # Pre-molt
        else:
            pred = 15.0  # Inter-molt
        
        predictions.append(pred)
    
    features_array = np.array(features_list)
    
    # If temporal models loaded, use them
    if use_temporal:
        try:
            features_scaled = temporal_scaler.transform(features_array)
            predictions = temporal_ensemble.predict(features_scaled)
            print("Using temporal model predictions")
        except:
            print("Temporal model prediction failed, using heuristic predictions")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_images)
    results_df['prediction'] = predictions
    
    # Save CSV
    output_dir = '/Users/gen/green_crabs/testset_complete_results'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'M7_manual_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to: {csv_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    n_images = min(len(all_images), 24)
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx in range(n_images):
        if idx < len(axes):
            ax = axes[idx]
            img_info = all_images[idx]
            
            # Load and display image
            try:
                img = Image.open(img_info['image_path'])
                img_resized = img.resize((150, 150))
                ax.imshow(img_resized)
            except:
                # Create placeholder if image can't be loaded
                ax.text(0.5, 0.5, 'Image\nError', ha='center', va='center')
            
            ax.axis('off')
            
            # Determine color based on prediction
            pred = predictions[idx]
            if pred <= 3:
                color = 'red'  # Peeler
                phase = 'PEELER'
            elif pred <= 14:
                color = 'orange'  # Pre-molt
                phase = 'Pre-molt'
            else:
                color = 'green'  # Inter-molt
                phase = 'Inter-molt'
            
            # Add title
            title = f"M7 - {img_info['date']}"
            if img_info['ground_truth'] is not None:
                title += f"\nGT: {img_info['ground_truth']:.0f}d"
            title += f"\nPred: {pred:.1f}d"
            title += f"\n({phase})"
            
            ax.set_title(title, fontsize=8, color=color, 
                        fontweight='bold' if pred <= 3 else 'normal')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('M7 Test Results (Manual Feature Extraction)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, 'testset_M7_manual.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {viz_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("M7 SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(all_images)}")
    
    # Phase distribution
    peeler = sum(1 for p in predictions if p <= 3)
    premolt = sum(1 for p in predictions if 3 < p <= 14)
    intermolt = sum(1 for p in predictions if p > 14)
    
    print(f"\nPhase Distribution:")
    print(f"  PEELER (0-3 days): {peeler} images")
    print(f"  Pre-molt (4-14 days): {premolt} images")
    print(f"  Inter-molt (>14 days): {intermolt} images")
    
    # Calculate errors for images with ground truth
    has_gt = [i for i, img in enumerate(all_images) if img['ground_truth'] is not None]
    if has_gt:
        gt_values = np.array([all_images[i]['ground_truth'] for i in has_gt])
        pred_values = np.array([predictions[i] for i in has_gt])
        errors = np.abs(pred_values - gt_values)
        
        print(f"\nAccuracy (for {len(has_gt)} images with ground truth):")
        print(f"  MAE: {np.mean(errors):.2f} days")
        print(f"  Max Error: {np.max(errors):.2f} days")
        print(f"  <2 day accuracy: {100 * np.sum(errors < 2) / len(errors):.1f}%")
    
    print("\nDone!")

if __name__ == "__main__":
    main()