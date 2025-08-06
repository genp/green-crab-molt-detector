"""
Analyze the molt bin population breakdown in the training dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_molt_phase_bins(days_until_molt):
    """Convert continuous days_until_molt to categorical bins."""
    bins = [-0.1, 5, 10, 15, 20, 30]  # 0-5, 6-10, 11-15, 16-20, 21+ days
    labels = ['0-5_days', '6-10_days', '11-15_days', '16-20_days', '21+_days']
    return pd.cut(days_until_molt, bins=bins, labels=labels, include_lowest=True)

def main():
    # Load dataset
    data_path = Path("data/processed/crab_dataset.csv")
    if not data_path.exists():
        print("Dataset not found. Please run feature extraction first.")
        return
    
    df = pd.read_csv(data_path)
    
    # Filter for samples with valid molt dates (training set)
    valid_mask = df['days_until_molt'].notna()
    training_df = df[valid_mask].copy()
    
    print("="*60)
    print("MOLT BIN POPULATION BREAKDOWN - TRAINING SET")
    print("="*60)
    
    print(f"\nTotal training samples: {len(training_df)}")
    print(f"Total crabs in training: {training_df['crab_id'].nunique()}")
    print(f"Sex distribution: {training_df['sex'].value_counts().to_dict()}")
    
    # Create molt phase bins
    training_df['molt_bins'] = create_molt_phase_bins(training_df['days_until_molt'])
    
    print(f"\n📊 MOLT BIN BREAKDOWN:")
    print("-" * 40)
    
    bin_counts = training_df['molt_bins'].value_counts().sort_index()
    total = len(training_df)
    
    for bin_name, count in bin_counts.items():
        percentage = (count / total) * 100
        print(f"{bin_name:12s}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Show distribution by sex
    print(f"\n👫 BY SEX:")
    print("-" * 40)
    
    for sex in ['F', 'M']:
        sex_df = training_df[training_df['sex'] == sex]
        if len(sex_df) == 0:
            continue
            
        print(f"\n{sex} (Female)" if sex == 'F' else f"\n{sex} (Male):")
        sex_bin_counts = sex_df['molt_bins'].value_counts().sort_index()
        sex_total = len(sex_df)
        
        for bin_name, count in sex_bin_counts.items():
            percentage = (count / sex_total) * 100
            print(f"  {bin_name:12s}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Show statistics
    print(f"\n📈 DAYS UNTIL MOLT STATISTICS:")
    print("-" * 40)
    print(f"Mean: {training_df['days_until_molt'].mean():.1f} days")
    print(f"Std:  {training_df['days_until_molt'].std():.1f} days")
    print(f"Min:  {training_df['days_until_molt'].min():.1f} days")
    print(f"Max:  {training_df['days_until_molt'].max():.1f} days")
    
    # Show quartiles
    quartiles = training_df['days_until_molt'].quantile([0.25, 0.5, 0.75])
    print(f"25%:  {quartiles[0.25]:.1f} days")
    print(f"50%:  {quartiles[0.5]:.1f} days (median)")
    print(f"75%:  {quartiles[0.75]:.1f} days")
    
    # Class imbalance analysis
    print(f"\n⚠️  CLASS IMBALANCE ANALYSIS:")
    print("-" * 40)
    
    min_count = bin_counts.min()
    max_count = bin_counts.max()
    imbalance_ratio = max_count / min_count
    
    print(f"Most populated bin: {bin_counts.idxmax()} ({max_count} samples)")
    print(f"Least populated bin: {bin_counts.idxmin()} ({min_count} samples)")
    print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 3:
        print("⚠️  SEVERE CLASS IMBALANCE detected!")
        print("   Consider using stratified sampling or class weights")
    elif imbalance_ratio > 2:
        print("⚠️  MODERATE CLASS IMBALANCE detected")
        print("   May benefit from balanced sampling techniques")
    else:
        print("✅ Relatively balanced dataset")

if __name__ == "__main__":
    main()