#!/usr/bin/env python3
"""
Get list of crab folders that would be in the test set for local testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv("data/processed/crab_dataset.csv")

# Get unique crab IDs
crab_ids = df['crab_id'].unique()
print(f"Total crabs: {len(crab_ids)}")

# Split using same seed as training
train_crabs, test_crabs = train_test_split(crab_ids, test_size=0.2, random_state=42)

print(f"\nTest set crabs ({len(test_crabs)} crabs):")
print("=" * 50)

# Find actual folders for test crabs
base_path = Path("NH Green Crab Project 2016")

test_folders = []
for period_folder in base_path.iterdir():
    if not period_folder.is_dir() or "Crabs" not in period_folder.name:
        continue
    
    for crab_folder in period_folder.iterdir():
        if not crab_folder.is_dir():
            continue
        
        # Extract crab ID from folder name
        crab_id = crab_folder.name.split()[0]
        
        if crab_id in test_crabs:
            test_folders.append(crab_folder)
            
            # Find sample images
            sample_images = []
            for date_folder in crab_folder.iterdir():
                if date_folder.is_dir():
                    images = list(date_folder.glob("*.jpg")) + list(date_folder.glob("*.JPG"))
                    if images and len(sample_images) < 2:
                        sample_images.append(str(images[0]))
            
            print(f"\n📁 {crab_folder.relative_to(base_path)}")
            print(f"   Crab ID: {crab_id}")
            
            # Get molt info from folder name
            if "molted" in crab_folder.name:
                import re
                molt_match = re.search(r'molted (\d+:\d+)', crab_folder.name)
                if molt_match:
                    print(f"   Molt date: {molt_match.group(1)}")
            
            if sample_images:
                print(f"   Sample images to test:")
                for img in sample_images[:2]:
                    print(f"     - {img}")

print("\n" + "=" * 50)
print("You can use these images to test the app locally:")
print("1. Run: python app.py")
print("2. Open http://localhost:5000")
print("3. Upload one of the sample images above")
print("4. The app will predict days until molt")