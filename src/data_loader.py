"""
Data loader for green crab images with molt phase labels.

This module handles:
- Parsing directory structure to extract molt dates
- Loading and preprocessing crab images
- Creating dataset with molt phase regression targets
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
from PIL import Image
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CrabImage:
    """Container for a single crab image with metadata."""
    
    crab_id: str
    image_path: Path
    capture_date: datetime
    molt_date: Optional[datetime]
    days_until_molt: Optional[float]
    is_molted: bool
    sex: str  # 'M' or 'F'
    
    
class GreenCrabDataLoader:
    """
    Load and preprocess green crab images with molt phase information.
    
    The data structure expected:
    - Folders named like "F1 (molted 9:23)" or "M1"
    - Subfolders with dates like "8:26", "9:1", etc.
    - Images within date folders
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            base_path: Path to the base directory containing crab data
        """
        self.base_path = Path(base_path)
        self.crab_data: List[CrabImage] = []
        self.molt_date_pattern = re.compile(r'([MF]\d+)\s*\(molted\s*(\d+):(\d+)\)')
        self.date_folder_pattern = re.compile(r'(\d+):(\d+)')
        
    def parse_molt_date(self, folder_name: str, year: int = 2016) -> Optional[Tuple[str, datetime]]:
        """
        Extract crab ID and molt date from folder name.
        
        Args:
            folder_name: Name of the folder (e.g., "F1 (molted 9:23)")
            year: Year of the study
            
        Returns:
            Tuple of (crab_id, molt_date) or None if not found
        """
        match = self.molt_date_pattern.match(folder_name)
        if match:
            crab_id = match.group(1)
            month = int(match.group(2))
            day = int(match.group(3))
            molt_date = datetime(year, month, day)
            return crab_id, molt_date
        
        # Check if it's just a crab ID without molt date
        if folder_name.startswith(('M', 'F')) and folder_name[1:].isdigit():
            return folder_name, None
            
        return None
        
    def parse_capture_date(self, date_folder: str, year: int = 2016) -> Optional[datetime]:
        """
        Parse capture date from folder name.
        
        Args:
            date_folder: Folder name like "8:26" or "9:1"
            year: Year of the study
            
        Returns:
            datetime object or None
        """
        # Check for special folders like "9:19 MOLTED"
        if 'MOLTED' in date_folder:
            date_part = date_folder.replace('MOLTED', '').strip()
        else:
            date_part = date_folder
            
        match = self.date_folder_pattern.match(date_part)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            return datetime(year, month, day)
        return None
        
    def calculate_days_until_molt(self, capture_date: datetime, molt_date: Optional[datetime]) -> Optional[float]:
        """
        Calculate days from capture to molt.
        
        Args:
            capture_date: Date image was captured
            molt_date: Date crab molted (if known)
            
        Returns:
            Days until molt (negative if after molt) or None
        """
        if molt_date is None:
            return None
        return (molt_date - capture_date).days
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load all crab images and create a dataset.
        
        Returns:
            DataFrame with columns: crab_id, image_path, capture_date, 
            molt_date, days_until_molt, is_molted, sex
        """
        print("Loading green crab dataset...")
        
        # Look for the main data directory
        data_dirs = [
            self.base_path / "NH Green Crab Project 2016" / "Crabs Aug 26 - Oct 4",
            self.base_path / "NH Green Crab Project 2016" / "Crabs July 22 - Aug 23",
            self.base_path / "NH Green Crab Project 2016" / "Crabs June 28- July 21"
        ]
        
        for data_dir in data_dirs:
            if not data_dir.exists():
                print(f"Warning: Directory not found: {data_dir}")
                continue
                
            print(f"Processing: {data_dir.name}")
            
            # Process each crab folder
            for crab_folder in data_dir.iterdir():
                if not crab_folder.is_dir():
                    continue
                    
                # Parse crab ID and molt date
                parsed = self.parse_molt_date(crab_folder.name)
                if not parsed:
                    continue
                    
                crab_id, molt_date = parsed
                sex = crab_id[0]  # 'M' or 'F'
                
                # Process date subfolders
                for date_folder in crab_folder.iterdir():
                    if not date_folder.is_dir():
                        continue
                        
                    capture_date = self.parse_capture_date(date_folder.name)
                    if not capture_date:
                        continue
                        
                    # Check if this is a post-molt folder
                    is_molted = 'MOLTED' in date_folder.name
                    
                    # Process images in the folder
                    for image_file in date_folder.iterdir():
                        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            days_until_molt = self.calculate_days_until_molt(capture_date, molt_date)
                            
                            crab_image = CrabImage(
                                crab_id=crab_id,
                                image_path=image_file,
                                capture_date=capture_date,
                                molt_date=molt_date,
                                days_until_molt=days_until_molt,
                                is_molted=is_molted,
                                sex=sex
                            )
                            self.crab_data.append(crab_image)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'crab_id': img.crab_id,
                'image_path': str(img.image_path),
                'capture_date': img.capture_date,
                'molt_date': img.molt_date,
                'days_until_molt': img.days_until_molt,
                'is_molted': img.is_molted,
                'sex': img.sex
            }
            for img in self.crab_data
        ])
        
        print(f"Loaded {len(df)} images from {df['crab_id'].nunique()} crabs")
        print(f"Sex distribution: {df['sex'].value_counts().to_dict()}")
        print(f"Molted images: {df['is_molted'].sum()}")
        
        return df
        
    def load_and_preprocess_image(self, image_path: Union[str, Path], target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image as numpy array
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)


def main():
    """Test the data loader."""
    loader = GreenCrabDataLoader("/Users/gen/green_crabs")
    df = loader.load_dataset()
    
    # Save dataset metadata
    output_path = Path("/Users/gen/green_crabs/data/processed/crab_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    
    # Display statistics
    print("\nDataset statistics:")
    print(f"Date range: {df['capture_date'].min()} to {df['capture_date'].max()}")
    
    if df['days_until_molt'].notna().any():
        print(f"Days until molt range: {df['days_until_molt'].min():.0f} to {df['days_until_molt'].max():.0f}")
    

if __name__ == "__main__":
    main()