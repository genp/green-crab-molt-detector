"""
Main script to run feature extraction and t-SNE visualization for green crab molt phase analysis.

This script:
1. Loads the crab image dataset
2. Extracts features using YOLO and/or CNN models
3. Creates t-SNE visualizations colored by molt status
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import GreenCrabDataLoader
from feature_extractor import YOLOFeatureExtractor, GeneralCrustaceanFeatureExtractor
from visualization import MoltPhaseVisualizer, create_feature_comparison_plot

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run the complete feature analysis pipeline."""
    
    # Paths
    base_path = Path("/Users/gen/green_crabs")
    yolo_model_path = Path("/Users/genp/BarderryAppliedResearch/FathomNet/qscp/jupyter_notebooks/fathomverse_detector/fathomverse-only-imgs_update_to_FathomNet-NoGameLabels-2024-09-28-model_yolo8_epochs_10_2024-10-22.pt")
    output_dir = base_path / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset
    logger.info("Loading dataset...")
    loader = GreenCrabDataLoader(base_path)
    df = loader.load_dataset()
    
    # Save dataset metadata
    dataset_path = output_dir / "crab_dataset.csv"
    df.to_csv(dataset_path, index=False)
    logger.info(f"Saved dataset metadata to {dataset_path}")
    
    # Step 2: Extract features
    image_paths = df['image_path'].tolist()
    
    # Try YOLO features
    yolo_features = None
    if yolo_model_path.exists():
        logger.info("Extracting YOLO features...")
        try:
            yolo_extractor = YOLOFeatureExtractor(yolo_model_path)
            yolo_features = yolo_extractor.extract_features_batch(image_paths, batch_size=8)
            
            # Save YOLO features
            yolo_features_path = output_dir / "yolo_features.npy"
            np.save(yolo_features_path, yolo_features)
            logger.info(f"Saved YOLO features to {yolo_features_path}")
            logger.info(f"YOLO features shape: {yolo_features.shape}")
        except Exception as e:
            logger.error(f"Failed to extract YOLO features: {e}")
    else:
        logger.warning(f"YOLO model not found at {yolo_model_path}")
        
    # Extract CNN features as backup/comparison
    logger.info("Extracting CNN features...")
    try:
        cnn_extractor = GeneralCrustaceanFeatureExtractor('resnet50')
        cnn_features = cnn_extractor.extract_features_batch(image_paths, batch_size=16)
        
        # Save CNN features
        cnn_features_path = output_dir / "cnn_features.npy"
        np.save(cnn_features_path, cnn_features)
        logger.info(f"Saved CNN features to {cnn_features_path}")
        logger.info(f"CNN features shape: {cnn_features.shape}")
    except Exception as e:
        logger.error(f"Failed to extract CNN features: {e}")
        cnn_features = None
        
    # Step 3: Create visualizations
    logger.info("Creating visualizations...")
    
    # Use whichever features are available
    if yolo_features is not None:
        features_to_use = yolo_features
        feature_type = "YOLO"
    elif cnn_features is not None:
        features_to_use = cnn_features
        feature_type = "CNN"
    else:
        logger.error("No features available for visualization")
        return
        
    # Create visualizer
    visualizer = MoltPhaseVisualizer(features_to_use, df)
    
    # Generate t-SNE embedding
    logger.info(f"Creating t-SNE embedding from {feature_type} features...")
    embedding = visualizer.create_tsne_embedding(perplexity=30)
    
    # Save embedding
    embedding_path = output_dir / f"{feature_type.lower()}_tsne_embedding.npy"
    np.save(embedding_path, embedding)
    logger.info(f"Saved t-SNE embedding to {embedding_path}")
    
    # Create visualizations
    plots_dir = base_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Main t-SNE plot with multiple views
    logger.info("Creating main t-SNE visualization...")
    visualizer.plot_tsne_by_molt_status(
        embedding, 
        save_path=plots_dir / f"{feature_type.lower()}_tsne_molt_status.png"
    )
    
    # Molt phase categories plot
    logger.info("Creating molt phase categories visualization...")
    visualizer.plot_molt_phase_categories(
        embedding,
        save_path=plots_dir / f"{feature_type.lower()}_tsne_molt_phases.png"
    )
    
    # Temporal progression plot
    logger.info("Creating temporal progression visualization...")
    visualizer.plot_temporal_progression(
        save_path=plots_dir / "temporal_molt_progression.png"
    )
    
    # Feature comparison plot (if both features available)
    if yolo_features is not None and cnn_features is not None:
        logger.info("Creating feature comparison visualization...")
        create_feature_comparison_plot(
            yolo_features, cnn_features, df,
            save_path=plots_dir / "feature_comparison.png"
        )
        
    logger.info("Feature analysis complete!")
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total images: {len(df)}")
    print(f"Total crabs: {df['crab_id'].nunique()}")
    print(f"Sex distribution: {df['sex'].value_counts().to_dict()}")
    print(f"Images with molt dates: {df['molt_date'].notna().sum()}")
    print(f"Post-molt images: {df['is_molted'].sum()}")
    
    if df['days_until_molt'].notna().any():
        print(f"\nDays until molt statistics:")
        print(df['days_until_molt'].describe())


if __name__ == "__main__":
    main()