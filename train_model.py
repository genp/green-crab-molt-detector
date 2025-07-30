"""
Train molt phase regression model for green crabs.

This script:
1. Loads extracted features
2. Trains multiple regression models
3. Evaluates and compares model performance
4. Saves the best model
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from model import MoltPhaseRegressor, plot_model_comparison, create_prediction_visualization

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Train and evaluate molt phase regression models."""
    
    # Paths
    base_path = Path("/Users/gen/green_crabs")
    data_dir = base_path / "data" / "processed"
    models_dir = base_path / "models"
    plots_dir = base_path / "plots"
    
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    # Load dataset metadata
    logger.info("Loading dataset metadata...")
    metadata_path = data_dir / "crab_dataset.csv"
    if not metadata_path.exists():
        logger.error(f"Dataset metadata not found at {metadata_path}")
        logger.error("Please run 'python run_feature_analysis.py' first")
        return
        
    metadata = pd.read_csv(metadata_path)
    metadata['capture_date'] = pd.to_datetime(metadata['capture_date'])
    metadata['molt_date'] = pd.to_datetime(metadata['molt_date'])
    
    # Load features - try YOLO first, then CNN
    features = None
    feature_type = None
    
    yolo_features_path = data_dir / "yolo_features.npy"
    cnn_features_path = data_dir / "cnn_features.npy"
    
    if yolo_features_path.exists():
        logger.info("Loading YOLO features...")
        features = np.load(yolo_features_path)
        feature_type = "YOLO"
    elif cnn_features_path.exists():
        logger.info("Loading CNN features...")
        features = np.load(cnn_features_path)
        feature_type = "CNN"
    else:
        logger.error("No features found. Please run 'python run_feature_analysis.py' first")
        return
        
    logger.info(f"Loaded {feature_type} features with shape: {features.shape}")
    
    # Filter for samples with molt dates
    has_molt_date = metadata['days_until_molt'].notna()
    n_samples_with_molt = has_molt_date.sum()
    logger.info(f"Found {n_samples_with_molt} samples with known molt dates for training")
    
    if n_samples_with_molt < 50:
        logger.warning(f"Only {n_samples_with_molt} samples available for training. "
                      "Results may be unreliable.")
    
    # Compare different algorithms
    logger.info("\n=== Comparing different regression algorithms ===")
    comparison_df = plot_model_comparison(
        features, metadata,
        save_path=plots_dir / f"{feature_type.lower()}_model_comparison.png"
    )
    
    print("\nModel Comparison Results:")
    print(comparison_df.to_string(index=False))
    
    # Train best model (based on MAE)
    best_algorithm = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Algorithm']
    logger.info(f"\n=== Training best model: {best_algorithm} ===")
    
    best_regressor = MoltPhaseRegressor(best_algorithm)
    metrics = best_regressor.fit(features, metadata)
    
    print(f"\nBest Model ({best_algorithm}) Performance:")
    print(f"Mean Absolute Error: {metrics['mae']:.2f} days")
    print(f"Root Mean Squared Error: {metrics['rmse']:.2f} days")
    print(f"R² Score: {metrics['r2']:.3f}")
    print(f"Median Absolute Error: {metrics['median_ae']:.2f} days")
    
    # Create prediction visualization
    logger.info("Creating prediction visualizations...")
    create_prediction_visualization(
        best_regressor, features, metadata,
        save_path=plots_dir / f"{feature_type.lower()}_{best_algorithm}_predictions.png"
    )
    
    # Save the best model
    model_path = models_dir / f"molt_regressor_{feature_type.lower()}_{best_algorithm}.joblib"
    best_regressor.save_model(model_path)
    logger.info(f"Saved best model to {model_path}")
    
    # Create a summary report
    report_path = models_dir / f"training_report_{feature_type.lower()}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Green Crab Molt Phase Regression Model Training Report\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Feature Type: {feature_type}\n")
        f.write(f"Feature Dimensions: {features.shape}\n")
        f.write(f"Training Samples: {n_samples_with_molt}\n\n")
        
        f.write("Model Comparison Results:\n")
        f.write(comparison_df.to_string(index=False))
        f.write(f"\n\nBest Model: {best_algorithm}\n")
        f.write(f"MAE: {metrics['mae']:.2f} days\n")
        f.write(f"RMSE: {metrics['rmse']:.2f} days\n")
        f.write(f"R²: {metrics['r2']:.3f}\n")
        f.write(f"Median AE: {metrics['median_ae']:.2f} days\n")
        
    logger.info(f"Training report saved to {report_path}")
    
    # Additional analysis: Feature importance (if using tree-based model)
    if best_algorithm in ['random_forest', 'gradient_boost']:
        logger.info("Analyzing feature importance...")
        
        if hasattr(best_regressor.model, 'feature_importances_'):
            importances = best_regressor.model.feature_importances_
            n_top_features = min(20, len(importances))
            top_indices = np.argsort(importances)[-n_top_features:][::-1]
            
            print(f"\nTop {n_top_features} Most Important Features:")
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
                
    logger.info("\nModel training complete!")


if __name__ == "__main__":
    main()