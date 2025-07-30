"""
Complete pipeline runner for green crab molt detection system.

This script runs the entire pipeline:
1. Data loading and preprocessing
2. Feature extraction
3. t-SNE visualization
4. Model training
5. Web app preparation
"""

import subprocess
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    logger.info('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error code {e.returncode}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False


def check_requirements():
    """Check if required files and directories exist."""
    logger.info("Checking requirements...")
    
    issues = []
    
    # Check for YOLO model
    yolo_path = Path("/Users/genp/BarderryAppliedResearch/FathomNet/qscp/jupyter_notebooks/fathomverse_detector/fathomverse-only-imgs_update_to_FathomNet-NoGameLabels-2024-09-28-model_yolo8_epochs_10_2024-10-22.pt")
    if not yolo_path.exists():
        issues.append(f"YOLO model not found at: {yolo_path}")
        logger.warning("YOLO model not found - will use CNN features instead")
    
    # Check for image data
    data_path = Path("NH Green Crab Project 2016")
    if not data_path.exists():
        issues.append(f"Image data directory not found: {data_path}")
    
    if issues:
        logger.warning("Some requirements are missing:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
            
    return True


def main():
    """Run the complete pipeline."""
    logger.info("Green Crab Molt Detection Pipeline")
    logger.info("==================================")
    
    # Check requirements
    if not check_requirements():
        logger.error("Exiting due to missing requirements")
        return 1
    
    # Create necessary directories
    for dir_name in ['data/processed', 'models', 'plots', 'temp_uploads']:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Feature extraction and visualization
    if not run_command(
        "python run_feature_analysis.py",
        "Feature extraction and t-SNE visualization"
    ):
        logger.error("Feature extraction failed")
        return 1
    
    # Step 2: Model training
    if not run_command(
        "python train_model.py",
        "Training molt phase regression models"
    ):
        logger.error("Model training failed")
        return 1
    
    # Step 3: Prepare deployment
    if not run_command(
        "python deploy.py",
        "Creating deployment files"
    ):
        logger.error("Deployment preparation failed")
        return 1
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)
    
    logger.info("\nGenerated files:")
    logger.info("- data/processed/: Dataset and extracted features")
    logger.info("- plots/: t-SNE visualizations and model comparisons")
    logger.info("- models/: Trained regression models")
    logger.info("- Deployment files: Dockerfile, DEPLOYMENT.md, etc.")
    
    logger.info("\nNext steps:")
    logger.info("1. Review the plots in the plots/ directory")
    logger.info("2. Test the web app locally: python app.py")
    logger.info("3. Deploy using instructions in DEPLOYMENT.md")
    
    # Check if models were successfully trained
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.joblib"))
    if model_files:
        logger.info(f"\nTrained models found: {len(model_files)}")
        for model_file in model_files:
            logger.info(f"  - {model_file.name}")
    else:
        logger.warning("\nNo trained models found - check training logs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())