"""
Flask web application for green crab molt phase detection.

This application allows users to:
- Upload crab images (top and/or underside views)
- Get molt phase predictions
- View confidence scores and recommendations
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from typing import Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extractor import YOLOFeatureExtractor, GeneralCrustaceanFeatureExtractor
from model import MoltPhaseRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
feature_extractor = None
regressor = None
feature_type = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load the feature extractor and regression model."""
    global feature_extractor, regressor, feature_type
    
    base_path = Path("/Users/gen/green_crabs")
    models_dir = base_path / "models"
    
    # Try to load YOLO feature extractor first
    yolo_model_path = Path("/Users/genp/BarderryAppliedResearch/FathomNet/qscp/jupyter_notebooks/fathomverse_detector/fathomverse-only-imgs_update_to_FathomNet-NoGameLabels-2024-09-28-model_yolo8_epochs_10_2024-10-22.pt")
    
    if yolo_model_path.exists():
        try:
            logger.info("Loading YOLO feature extractor...")
            feature_extractor = YOLOFeatureExtractor(yolo_model_path)
            feature_type = "yolo"
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            feature_extractor = None
    
    # Fall back to CNN if YOLO not available
    if feature_extractor is None:
        logger.info("Loading CNN feature extractor...")
        feature_extractor = GeneralCrustaceanFeatureExtractor('resnet50')
        feature_type = "cnn"
    
    # Load regression model
    # Look for the most recent model file
    model_files = list(models_dir.glob(f"molt_regressor_{feature_type}_*.joblib"))
    
    if model_files:
        model_path = sorted(model_files)[-1]  # Get most recent
        logger.info(f"Loading regression model from {model_path}")
        
        algorithm = model_path.stem.split('_')[-1]
        regressor = MoltPhaseRegressor(algorithm)
        regressor.load_model(model_path)
    else:
        logger.error(f"No regression model found for {feature_type} features")
        logger.error("Please run 'python train_model.py' first")


def get_molt_phase_category(days_until_molt: float) -> Dict[str, any]:
    """
    Convert days until molt to category and recommendation.
    
    Args:
        days_until_molt: Predicted days until molt
        
    Returns:
        Dictionary with phase, color, and recommendation
    """
    if days_until_molt < 0:
        return {
            'phase': 'Post-molt',
            'color': 'success',
            'recommendation': 'Crab has recently molted. Shell is likely soft.',
            'harvest_ready': False
        }
    elif days_until_molt <= 3:
        return {
            'phase': 'Peeler (Imminent molt)',
            'color': 'danger',
            'recommendation': 'HARVEST NOW! Crab will molt within 3 days.',
            'harvest_ready': True
        }
    elif days_until_molt <= 7:
        return {
            'phase': 'Pre-molt (Near)',
            'color': 'warning',
            'recommendation': 'Monitor closely. Harvest window approaching.',
            'harvest_ready': False
        }
    elif days_until_molt <= 14:
        return {
            'phase': 'Pre-molt (Early)',
            'color': 'info',
            'recommendation': 'Check again in a week.',
            'harvest_ready': False
        }
    else:
        return {
            'phase': 'Inter-molt',
            'color': 'primary',
            'recommendation': 'Crab is not close to molting.',
            'harvest_ready': False
        }


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return molt phase prediction."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess image
            image = Image.open(filepath)
            
            # Extract features
            features = feature_extractor.extract_features(filepath)
            
            # Make prediction
            if regressor is None or not regressor.is_fitted:
                return jsonify({'error': 'Model not loaded or trained'}), 500
            
            days_until_molt = regressor.predict(features.reshape(1, -1))[0]
            
            # Get phase category and recommendation
            phase_info = get_molt_phase_category(days_until_molt)
            
            # Create thumbnail for response
            image.thumbnail((300, 300))
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Prepare response
            response = {
                'success': True,
                'days_until_molt': float(days_until_molt),
                'phase': phase_info['phase'],
                'color': phase_info['color'],
                'recommendation': phase_info['recommendation'],
                'harvest_ready': phase_info['harvest_ready'],
                'confidence': 'High' if abs(days_until_molt) < 20 else 'Medium',
                'thumbnail': f"data:image/jpeg;base64,{img_base64}",
                'feature_type': feature_type.upper()
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'feature_extractor': feature_extractor is not None,
        'regressor': regressor is not None and regressor.is_fitted,
        'feature_type': feature_type
    })


if __name__ == '__main__':
    # Load models
    logger.info("Loading models...")
    load_models()
    
    if feature_extractor is None or regressor is None:
        logger.error("Failed to load models. Please ensure models are trained.")
        sys.exit(1)
    
    # Run app
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)