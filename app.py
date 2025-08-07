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

# Patch for torch/transformers compatibility issue
import torch
if not hasattr(torch, 'uint64'):
    torch.uint64 = torch.int64  # Workaround for compatibility
if not hasattr(torch, 'uint32'):
    torch.uint32 = torch.int32  # Workaround for compatibility
if not hasattr(torch, 'uint16'):
    torch.uint16 = torch.int16  # Workaround for compatibility

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
    
    # Load VIT feature extractor
    logger.info("Loading VIT feature extractor...")
    feature_extractor = GeneralCrustaceanFeatureExtractor('vit_base')
    feature_type = "vit"
    
    # Load temporal model with VIT features
    # First check for VIT-specific temporal model
    vit_temporal_path = models_dir / "molt_regressor_vit_temporal.joblib"
    vit_random_forest_path = models_dir / "molt_regressor_vit_random_forest.joblib"
    temporal_model_path = models_dir / "temporal" / "Random_Forest_Temporal.pkl"
    
    if vit_temporal_path.exists():
        model_path = vit_temporal_path
        logger.info(f"Loading VIT temporal model from {model_path}")
    elif vit_random_forest_path.exists():
        model_path = vit_random_forest_path
        logger.info(f"Loading VIT random forest model from {model_path}")
    elif temporal_model_path.exists():
        # Use the temporal model but we need to ensure it has VIT scaler
        model_path = temporal_model_path
        logger.info(f"Loading temporal model from {model_path}")
    else:
        logger.error("No VIT or temporal model found")
        logger.error("Available models should be in models/molt_regressor_vit_*.joblib or models/temporal/")
        return
    
    # Load the model
    try:
        regressor = MoltPhaseRegressor('random_forest')
        regressor.load_model(model_path)
        
        # If using temporal model, ensure we have the VIT scaler
        if 'temporal' in str(model_path).lower() and not hasattr(regressor.scaler, 'mean_'):
            vit_scaler_path = models_dir / "vit_scaler.joblib"
            if vit_scaler_path.exists():
                import joblib
                regressor.scaler = joblib.load(vit_scaler_path)
                logger.info("Loaded VIT scaler for temporal model")
        
        logger.info(f"Successfully loaded model with VIT features")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        regressor = None


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
    app.run(debug=True, host='0.0.0.0', port=5001)