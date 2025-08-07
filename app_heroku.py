"""
Flask web application for green crab molt phase detection - Heroku version.

This simplified version uses basic CNN features without PyTorch/transformers.
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
import joblib
import cv2
from sklearn.preprocessing import StandardScaler

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
regressor = None
scaler = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_basic_features(image: Image.Image) -> np.ndarray:
    """Extract basic image features without deep learning models."""
    # Convert to numpy array
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to standard size
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Basic color statistics
    mean_color = np.mean(img_resized, axis=(0, 1))
    std_color = np.std(img_resized, axis=(0, 1))
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))
    std_hsv = np.std(hsv, axis=(0, 1))
    
    # Basic texture features (simple statistical measures)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    texture_features = [
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray)
    ]
    
    # Combine all features
    features = np.concatenate([
        mean_color.flatten(),
        std_color.flatten(),
        mean_hsv.flatten(),
        std_hsv.flatten(),
        texture_features
    ])
    
    return features


def load_models():
    """Load the regression model and scaler."""
    global regressor, scaler
    
    models_dir = Path("models")
    
    # Try to load CNN-based model first
    model_paths = [
        models_dir / "molt_regressor_cnn_random_forest.joblib",
        models_dir / "best_cnn_regressor.joblib",
        models_dir / "random_forest_model.joblib"
    ]
    
    scaler_paths = [
        models_dir / "cnn_scaler.joblib",
        models_dir / "molt_scaler_cnn.joblib"
    ]
    
    # Load model
    regressor = None
    for model_path in model_paths:
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            regressor = joblib.load(model_path)
            break
    
    if regressor is None:
        logger.error("No suitable model found")
        return False
    
    # Load scaler
    scaler = None
    for scaler_path in scaler_paths:
        if scaler_path.exists():
            logger.info(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
            break
    
    if scaler is None:
        logger.warning("No scaler found, using default StandardScaler")
        scaler = StandardScaler()
        # Fit with dummy data - this is not ideal but allows the app to run
        dummy_features = np.random.rand(10, 16)  # 16 basic features
        scaler.fit(dummy_features)
    
    logger.info("Models loaded successfully")
    return True


def get_molt_phase_category(days_until_molt: float) -> Dict[str, any]:
    """Convert days until molt to category and recommendation."""
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
            'recommendation': 'Keep monitoring. Still some time before molt.',
            'harvest_ready': False
        }
    else:
        return {
            'phase': 'Intermolt',
            'color': 'secondary',
            'recommendation': 'Not ready for harvest. Continue monitoring.',
            'harvest_ready': False
        }


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    try:
        if regressor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Load and process image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract features
        features = extract_basic_features(image)
        features = features.reshape(1, -1)
        
        # Scale features
        if scaler:
            features = scaler.transform(features)
        
        # Make prediction
        days_until_molt = regressor.predict(features)[0]
        
        # Get molt phase category
        phase_info = get_molt_phase_category(days_until_molt)
        
        # Convert image to base64 for display
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'days_until_molt': float(days_until_molt),
            'phase': phase_info['phase'],
            'color': phase_info['color'],
            'recommendation': phase_info['recommendation'],
            'harvest_ready': phase_info['harvest_ready'],
            'image': img_base64,
            'confidence': 0.75  # Placeholder confidence
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': regressor is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    logger.info("Loading models...")
    if load_models():
        logger.info("Starting Flask application...")
        # Use different ports for local vs Heroku
        port = int(os.environ.get('PORT', 5001))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Failed to load models. Exiting.")
        sys.exit(1)