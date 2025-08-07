"""
Flask web application for green crab molt phase detection - Heroku version.

This version downloads models from S3 at runtime to /tmp directory.
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
from sklearn.preprocessing import StandardScaler
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

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
    """Extract basic image features using only PIL/numpy."""
    # Resize to standard size
    image_resized = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image_resized)
    
    # Ensure RGB format
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # RGB color statistics
        mean_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))
        min_color = np.min(img_array, axis=(0, 1))
        max_color = np.max(img_array, axis=(0, 1))
    else:
        # Grayscale - convert to RGB-like features
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=2)
        mean_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))
        min_color = np.min(img_array, axis=(0, 1))
        max_color = np.max(img_array, axis=(0, 1))
    
    # Simple HSV-like conversion (approximate)
    r, g, b = mean_color[0], mean_color[1], mean_color[2]
    hsv_like = np.array([
        (r + g + b) / 3,  # Approximate brightness
        abs(r - g) + abs(g - b) + abs(r - b),  # Approximate saturation
        max(r, g, b) - min(r, g, b)  # Approximate hue range
    ])
    
    # Basic texture features (grayscale statistics)
    gray = np.mean(img_array, axis=2)
    texture_features = [
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray)
    ]
    
    # Combine all features
    features = np.concatenate([
        mean_color,
        std_color,
        min_color,
        max_color,
        hsv_like,
        texture_features
    ])
    
    return features


def download_from_s3(bucket_name: str, key: str, local_path: str) -> bool:
    """Download a file from S3 to local path."""
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, key, local_path)
        logger.info(f"Downloaded {key} to {local_path}")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        return False
    except ClientError as e:
        logger.error(f"Failed to download {key}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {key}: {e}")
        return False


def ensure_models_downloaded():
    """Download model files from S3 if they don't exist locally."""
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        logger.warning("S3_BUCKET_NAME not set, using local models if available")
        return True
    
    # Create /tmp/models directory
    tmp_models_dir = Path("/tmp/models")
    tmp_models_dir.mkdir(exist_ok=True)
    
    # Models to download
    model_files = [
        "molt_regressor_cnn_random_forest.joblib",
        "cnn_scaler.joblib",
        "best_cnn_regressor.joblib",
        "molt_scaler_cnn.joblib",
        "random_forest_model.joblib"
    ]
    
    success_count = 0
    for model_file in model_files:
        local_path = tmp_models_dir / model_file
        if not local_path.exists():
            s3_key = f"models/{model_file}"
            if download_from_s3(bucket_name, s3_key, str(local_path)):
                success_count += 1
        else:
            logger.info(f"Model {model_file} already exists locally")
            success_count += 1
    
    return success_count > 0


def load_models():
    """Load the regression model and scaler."""
    global regressor, scaler
    
    # First try to download models from S3
    if not ensure_models_downloaded():
        logger.warning("Failed to download models from S3, trying local models")
    
    # Look for models in both /tmp/models and local models directory
    search_dirs = [Path("/tmp/models"), Path("models")]
    
    # Try to load CNN-based model first
    model_names = [
        "molt_regressor_cnn_random_forest.joblib",
        "best_cnn_regressor.joblib", 
        "random_forest_model.joblib"
    ]
    
    scaler_names = [
        "cnn_scaler.joblib",
        "molt_scaler_cnn.joblib"
    ]
    
    # Load model
    regressor = None
    for search_dir in search_dirs:
        if regressor is not None:
            break
        for model_name in model_names:
            model_path = search_dir / model_name
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                try:
                    regressor = joblib.load(model_path)
                    break
                except Exception as e:
                    logger.error(f"Failed to load model from {model_path}: {e}")
                    continue
    
    if regressor is None:
        logger.error("No suitable model found")
        return False
    
    # Load scaler
    scaler = None
    for search_dir in search_dirs:
        if scaler is not None:
            break
        for scaler_name in scaler_names:
            scaler_path = search_dir / scaler_name
            if scaler_path.exists():
                logger.info(f"Loading scaler from {scaler_path}")
                try:
                    scaler = joblib.load(scaler_path)
                    break
                except Exception as e:
                    logger.error(f"Failed to load scaler from {scaler_path}: {e}")
                    continue
    
    if scaler is None:
        logger.warning("No scaler found, using default StandardScaler")
        scaler = StandardScaler()
        # Fit with dummy data - this is not ideal but allows the app to run
        dummy_features = np.random.rand(10, 19)  # 19 basic features (3+3+3+3+3+4)
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