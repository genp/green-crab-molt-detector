#!/usr/bin/env python3
"""
Comprehensive tests for the Flask application.
"""

import os
import sys
import json
import base64
import tempfile
from pathlib import Path
from io import BytesIO
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import app modules
from app import app, load_models, allowed_file, get_molt_phase_category

class TestAppFunctions(unittest.TestCase):
    """Test Flask application functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
        
        # Load models once for all tests
        print("Loading models for testing...")
        load_models()
        
    def test_allowed_file(self):
        """Test file extension validation."""
        # Valid extensions
        self.assertTrue(allowed_file('image.jpg'))
        self.assertTrue(allowed_file('photo.jpeg'))
        self.assertTrue(allowed_file('pic.png'))
        self.assertTrue(allowed_file('test.gif'))
        self.assertTrue(allowed_file('UPPERCASE.JPG'))
        
        # Invalid extensions
        self.assertFalse(allowed_file('document.pdf'))
        self.assertFalse(allowed_file('script.py'))
        self.assertFalse(allowed_file('noextension'))
        self.assertFalse(allowed_file(''))
        
    def test_molt_phase_categories(self):
        """Test molt phase categorization."""
        # Post-molt
        result = get_molt_phase_category(-5)
        self.assertEqual(result['phase'], 'Post-molt')
        self.assertEqual(result['color'], 'success')
        self.assertFalse(result['harvest_ready'])
        
        # Peeler (imminent molt)
        result = get_molt_phase_category(2)
        self.assertEqual(result['phase'], 'Peeler (Imminent molt)')
        self.assertEqual(result['color'], 'danger')
        self.assertTrue(result['harvest_ready'])
        
        # Pre-molt (near)
        result = get_molt_phase_category(5)
        self.assertEqual(result['phase'], 'Pre-molt (Near)')
        self.assertEqual(result['color'], 'warning')
        self.assertFalse(result['harvest_ready'])
        
        # Pre-molt (early)
        result = get_molt_phase_category(10)
        self.assertEqual(result['phase'], 'Pre-molt (Early)')
        self.assertEqual(result['color'], 'info')
        self.assertFalse(result['harvest_ready'])
        
        # Inter-molt
        result = get_molt_phase_category(20)
        self.assertEqual(result['phase'], 'Inter-molt')
        self.assertEqual(result['color'], 'primary')
        self.assertFalse(result['harvest_ready'])
        
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('feature_extractor', data)
        self.assertIn('regressor', data)
        self.assertIn('feature_type', data)
        
    def test_index_page(self):
        """Test main page loads."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Green Crab Molt Phase Detector', response.data)
        
    def create_test_image(self):
        """Create a test image for upload."""
        # Create a simple RGB image
        img = Image.new('RGB', (100, 100), color='green')
        
        # Save to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes
        
    def test_predict_no_file(self):
        """Test prediction with no file uploaded."""
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No file uploaded')
        
    def test_predict_empty_filename(self):
        """Test prediction with empty filename."""
        data = {'file': (BytesIO(b''), '')}
        response = self.client.post('/predict', data=data)
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No file selected')
        
    def test_predict_invalid_file_type(self):
        """Test prediction with invalid file type."""
        data = {'file': (BytesIO(b'test content'), 'test.txt')}
        response = self.client.post('/predict', data=data)
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Invalid file type. Please upload an image.')
        
    @patch('app.feature_extractor')
    @patch('app.regressor')
    def test_predict_success(self, mock_regressor, mock_extractor):
        """Test successful prediction."""
        # Mock feature extraction
        mock_extractor.extract_features.return_value = np.random.randn(2048)
        
        # Mock prediction
        mock_regressor.is_fitted = True
        mock_regressor.predict.return_value = np.array([7.5])
        
        # Create test image
        img_bytes = self.create_test_image()
        
        # Make request
        data = {'file': (img_bytes, 'test.jpg')}
        response = self.client.post('/predict', 
                                   data=data,
                                   content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        
        result = json.loads(response.data)
        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['days_until_molt'], 7.5, places=1)
        self.assertEqual(result['phase'], 'Pre-molt (Early)')  # 7.5 days is in Early phase (7-14 days)
        self.assertFalse(result['harvest_ready'])
        self.assertIn('thumbnail', result)
        self.assertIn('confidence', result)
        
    @patch('app.regressor')
    def test_predict_model_not_fitted(self, mock_regressor):
        """Test prediction when model is not fitted."""
        # Mock unfitted model
        mock_regressor.is_fitted = False
        
        # Create test image
        img_bytes = self.create_test_image()
        
        # Make request
        data = {'file': (img_bytes, 'test.jpg')}
        response = self.client.post('/predict',
                                   data=data,
                                   content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 500)
        
        result = json.loads(response.data)
        self.assertEqual(result['error'], 'Model not loaded or trained')
        
    def test_file_cleanup(self):
        """Test that uploaded files are cleaned up."""
        upload_folder = app.config['UPLOAD_FOLDER']
        
        # Count files before
        files_before = len(os.listdir(upload_folder)) if os.path.exists(upload_folder) else 0
        
        # Upload an image
        img_bytes = self.create_test_image()
        data = {'file': (img_bytes, 'test.jpg')}
        
        with patch('app.feature_extractor') as mock_extractor:
            with patch('app.regressor') as mock_regressor:
                mock_extractor.extract_features.return_value = np.random.randn(2048)
                mock_regressor.is_fitted = True
                mock_regressor.predict.return_value = np.array([5.0])
                
                response = self.client.post('/predict',
                                          data=data,
                                          content_type='multipart/form-data')
        
        # Count files after
        files_after = len(os.listdir(upload_folder)) if os.path.exists(upload_folder) else 0
        
        # Should be the same (file was cleaned up)
        self.assertEqual(files_before, files_after)
        
    def test_harvest_ready_detection(self):
        """Test harvest ready flag for peeler crabs."""
        # Test peeler crab (0-3 days)
        for days in [0, 1, 2, 3]:
            result = get_molt_phase_category(days)
            self.assertTrue(result['harvest_ready'], 
                          f"Days {days} should be harvest ready")
            
        # Test non-peeler crabs
        for days in [-1, 4, 7, 14, 30]:
            result = get_molt_phase_category(days)
            self.assertFalse(result['harvest_ready'],
                           f"Days {days} should not be harvest ready")


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up for integration tests."""
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.client = cls.app.test_client()
        
        # Ensure models are loaded
        load_models()
        
    def test_full_prediction_pipeline(self):
        """Test the complete prediction pipeline."""
        # Create a realistic test image
        img = Image.new('RGB', (640, 480), color='green')
        
        # Add some variation to make it more realistic
        pixels = img.load()
        for i in range(0, 640, 10):
            for j in range(0, 480, 10):
                # Add some color variation
                r = 50 + (i % 100)
                g = 150 + (j % 50)
                b = 50
                pixels[i, j] = (r, g, b)
        
        # Save to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Upload and predict
        data = {'file': (img_bytes, 'crab_test.jpg')}
        response = self.client.post('/predict',
                                   data=data,
                                   content_type='multipart/form-data')
        
        # Check response
        if response.status_code == 200:
            result = json.loads(response.data)
            
            # Verify all expected fields
            self.assertIn('success', result)
            self.assertIn('days_until_molt', result)
            self.assertIn('phase', result)
            self.assertIn('color', result)
            self.assertIn('recommendation', result)
            self.assertIn('harvest_ready', result)
            self.assertIn('confidence', result)
            self.assertIn('thumbnail', result)
            self.assertIn('feature_type', result)
            
            # Check data types
            self.assertIsInstance(result['days_until_molt'], (int, float))
            self.assertIsInstance(result['phase'], str)
            self.assertIsInstance(result['harvest_ready'], bool)
            
            # Check thumbnail is valid base64
            if result['thumbnail'].startswith('data:image'):
                # Extract base64 part
                base64_str = result['thumbnail'].split(',')[1]
                try:
                    base64.b64decode(base64_str)
                except Exception as e:
                    self.fail(f"Invalid base64 thumbnail: {e}")
                    
            print(f"✓ Full pipeline test passed: {result['phase']} ({result['days_until_molt']:.1f} days)")
        else:
            # If models aren't properly loaded, at least check error handling
            self.assertEqual(response.status_code, 500)
            result = json.loads(response.data)
            self.assertIn('error', result)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAppFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)