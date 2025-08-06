"""
Feature extraction using YOLO model pre-trained on marine species.

This module:
- Loads the pre-trained YOLO model
- Extracts penultimate layer features from crab images
- Provides utilities for batch feature extraction
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOFeatureExtractor:
    """
    Extract features from crab images using a pre-trained YOLO model.
    
    The features are extracted from the penultimate layer of the model,
    which contains high-level representations suitable for transfer learning.
    """
    
    def __init__(self, model_path: Union[str, Path], device: Optional[str] = None):
        """
        Initialize the feature extractor with a pre-trained YOLO model.
        
        Args:
            model_path: Path to the pre-trained YOLO model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = self._load_model()
        
        # Hook to capture penultimate layer features
        self.features = None
        self._register_hook()
        
    def _load_model(self) -> YOLO:
        """Load the pre-trained YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        logger.info(f"Loading YOLO model from: {self.model_path}")
        model = YOLO(str(self.model_path))
        model.to(self.device)
        return model
        
    def _register_hook(self):
        """
        Register a forward hook to capture features from the penultimate layer.
        
        For YOLOv8, we'll capture features from the backbone's final layer
        before the detection heads.
        """
        def hook_fn(module, input, output):
            """Hook function to capture features."""
            # Store features - we'll need to handle different output types
            if isinstance(output, torch.Tensor):
                self.features = output.detach()
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                # For multi-scale outputs, take the smallest spatial resolution (highest level features)
                self.features = output[-1].detach() if isinstance(output[-1], torch.Tensor) else output[0].detach()
                
        # Register hook on the backbone's final layer
        # For YOLOv8, this is typically model.model.model[-2] (before detection heads)
        try:
            # Try to access the backbone
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, 'model'):
                    # YOLOv8 structure
                    backbone_layers = self.model.model.model
                    # Get the last layer before detection heads (usually -2 or -3)
                    target_layer = backbone_layers[-2]
                    target_layer.register_forward_hook(hook_fn)
                    logger.info("Registered feature extraction hook on backbone")
                else:
                    logger.warning("Could not find expected model structure")
        except Exception as e:
            logger.error(f"Error registering hook: {e}")
            
    def extract_features(self, image: Union[np.ndarray, Image.Image, str, Path], 
                        input_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: Input image (numpy array, PIL Image, or path to image)
            input_size: Size to resize image to before feature extraction
            
        Returns:
            Feature vector as numpy array
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
            
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Ensure image is in correct format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Resize image
        image = cv2.resize(image, input_size)
        
        # Run inference to trigger feature extraction
        with torch.no_grad():
            # Reset features
            self.features = None
            
            # Run model (we don't need the detection results, just the features)
            _ = self.model(image, verbose=False)
            
            if self.features is None:
                logger.warning("No features captured. Using alternative extraction method.")
                # Alternative: use model prediction with intermediate outputs
                return self._extract_features_alternative(image)
                
            # Process captured features
            features = self.features
            
            # Global average pooling if features are spatial
            if len(features.shape) == 4:  # B x C x H x W
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.squeeze(-1).squeeze(-1)
            elif len(features.shape) == 3:  # B x N x C (transformer-like)
                features = features.mean(dim=1)
                
            # Convert to numpy and flatten
            features = features.cpu().numpy()
            if len(features.shape) > 1:
                features = features.reshape(features.shape[0], -1)[0]  # Take first batch item
            
            return features
            
    def _extract_features_alternative(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative feature extraction method using model embeddings.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Feature vector
        """
        # Use YOLO's built-in embedding extraction if available
        try:
            # Get model predictions with embeddings
            results = self.model(image, verbose=False)
            
            # Try to extract embeddings from results
            if hasattr(results[0], 'probs') and results[0].probs is not None:
                # Classification model - use logits/probs as features
                features = results[0].probs.data.cpu().numpy()
            else:
                # Detection model - use box features or create a simple representation
                # This is a fallback - aggregate detected objects
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Use confidence scores and box areas as simple features
                    confs = boxes.conf.cpu().numpy()
                    if boxes.xywh is not None:
                        areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy()
                        features = np.concatenate([confs, areas])
                    else:
                        features = confs
                else:
                    # No detections - return zero features
                    features = np.zeros(256)  # Default feature size
                    
            return features.flatten()
            
        except Exception as e:
            logger.error(f"Alternative feature extraction failed: {e}")
            # Return zero features as last resort
            return np.zeros(256)
            
    def extract_features_batch(self, image_paths: List[Union[str, Path]], 
                             batch_size: int = 16) -> np.ndarray:
        """
        Extract features from multiple images in batches.
        
        Args:
            image_paths: List of paths to images
            batch_size: Number of images to process at once
            
        Returns:
            Feature matrix of shape (n_images, n_features)
        """
        features_list = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + batch_size]
            
            for path in batch_paths:
                try:
                    features = self.extract_features(path)
                    features_list.append(features)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    # Add zero features for failed images
                    features_list.append(np.zeros_like(features_list[0]) if features_list else np.zeros(256))
                    
        return np.vstack(features_list)


class GeneralCrustaceanFeatureExtractor:
    """
    Extract features using a general pre-trained model that includes crustaceans.
    
    This uses a standard CNN model (e.g., ResNet) pre-trained on ImageNet,
    which includes some crustacean classes.
    """
    
    def __init__(self, model_name: str = 'resnet50', device: Optional[str] = None):
        """
        Initialize the feature extractor with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model ('resnet50', 'resnet101', 'vit_base')
            device: Device to run model on
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained model
        self.model, self.preprocess = self._load_model()
        
    def _load_model(self):
        """Load a pre-trained CNN model."""
        try:
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # Load pre-trained model
            if self.model_name == 'resnet50':
                model = models.resnet50(pretrained=True)
                # Remove the final classification layer
                model = nn.Sequential(*list(model.children())[:-1])
                # Define preprocessing
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            elif self.model_name == 'resnet101':
                model = models.resnet101(pretrained=True)
                # Remove the final classification layer
                model = nn.Sequential(*list(model.children())[:-1])
                # Define preprocessing
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            elif self.model_name == 'vit_base':
                # Use Vision Transformer
                try:
                    model = models.vit_b_16(pretrained=True)
                    # Remove classifier head, keep feature representation
                    model.heads = nn.Identity()
                    # ViT preprocessing
                    preprocess = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                except AttributeError:
                    # Fallback if ViT not available in this torchvision version
                    logger.warning("ViT not available in this torchvision version, falling back to ResNet50")
                    model = models.resnet50(pretrained=True)
                    model = nn.Sequential(*list(model.children())[:-1])
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            model.to(self.device)
            model.eval()
            
            return model, preprocess
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def extract_features(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector as numpy array
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
            features = features.squeeze().cpu().numpy()
            
        return features
        
    def extract_features_batch(self, image_paths: List[Union[str, Path]], 
                             batch_size: int = 16) -> np.ndarray:
        """
        Extract features from multiple images in batches.
        
        Args:
            image_paths: List of paths to images
            batch_size: Number of images to process at once
            
        Returns:
            Feature matrix of shape (n_images, n_features)
        """
        features_list = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess batch
            batch_tensors = []
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.preprocess(image).unsqueeze(0)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    
            if batch_tensors:
                # Stack tensors and move to device
                batch = torch.cat(batch_tensors, dim=0).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.model(batch)
                    features = features.squeeze().cpu().numpy()
                    
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                    
                features_list.append(features)
                
        return np.vstack(features_list) if features_list else np.array([])


def main():
    """Test feature extraction."""
    # Test with a sample image
    from data_loader import GreenCrabDataLoader
    
    # Load dataset
    loader = GreenCrabDataLoader("/Users/gen/green_crabs")
    df = loader.load_dataset()
    
    # Test YOLO feature extraction (if model exists)
    yolo_model_path = Path("/Users/genp/BarderryAppliedResearch/FathomNet/qscp/jupyter_notebooks/fathomverse_detector/fathomverse-only-imgs_update_to_FathomNet-NoGameLabels-2024-09-28-model_yolo8_epochs_10_2024-10-22.pt")
    
    if yolo_model_path.exists():
        print("Testing YOLO feature extraction...")
        yolo_extractor = YOLOFeatureExtractor(yolo_model_path)
        
        # Extract features from first image
        first_image_path = df.iloc[0]['image_path']
        features = yolo_extractor.extract_features(first_image_path)
        print(f"YOLO features shape: {features.shape}")
    else:
        print(f"YOLO model not found at: {yolo_model_path}")
        
    # Test general CNN feature extraction
    print("\nTesting general CNN feature extraction...")
    cnn_extractor = GeneralCrustaceanFeatureExtractor('resnet50')
    
    # Extract features from first image
    first_image_path = df.iloc[0]['image_path']
    features = cnn_extractor.extract_features(first_image_path)
    print(f"CNN features shape: {features.shape}")
    

if __name__ == "__main__":
    main()