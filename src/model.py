"""
Molt phase regression model for green crabs.

This module implements:
- Regression models to predict days until molt
- Transfer learning from YOLO features
- Model training and evaluation utilities
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import joblib
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrabMoltDataset(Dataset):
    """PyTorch dataset for crab molt regression."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values (days until molt)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class MoltPhaseNN(nn.Module):
    """Neural network for molt phase regression."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
        """
        super(MoltPhaseNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MoltPhaseRegressor:
    """
    Main class for molt phase regression.
    
    Supports multiple algorithms and provides utilities for
    training, evaluation, and prediction.
    """
    
    def __init__(self, algorithm: str = 'random_forest'):
        """
        Initialize the regressor.
        
        Args:
            algorithm: Algorithm to use ('random_forest', 'gradient_boost', 
                      'ridge', 'lasso', 'elastic_net', 'svr', 'neural_net')
        """
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize model based on algorithm
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model based on selected algorithm."""
        if self.algorithm == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.algorithm == 'gradient_boost':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.algorithm == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.algorithm == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif self.algorithm == 'elastic_net':
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif self.algorithm == 'svr':
            self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif self.algorithm == 'neural_net':
            # Will be initialized in fit() when we know input dimensions
            self.model = None
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
    def prepare_data(self, features: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            features: Feature matrix
            metadata: DataFrame with molt information
            
        Returns:
            Tuple of (features, targets, sample_indices)
        """
        # Filter samples with known molt dates
        has_molt_date = metadata['days_until_molt'].notna()
        
        # Get features and targets for samples with molt dates
        X = features[has_molt_date]
        y = metadata.loc[has_molt_date, 'days_until_molt'].values
        indices = np.where(has_molt_date)[0]
        
        logger.info(f"Prepared {len(X)} samples with molt dates for training")
        
        return X, y, indices
        
    def train_neural_net(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Train neural network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = MoltPhaseNN(input_dim)
        
        # Create datasets
        train_dataset = CrabMoltDataset(X_train, y_train)
        val_dataset = CrabMoltDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for features, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(features).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * features.size(0)
                
            train_loss /= len(train_dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            with torch.no_grad():
                for features, targets in val_loader:
                    outputs = self.model(features).squeeze()
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * features.size(0)
                    val_mae += torch.abs(outputs - targets).sum().item()
                    
            val_loss /= len(val_dataset)
            val_mae /= len(val_dataset)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.2f}")
                
        return history
        
    def fit(self, features: np.ndarray, metadata: pd.DataFrame, 
            test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train the regression model.
        
        Args:
            features: Feature matrix
            metadata: DataFrame with molt information
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Prepare data
        X, y, _ = self.prepare_data(features, metadata)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training {self.algorithm} model...")
        
        if self.algorithm == 'neural_net':
            # Further split for validation
            X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=random_state
            )
            history = self.train_neural_net(X_train_nn, y_train_nn, X_val_nn, y_val_nn)
            
            # Final predictions
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()
        else:
            # Train sklearn model
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            
        self.is_fitted = True
        
        # Evaluate
        metrics = self.evaluate(y_test, y_pred)
        
        # Log results
        logger.info(f"Model performance: MAE={metrics['mae']:.2f}, "
                   f"RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        
        return metrics
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'median_ae': np.median(np.abs(y_true - y_pred))
        }
        
        return metrics
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted days until molt
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        if self.algorithm == 'neural_net':
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(torch.FloatTensor(features_scaled)).squeeze().numpy()
        else:
            predictions = self.model.predict(features_scaled)
            
        return predictions
        
    def save_model(self, path: Path):
        """Save the trained model."""
        model_data = {
            'algorithm': self.algorithm,
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        
        if self.algorithm == 'neural_net':
            # Save PyTorch model separately
            torch.save(self.model.state_dict(), path.with_suffix('.pth'))
            model_data['model'] = None  # Don't pickle the PyTorch model
            
        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")
        
    def load_model(self, path: Path):
        """Load a trained model."""
        model_data = joblib.load(path)
        
        # Handle both dictionary format and direct model format
        if isinstance(model_data, dict):
            self.algorithm = model_data['algorithm']
            self.scaler = model_data.get('scaler', StandardScaler())
            self.is_fitted = model_data.get('is_fitted', True)
            self.model = model_data.get('model')
        else:
            # Direct model object (legacy format)
            self.model = model_data
            # Try to load scaler if it exists
            scaler_path = path.parent / f"{path.stem.replace('regressor', 'scaler')}.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = StandardScaler()
            self.is_fitted = True
            return  # Exit early for legacy format
        
        if self.algorithm == 'neural_net':
            # Load PyTorch model
            # Need to initialize model first (we'll need to know input dim)
            # This is a limitation - we'd need to save input dim too
            pth_path = path.with_suffix('.pth')
            if pth_path.exists():
                # Load state dict after model is initialized
                # (would need to be done after knowing input dimensions)
                pass
        else:
            self.model = model_data['model']
            
        logger.info(f"Loaded model from {path}")


def plot_model_comparison(features: np.ndarray, metadata: pd.DataFrame, 
                         save_path: Optional[Path] = None):
    """
    Compare different regression algorithms.
    
    Args:
        features: Feature matrix
        metadata: DataFrame with molt information
        save_path: Optional path to save plot
    """
    algorithms = ['random_forest', 'gradient_boost', 'ridge', 'svr']
    results = []
    
    for algorithm in algorithms:
        logger.info(f"Training {algorithm}...")
        regressor = MoltPhaseRegressor(algorithm)
        metrics = regressor.fit(features, metadata)
        
        results.append({
            'Algorithm': algorithm,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R²': metrics['r2']
        })
        
    # Create comparison plot
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE comparison
    axes[0].bar(results_df['Algorithm'], results_df['MAE'])
    axes[0].set_title('Mean Absolute Error (days)')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # RMSE comparison
    axes[1].bar(results_df['Algorithm'], results_df['RMSE'])
    axes[1].set_title('Root Mean Squared Error (days)')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[2].bar(results_df['Algorithm'], results_df['R²'])
    axes[2].set_title('R² Score')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
        
    plt.show()
    
    return results_df


def create_prediction_visualization(regressor: MoltPhaseRegressor, 
                                  features: np.ndarray, 
                                  metadata: pd.DataFrame,
                                  save_path: Optional[Path] = None):
    """
    Visualize model predictions vs actual values.
    
    Args:
        regressor: Trained regression model
        features: Feature matrix
        metadata: DataFrame with molt information
        save_path: Optional path to save plot
    """
    # Prepare data
    X, y_true, indices = regressor.prepare_data(features, metadata)
    
    # Make predictions
    y_pred = regressor.predict(X)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Predicted vs Actual scatter plot
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Days Until Molt')
    ax.set_ylabel('Predicted Days Until Molt')
    ax.set_title('Predicted vs Actual Values')
    
    # 2. Residuals plot
    ax = axes[0, 1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Days Until Molt')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    
    # 3. Distribution of residuals
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, edgecolor='black')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Residuals')
    
    # 4. Error by actual value
    ax = axes[1, 1]
    abs_errors = np.abs(residuals)
    ax.scatter(y_true, abs_errors, alpha=0.6)
    ax.set_xlabel('Actual Days Until Molt')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Error vs Actual Value')
    
    plt.suptitle(f'{regressor.algorithm.replace("_", " ").title()} Model Predictions', 
                 fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction visualization to {save_path}")
        
    plt.show()


def main():
    """Test the regression models."""
    logger.info("Model module loaded successfully")
    

if __name__ == "__main__":
    main()