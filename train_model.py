"""
Train molt phase regression model for green crabs.

This script:
1. Loads extracted features (YOLO + CNN)
2. Trains multiple regression models with 5-fold CV
3. Evaluates regression and classification performance
4. Creates precision/recall figures
5. Saves the best model
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_molt_phase_bins(days_until_molt):
    """Convert continuous days_until_molt to categorical bins for classification metrics."""
    bins = [-0.1, 5, 10, 15, 20, 30]  # 0-5, 6-10, 11-15, 16-20, 21+ days
    labels = ['0-5_days', '6-10_days', '11-15_days', '16-20_days', '21+_days']
    return pd.cut(days_until_molt, bins=bins, labels=labels, include_lowest=True)


def perform_cross_validation(models, X, y, groups, cv_folds=5):
    """Perform 5-fold cross-validation for all models."""
    results = {}
    
    # Use GroupKFold to prevent data leakage (same crab in train/test)
    cv = GroupKFold(n_splits=cv_folds)
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation scores
        mae_scores = -cross_val_score(model, X, y, cv=cv, groups=groups, 
                                     scoring='neg_mean_absolute_error')
        rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, groups=groups, 
                                              scoring='neg_mean_squared_error'))
        r2_scores = cross_val_score(model, X, y, cv=cv, groups=groups, 
                                   scoring='r2')
        
        results[name] = {
            'MAE': mae_scores,
            'RMSE': rmse_scores,
            'R2': r2_scores,
            'MAE_mean': mae_scores.mean(),
            'MAE_std': mae_scores.std(),
            'RMSE_mean': rmse_scores.mean(), 
            'RMSE_std': rmse_scores.std(),
            'R2_mean': r2_scores.mean(),
            'R2_std': r2_scores.std()
        }
        
        logger.info(f"{name} - MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
        logger.info(f"{name} - RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")
        logger.info(f"{name} - R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
        
    return results


def plot_cv_results(results, save_path):
    """Create box plots of cross-validation results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['MAE', 'RMSE', 'R2']
    metric_names = ['Mean Absolute Error', 'Root Mean Squared Error', 'R² Score']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        data = []
        labels = []
        
        for name, scores in results.items():
            data.append(scores[metric])
            labels.append(name)
            
        axes[i].boxplot(data, labels=labels)
        axes[i].set_title(f'{metric_name} (5-fold CV)')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        if metric == 'MAE' or metric == 'RMSE':
            axes[i].set_ylabel('Days')
        else:
            axes[i].set_ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CV results plot to {save_path}")


def evaluate_classification_performance(models, X, y, groups, save_path):
    """Train models and evaluate classification performance using binned predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    cv = GroupKFold(n_splits=5)
    
    for idx, (name, model) in enumerate(models.items()):
        all_true_bins = []
        all_pred_bins = []
        
        # Collect predictions from all CV folds
        for train_idx, val_idx in cv.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Convert to bins for classification metrics
            true_bins = create_molt_phase_bins(y_val)
            pred_bins = create_molt_phase_bins(y_pred)
            
            all_true_bins.extend(true_bins)
            all_pred_bins.extend(pred_bins)
        
        # Calculate precision, recall, f1 with all possible labels
        all_labels = ['0-5_days', '6-10_days', '11-15_days', '16-20_days', '21+_days']
        precision, recall, f1, support = precision_recall_fscore_support(
            all_true_bins, all_pred_bins, labels=all_labels, average=None, zero_division=0
        )
        
        # Create precision-recall bar plot
        bin_labels = ['0-5d', '6-10d', '11-15d', '16-20d', '21+d']
        x_pos = np.arange(len(bin_labels))
        
        width = 0.35
        axes[idx].bar(x_pos - width/2, precision, width, label='Precision', alpha=0.8)
        axes[idx].bar(x_pos + width/2, recall, width, label='Recall', alpha=0.8)
        
        axes[idx].set_xlabel('Molt Phase Bins')
        axes[idx].set_ylabel('Score')
        axes[idx].set_title(f'{name} - Precision & Recall')
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(bin_labels)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, 1)
        
        # Add text annotations
        for i, (p, r) in enumerate(zip(precision, recall)):
            axes[idx].text(i - width/2, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
            axes[idx].text(i + width/2, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved precision/recall plot to {save_path}")


def main():
    """Train and evaluate molt phase regression models with comprehensive evaluation."""
    
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
    
    # Filter samples with valid molt dates
    valid_mask = metadata['days_until_molt'].notna()
    metadata_valid = metadata[valid_mask].copy()
    logger.info(f"Using {len(metadata_valid)} samples with valid molt dates")
    
    # Load features
    feature_sets = {}
    
    # Try to load YOLO features
    yolo_path = data_dir / "yolo_features.npy"
    if yolo_path.exists():
        yolo_features = np.load(yolo_path)[valid_mask]
        feature_sets['YOLO'] = yolo_features
        logger.info(f"Loaded YOLO features: {yolo_features.shape}")
    
    # Try to load CNN features  
    cnn_path = data_dir / "cnn_features.npy"
    if cnn_path.exists():
        cnn_features = np.load(cnn_path)[valid_mask]
        feature_sets['CNN'] = cnn_features
        logger.info(f"Loaded CNN features: {cnn_features.shape}")
    
    # Try to load ViT features
    vit_path = data_dir / "vit_features.npy"
    if vit_path.exists():
        vit_features = np.load(vit_path)[valid_mask]
        feature_sets['ViT'] = vit_features
        logger.info(f"Loaded ViT features: {vit_features.shape}")
    
    if not feature_sets:
        logger.error("No feature files found. Please run feature extraction first.")
        return
    
    # Create combined features
    if len(feature_sets) > 1:
        combined_features = np.hstack(list(feature_sets.values()))
        feature_sets['Combined'] = combined_features
        logger.info(f"Created combined features: {combined_features.shape}")
    
    # Prepare target variable
    y = metadata_valid['days_until_molt'].values
    groups = metadata_valid['crab_id'].values  # For GroupKFold
    
    logger.info(f"Target statistics - Min: {y.min():.1f}, Max: {y.max():.1f}, Mean: {y.mean():.1f}")
    
    # Test each feature set
    for feature_name, X in feature_sets.items():
        logger.info(f"\n=== Training models with {feature_name} features ===")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models
        models = {
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
            'Neural_Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Perform cross-validation
        results = perform_cross_validation(models, X_scaled, y, groups)
        
        # Plot CV results
        plot_cv_results(results, plots_dir / f"{feature_name.lower()}_cv_results.png")
        
        # Evaluate classification performance
        evaluate_classification_performance(
            models, X_scaled, y, groups, 
            plots_dir / f"{feature_name.lower()}_precision_recall.png"
        )
        
        # Train final models and save the best one
        best_model_name = min(results.keys(), key=lambda k: results[k]['MAE_mean'])
        best_model = models[best_model_name]
        
        logger.info(f"Best model for {feature_name}: {best_model_name}")
        logger.info(f"Best MAE: {results[best_model_name]['MAE_mean']:.3f} ± {results[best_model_name]['MAE_std']:.3f}")
        
        # Train on full dataset
        best_model.fit(X_scaled, y)
        
        # Save model and scaler
        model_path = models_dir / f"best_{feature_name.lower()}_regressor.joblib"
        scaler_path = models_dir / f"{feature_name.lower()}_scaler.joblib"
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Saved best model to {model_path}")
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save results summary
        results_df = pd.DataFrame({
            'Model': results.keys(),
            'MAE_mean': [results[k]['MAE_mean'] for k in results.keys()],
            'MAE_std': [results[k]['MAE_std'] for k in results.keys()],
            'RMSE_mean': [results[k]['RMSE_mean'] for k in results.keys()], 
            'RMSE_std': [results[k]['RMSE_std'] for k in results.keys()],
            'R2_mean': [results[k]['R2_mean'] for k in results.keys()],
            'R2_std': [results[k]['R2_std'] for k in results.keys()]
        })
        
        results_path = models_dir / f"{feature_name.lower()}_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results summary to {results_path}")
    
    logger.info("\n=== Model training complete! ===")
    logger.info("Check the plots/ directory for visualizations")
    logger.info("Check the models/ directory for saved models")


if __name__ == "__main__":
    main()