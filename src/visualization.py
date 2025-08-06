"""
Visualization utilities for green crab molt phase analysis.

This module provides:
- t-SNE visualization of crab images colored by molt status
- Feature space exploration tools
- Molt phase progression visualizations
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MoltPhaseVisualizer:
    """
    Visualize green crab images in feature space colored by molt phase.
    """
    
    def __init__(self, features: np.ndarray, metadata: pd.DataFrame):
        """
        Initialize the visualizer.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            metadata: DataFrame with image metadata including molt information
        """
        self.features = features
        self.metadata = metadata
        
        # Validate that features and metadata have same number of samples
        if len(features) != len(metadata):
            raise ValueError(f"Features ({len(features)}) and metadata ({len(metadata)}) must have same length")
            
    def create_tsne_embedding(self, n_components: int = 2, perplexity: float = 30.0, 
                            random_state: int = 42) -> np.ndarray:
        """
        Create t-SNE embedding of features.
        
        Args:
            n_components: Number of dimensions for embedding
            perplexity: t-SNE perplexity parameter
            random_state: Random seed for reproducibility
            
        Returns:
            t-SNE embedded features of shape (n_samples, n_components)
        """
        logger.info(f"Creating t-SNE embedding with perplexity={perplexity}")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # Create t-SNE embedding
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                    random_state=random_state, max_iter=1000)
        embedding = tsne.fit_transform(features_scaled)
        
        return embedding
        
    def plot_tsne_by_molt_status(self, embedding: np.ndarray, save_path: Optional[Path] = None):
        """
        Plot t-SNE visualization colored by molt status.
        
        Args:
            embedding: t-SNE embedded features
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. Color by is_molted (binary)
        ax = axes[0, 0]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=self.metadata['is_molted'].astype(int), 
                           cmap='RdYlBu', alpha=0.6, s=50)
        ax.set_title('t-SNE colored by Molt Status (Binary)', fontsize=14)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Molted')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Not Molted', 'Molted'])
        
        # 2. Color by days until molt (continuous)
        ax = axes[0, 1]
        # Filter out samples without molt date
        has_molt_date = self.metadata['days_until_molt'].notna()
        scatter = ax.scatter(embedding[has_molt_date, 0], embedding[has_molt_date, 1], 
                           c=self.metadata.loc[has_molt_date, 'days_until_molt'], 
                           cmap='viridis', alpha=0.6, s=50)
        ax.set_title('t-SNE colored by Days Until Molt', fontsize=14)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Days Until Molt')
        
        # 3. Color by sex
        ax = axes[1, 0]
        sex_colors = {'M': 'blue', 'F': 'red'}
        for sex, color in sex_colors.items():
            mask = self.metadata['sex'] == sex
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=color, label=f'{sex} ({mask.sum()} samples)', 
                      alpha=0.6, s=50)
        ax.set_title('t-SNE colored by Sex', fontsize=14)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend()
        
        # 4. Color by individual crab
        ax = axes[1, 1]
        # Get unique crab IDs
        unique_crabs = self.metadata['crab_id'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_crabs)))
        
        for i, crab_id in enumerate(unique_crabs):
            mask = self.metadata['crab_id'] == crab_id
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=[colors[i]], label=crab_id if i < 10 else '', 
                      alpha=0.6, s=50)
        ax.set_title('t-SNE colored by Individual Crab', fontsize=14)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        if len(unique_crabs) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved t-SNE plot to {save_path}")
            
        plt.show()
        
    def plot_molt_phase_categories(self, embedding: np.ndarray, save_path: Optional[Path] = None):
        """
        Plot t-SNE with molt phase categories based on days until molt.
        
        Categories:
        - Post-molt: days_until_molt < 0 or is_molted = True
        - Peeler (imminent): 0-3 days until molt
        - Pre-molt early: 4-10 days until molt
        - Inter-molt: > 10 days until molt or no molt date
        """
        # Create molt phase categories
        def categorize_molt_phase(row):
            if row['is_molted']:
                return 'Post-molt'
            elif pd.isna(row['days_until_molt']):
                return 'Inter-molt'
            elif row['days_until_molt'] < 0:
                return 'Post-molt'
            elif row['days_until_molt'] <= 3:
                return 'Peeler (0-3 days)'
            elif row['days_until_molt'] <= 10:
                return 'Pre-molt (4-10 days)'
            else:
                return 'Inter-molt (>10 days)'
                
        self.metadata['molt_phase'] = self.metadata.apply(categorize_molt_phase, axis=1)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Define colors for each phase
        phase_colors = {
            'Post-molt': '#2ecc71',           # Green
            'Peeler (0-3 days)': '#e74c3c',   # Red
            'Pre-molt (4-10 days)': '#f39c12', # Orange
            'Inter-molt (>10 days)': '#3498db' # Blue
        }
        
        # Plot each phase
        for phase, color in phase_colors.items():
            mask = self.metadata['molt_phase'] == phase
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=color, label=f'{phase} (n={mask.sum()})', 
                       alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
                       
        plt.title('t-SNE Visualization of Crab Images by Molt Phase Category', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved molt phase category plot to {save_path}")
            
        plt.show()
        
    def plot_temporal_progression(self, save_path: Optional[Path] = None):
        """
        Plot temporal progression of molt phases for individual crabs.
        """
        # Select crabs that have molt dates
        crabs_with_molt = self.metadata[self.metadata['molt_date'].notna()]['crab_id'].unique()
        
        # Create subplots for first 6 crabs with molt dates
        n_crabs = min(6, len(crabs_with_molt))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, crab_id in enumerate(crabs_with_molt[:n_crabs]):
            ax = axes[i]
            
            # Get data for this crab
            crab_data = self.metadata[self.metadata['crab_id'] == crab_id].copy()
            crab_data = crab_data.sort_values('capture_date')
            
            # Plot days until molt over time
            ax.plot(crab_data['capture_date'], crab_data['days_until_molt'], 
                   'o-', markersize=8, linewidth=2)
            
            # Mark molt date
            molt_date = crab_data['molt_date'].iloc[0]
            ax.axvline(molt_date, color='red', linestyle='--', alpha=0.7, label='Molt date')
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f'{crab_id} ({"Female" if crab_id[0] == "F" else "Male"})', fontsize=12)
            ax.set_xlabel('Date')
            ax.set_ylabel('Days Until Molt')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
            
        plt.suptitle('Temporal Progression of Molt Phase for Individual Crabs', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved temporal progression plot to {save_path}")
            
        plt.show()


def create_feature_comparison_plot(yolo_features: Optional[np.ndarray], 
                                 cnn_features: Optional[np.ndarray], 
                                 metadata: pd.DataFrame,
                                 save_path: Optional[Path] = None):
    """
    Compare t-SNE embeddings from different feature extractors.
    
    Args:
        yolo_features: Features from YOLO model
        cnn_features: Features from general CNN model
        metadata: DataFrame with image metadata
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Helper function to plot t-SNE
    def plot_embedding(features, ax, title):
        if features is None:
            ax.text(0.5, 0.5, 'Features not available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            return
            
        # Create t-SNE embedding
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding = tsne.fit_transform(features_scaled)
        
        # Color by days until molt
        has_molt_date = metadata['days_until_molt'].notna()
        scatter = ax.scatter(embedding[has_molt_date, 0], embedding[has_molt_date, 1], 
                           c=metadata.loc[has_molt_date, 'days_until_molt'], 
                           cmap='viridis', alpha=0.6, s=50)
        ax.scatter(embedding[~has_molt_date, 0], embedding[~has_molt_date, 1], 
                  c='gray', alpha=0.3, s=30, label='No molt date')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Days Until Molt')
        
    # Plot YOLO features
    plot_embedding(yolo_features, axes[0], 't-SNE of YOLO Features')
    
    # Plot CNN features
    plot_embedding(cnn_features, axes[1], 't-SNE of CNN Features')
    
    plt.suptitle('Comparison of Feature Representations', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature comparison plot to {save_path}")
        
    plt.show()


def main():
    """Test visualization functions."""
    # This is a placeholder for testing
    logger.info("Visualization module loaded successfully")
    

if __name__ == "__main__":
    main()