#!/usr/bin/env python3
"""
Temporal model training using ViT features with 7-day window for green crab molt prediction.
This script specifically trains with a 7-day observation window to capture longer-term temporal patterns.
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TemporalFeatureExtractor:
    """Extract temporal features from crab observation sequences with 7-day window."""

    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            # Prefer 2016 ViT-featured dataset if present
            default = Path("data/processed/crab_dataset_2016_vit.csv")
            dataset_path = default if default.exists() else "data/processed/crab_dataset_merged.csv"
        self.dataset_path = Path(dataset_path)
        self.df = None
        self.temporal_features = []

    def load_dataset(self) -> pd.DataFrame:
        """Load the crab dataset with extracted features."""
        print(f"Loading dataset from {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)

        # Parse dates
        self.df['capture_date'] = pd.to_datetime(self.df['capture_date'])
        self.df['molt_date'] = pd.to_datetime(self.df['molt_date'])

        print(f"Loaded {len(self.df)} records from {self.df['crab_id'].nunique()} crabs")

        # Show crab observation counts
        crab_counts = self.df.groupby('crab_id').size().sort_values(ascending=False)
        print(f"\nObservations per crab:")
        print(crab_counts)
        print(f"\nCrabs with >=7 observations: {(crab_counts >= 7).sum()}/{len(crab_counts)}")

        return self.df

    def create_temporal_sequences(self, window_size: int = 7) -> pd.DataFrame:
        """Create temporal sequences from individual observations."""
        print(f"\nCreating temporal sequences with window size {window_size}...")

        # Group by crab
        grouped = self.df.groupby('crab_id')

        sequences = []
        skipped_crabs = []

        for crab_id, crab_data in grouped:
            # Sort by capture date
            crab_data = crab_data.sort_values('capture_date')

            # Skip if not enough observations
            if len(crab_data) < window_size:
                skipped_crabs.append((crab_id, len(crab_data)))
                continue

            # Create sliding windows
            for i in range(len(crab_data) - window_size + 1):
                window = crab_data.iloc[i:i + window_size]

                # Target is days until molt at last observation
                target = window.iloc[-1]['days_until_molt']

                # Skip if target is negative (post-molt)
                if target < 0:
                    continue

                # Extract temporal features
                seq_features = {
                    'crab_id': crab_id,
                    'sex': window.iloc[0]['sex'],
                    'sequence_length': len(window),
                    'target_days_until_molt': target,
                    'first_days_until_molt': window.iloc[0]['days_until_molt'],
                    'last_days_until_molt': window.iloc[-1]['days_until_molt'],
                    'observation_span': (window.iloc[-1]['capture_date'] - window.iloc[0]['capture_date']).days,
                    'observation_frequency': len(window) / max((window.iloc[-1]['capture_date'] - window.iloc[0]['capture_date']).days, 1)
                }

                # Calculate temporal statistics for numeric features
                numeric_cols = window.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col.startswith('feature_')]

                if feature_cols:
                    # Use top 50 features for efficiency
                    for col in feature_cols[:50]:
                        # Statistical measures
                        seq_features[f'mean_{col}'] = window[col].mean()
                        seq_features[f'std_{col}'] = window[col].std()
                        seq_features[f'min_{col}'] = window[col].min()
                        seq_features[f'max_{col}'] = window[col].max()
                        seq_features[f'range_{col}'] = window[col].max() - window[col].min()

                        # Temporal dynamics
                        if len(window) > 1:
                            seq_features[f'change_{col}'] = window[col].iloc[-1] - window[col].iloc[0]
                            seq_features[f'trend_{col}'] = np.polyfit(range(len(window)), window[col].values, 1)[0]

                            # Early vs late period comparison (first 3 vs last 3 obs in 7-day window)
                            if len(window) >= 6:
                                early_mean = window[col].iloc[:3].mean()
                                late_mean = window[col].iloc[-3:].mean()
                                seq_features[f'early_late_diff_{col}'] = late_mean - early_mean

                # Add molt phase category
                if target <= 3:
                    seq_features['molt_phase'] = 'peeler'
                elif target <= 7:
                    seq_features['molt_phase'] = 'pre_molt'
                elif target <= 14:
                    seq_features['molt_phase'] = 'inter_molt_early'
                else:
                    seq_features['molt_phase'] = 'inter_molt_late'

                sequences.append(seq_features)

        if skipped_crabs:
            print(f"\nSkipped {len(skipped_crabs)} crabs with insufficient observations:")
            for crab_id, count in skipped_crabs:
                print(f"  {crab_id}: {count} observations (need {window_size})")

        seq_df = pd.DataFrame(sequences)
        print(f"\nCreated {len(seq_df)} temporal sequences from {seq_df['crab_id'].nunique()} crabs")

        # Print phase distribution
        if 'molt_phase' in seq_df.columns:
            print("\nMolt phase distribution:")
            print(seq_df['molt_phase'].value_counts())

        return seq_df


class AdvancedTemporalModels:
    """Train and evaluate advanced temporal models with 7-day window."""

    def __init__(self, sequences_df: pd.DataFrame):
        self.sequences_df = sequences_df
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_cols = []

    def prepare_data(self, test_size: float = 0.2) -> Tuple:
        """Prepare data for training."""
        # Drop sequences with missing targets
        self.sequences_df = self.sequences_df[self.sequences_df['target_days_until_molt'].notna()].copy()

        # Identify feature columns
        exclude_cols = ['crab_id', 'sex', 'sequence_length', 'target_days_until_molt',
                       'molt_phase', 'first_days_until_molt', 'last_days_until_molt']
        self.feature_cols = [col for col in self.sequences_df.columns if col not in exclude_cols]

        print(f"\nUsing {len(self.feature_cols)} temporal features")

        # Prepare features and target
        X = self.sequences_df[self.feature_cols].fillna(0).values
        y = self.sequences_df['target_days_until_molt'].values

        # Split by crab to avoid leakage
        crab_ids = self.sequences_df['crab_id'].unique()
        train_crabs, test_crabs = train_test_split(crab_ids, test_size=test_size, random_state=42)
        self.train_crabs = train_crabs
        self.test_crabs = test_crabs

        train_mask = self.sequences_df['crab_id'].isin(train_crabs)
        test_mask = self.sequences_df['crab_id'].isin(test_crabs)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Train: {len(X_train)} samples from {len(train_crabs)} crabs")
        print(f"Test: {len(X_test)} samples from {len(test_crabs)} crabs")
        print(f"Train crabs: {sorted(train_crabs)}")
        print(f"Test crabs: {sorted(test_crabs)}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple temporal models with 7-day window."""
        print("\n" + "="*80)
        print("Training temporal models with ViT features and 7-day window...")
        print("="*80)

        models = {
            'Random Forest (Temporal-ViT-7day)': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting (Temporal-ViT-7day)': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            )
        }

        best_mae = float('inf')

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Cross-validation on training set
            cv_scores = cross_val_score(model, X_train, y_train,
                                       cv=3, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()

            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mae': cv_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'y_train': y_train,
                'train_pred': train_pred,
                'y_test': y_test,
                'test_pred': test_pred
            }

            print(f"  Train MAE: {train_mae:.3f} days")
            print(f"  CV MAE: {cv_mae:.3f} days")
            print(f"  Test MAE: {test_mae:.3f} days")
            print(f"  Test RMSE: {test_rmse:.3f} days")
            print(f"  Test R²: {test_r2:.4f}")

            # Track best model
            if test_mae < best_mae:
                best_mae = test_mae
                self.best_model = name

        print(f"\n🏆 Best model: {self.best_model} with Test MAE: {best_mae:.3f} days")

    def plot_comprehensive_evaluation(self):
        """Create comprehensive evaluation plots."""
        print("\nGenerating comprehensive evaluation figures...")

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Model Performance Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        models = list(self.results.keys())

        metrics_df = pd.DataFrame({
            'Model': models,
            'Train MAE': [self.results[m]['train_mae'] for m in models],
            'CV MAE': [self.results[m]['cv_mae'] for m in models],
            'Test MAE': [self.results[m]['test_mae'] for m in models]
        })

        x = np.arange(len(models))
        width = 0.25

        ax1.bar(x - width, metrics_df['Train MAE'], width, label='Train MAE', alpha=0.8)
        ax1.bar(x, metrics_df['CV MAE'], width, label='CV MAE', alpha=0.8)
        ax1.bar(x + width, metrics_df['Test MAE'], width, label='Test MAE', alpha=0.8)

        ax1.set_xlabel('Model')
        ax1.set_ylabel('Mean Absolute Error (days)')
        ax1.set_title('7-Day Temporal-ViT Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace(' (Temporal-ViT-7day)', '') for m in models], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Highlight best model
        best_idx = models.index(self.best_model)
        ax1.axvspan(best_idx - 0.4, best_idx + 0.4, alpha=0.2, color='green')

        # 2. R² and RMSE Comparison
        ax2 = fig.add_subplot(gs[0, 2:])

        metrics_df2 = pd.DataFrame({
            'Test R²': [self.results[m]['test_r2'] for m in models],
            'Test RMSE': [self.results[m]['test_rmse'] for m in models]
        }, index=[m.replace(' (Temporal-ViT-7day)', '') for m in models])

        ax2_twin = ax2.twinx()

        bars1 = ax2.bar(x - 0.2, metrics_df2['Test R²'], 0.4, label='Test R²', color='blue', alpha=0.7)
        bars2 = ax2_twin.bar(x + 0.2, metrics_df2['Test RMSE'], 0.4, label='Test RMSE', color='red', alpha=0.7)

        ax2.set_xlabel('Model')
        ax2.set_ylabel('R² Score', color='blue')
        ax2_twin.set_ylabel('RMSE (days)', color='red')
        ax2.set_title('Model Quality Metrics (7-day window)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_df2.index, rotation=45, ha='right')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)

        # 3-4. Actual vs Predicted for each model
        for idx, (name, res) in enumerate(self.results.items()):
            ax = fig.add_subplot(gs[1, idx*2:idx*2+2])

            # Create hexbin plot for dense data
            if len(res['y_test']) > 50:
                hexbin = ax.hexbin(res['y_test'], res['test_pred'],
                                  gridsize=20, cmap='YlOrRd', mincnt=1)
                plt.colorbar(hexbin, ax=ax, label='Count')
            else:
                ax.scatter(res['y_test'], res['test_pred'], alpha=0.6, s=50)

            # Add diagonal line
            max_val = max(res['y_test'].max(), res['test_pred'].max())
            ax.plot([0, max_val], [0, max_val], 'b--', lw=2, alpha=0.7, label='Perfect prediction')

            # Add ±3 day bands
            ax.fill_between([0, max_val], [-3, max_val-3], [3, max_val+3],
                          alpha=0.2, color='green', label='±3 days')

            ax.set_xlabel('Actual Days Until Molt')
            ax.set_ylabel('Predicted Days')
            ax.set_title(f'{name.replace(" (Temporal-ViT-7day)", "")}\nMAE: {res["test_mae"]:.3f} days',
                        fontsize=10)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

            # Mark if best model
            if name == self.best_model:
                ax.set_facecolor('#f0fff0')

        # 5. Error Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        for name, res in self.results.items():
            errors = res['y_test'] - res['test_pred']
            ax5.hist(errors, bins=30, alpha=0.5, label=name.replace(' (Temporal-ViT-7day)', ''))

        ax5.set_xlabel('Prediction Error (days)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Error Distribution', fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero error')

        # 6. Performance by Days Until Molt
        ax6 = fig.add_subplot(gs[2, 1])
        bins = [0, 3, 7, 14, 21, 30, 100]
        bin_labels = ['0-3', '4-7', '8-14', '15-21', '22-30', '30+']

        for name, res in self.results.items():
            mae_by_bin = []
            for i in range(len(bins)-1):
                mask = (res['y_test'] >= bins[i]) & (res['y_test'] < bins[i+1])
                if mask.sum() > 0:
                    mae = mean_absolute_error(res['y_test'][mask], res['test_pred'][mask])
                    mae_by_bin.append(mae)
                else:
                    mae_by_bin.append(np.nan)

            ax6.plot(bin_labels, mae_by_bin, marker='o',
                    label=name.replace(' (Temporal-ViT-7day)', ''), linewidth=2)

        ax6.set_xlabel('Days Until Molt (bins)')
        ax6.set_ylabel('Mean Absolute Error (days)')
        ax6.set_title('Performance by Molt Phase', fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(bottom=0)

        # 7. Feature Importance (for best model)
        ax7 = fig.add_subplot(gs[2, 2:])
        if self.best_model and hasattr(self.models[self.best_model], 'feature_importances_'):
            importances = self.models[self.best_model].feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features

            ax7.barh(range(len(indices)), importances[indices], alpha=0.8)
            ax7.set_yticks(range(len(indices)))
            ax7.set_yticklabels([self.feature_cols[i][:35] for i in indices], fontsize=8)
            ax7.set_xlabel('Importance')
            ax7.set_title(f'Top 20 Features - {self.best_model.replace(" (Temporal-ViT-7day)", "")}',
                         fontweight='bold')
            ax7.grid(True, alpha=0.3)

        plt.suptitle('7-Day Temporal-ViT Model Evaluation - Extended Sequential Features',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save figure
        output_dir = Path("plots")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "temporal_vit_7day_evaluation.png", dpi=300, bbox_inches='tight')
        print(f"Saved evaluation figure to {output_dir / 'temporal_vit_7day_evaluation.png'}")

        plt.close()

    def save_best_model(self):
        """Save the best performing model."""
        if not self.best_model:
            print("No best model identified!")
            return

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        # Save best model
        best_model_path = model_dir / "best_temporal_model_7day.pkl"
        with open(best_model_path, 'wb') as f:
            pickle.dump({
                'model': self.models[self.best_model],
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'name': self.best_model,
                'metrics': self.results[self.best_model],
                'window_size': 7
            }, f)

        print(f"\n✅ Saved best model ({self.best_model}) to {best_model_path}")

        # Save all models
        temporal_dir = model_dir / "temporal_7day"
        temporal_dir.mkdir(exist_ok=True)

        for name, model in self.models.items():
            model_path = temporal_dir / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Save results summary
        results_summary = {
            name: {k: v for k, v in res.items()
                  if k not in ['y_train', 'train_pred', 'y_test', 'test_pred']}
            for name, res in self.results.items()
        }

        with open(temporal_dir / "results.json", 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        print(f"Saved all models to {temporal_dir}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("7-Day Temporal-ViT Model Training for Green Crab Molt Prediction")
    print("=" * 80)

    # Extract temporal features with 7-day window
    extractor = TemporalFeatureExtractor()
    df = extractor.load_dataset()

    sequences_df = extractor.create_temporal_sequences(window_size=7)

    if len(sequences_df) < 20:
        print(f"ERROR: Not enough sequences created ({len(sequences_df)}). Need at least 20.")
        return

    # Train models
    trainer = AdvancedTemporalModels(sequences_df)
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    trainer.train_models(X_train, X_test, y_train, y_test)

    # Generate plots and save models
    trainer.plot_comprehensive_evaluation()
    trainer.save_best_model()

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - 7-Day Temporal-ViT Models")
    print("=" * 80)

    for name, res in trainer.results.items():
        print(f"\n{name}:")
        print(f"  Train MAE: {res['train_mae']:.3f} days")
        print(f"  CV MAE: {res['cv_mae']:.3f} days")
        print(f"  Test MAE: {res['test_mae']:.3f} days")
        print(f"  Test RMSE: {res['test_rmse']:.3f} days")
        print(f"  Test R²: {res['test_r2']:.4f}")

    print("\n🎯 Key Achievements:")
    print(f"- Best model: {trainer.best_model}")
    print(f"- Best Test MAE: {min([res['test_mae'] for res in trainer.results.values()]):.3f} days")
    print(f"- Window size: 7 observations")
    print(f"- Total sequences: {len(sequences_df)}")
    print(f"- Crabs used: {sequences_df['crab_id'].nunique()}")
    print("- Leveraged ViT features with extended temporal patterns")
    print("- Model saved for production use")

    print("\n✅ 7-Day Temporal-ViT model training complete!")


if __name__ == "__main__":
    main()
