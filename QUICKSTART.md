# Green Crab Molt Detection - Quick Start Guide

## Overview
This system uses computer vision and machine learning to predict when green crabs will molt, helping fishermen harvest them at the optimal time (just before molting).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python run_pipeline.py
```

This will:
- Load and analyze crab images
- Extract features using neural networks
- Create t-SNE visualizations
- Train regression models
- Prepare the web application

### 3. Start Web Application
```bash
python app.py
```

Then open http://localhost:5000 in your browser.

## Manual Steps

If you prefer to run steps individually:

### Step 1: Feature Extraction
```bash
python run_feature_analysis.py
```
- Extracts features from crab images
- Creates t-SNE visualizations in `plots/`

### Step 2: Model Training
```bash
python train_model.py
```
- Trains regression models
- Saves best model to `models/`

### Step 3: Web App
```bash
python app.py
```
- Starts the web interface
- Upload crab photos to get molt predictions

## Project Structure
```
green_crabs/
├── src/                    # Core modules
│   ├── data_loader.py     # Load crab images and metadata
│   ├── feature_extractor.py # Extract features using YOLO/CNN
│   ├── model.py           # Regression models
│   └── visualization.py   # t-SNE and plots
├── NH Green Crab Project 2016/  # Image data
├── models/                # Trained models
├── plots/                 # Visualizations
├── templates/             # Web interface
├── app.py                 # Flask web app
├── train_model.py         # Model training script
└── run_pipeline.py        # Run complete pipeline
```

## Key Features

### Molt Phase Prediction
- **Peeler (0-3 days)**: Ready to harvest! 🦀
- **Pre-molt (4-10 days)**: Monitor closely
- **Inter-molt (>10 days)**: Check later

### Visualizations
Check the `plots/` directory for:
- t-SNE clustering by molt phase
- Model performance comparisons
- Temporal progression charts

### Web Interface
- Drag-and-drop image upload
- Real-time molt phase prediction
- Harvest recommendations
- Mobile-friendly design

## Model Information

The system uses:
1. **Feature Extraction**: YOLO (marine-trained) or ResNet50
2. **Regression**: Random Forest, Gradient Boosting, or Neural Network
3. **Output**: Days until molt (continuous value)

## Troubleshooting

### No YOLO model found
The system will fall back to CNN features (ResNet50).

### Low accuracy
- Ensure good image quality
- Need more training data for better results
- Check that molt dates in folders are accurate

### Web app not starting
- Check models are trained: `ls models/`
- Verify port 5000 is available
- Check error logs for missing dependencies

## For Developers

### Adding New Features
1. Modify feature extraction in `src/feature_extractor.py`
2. Add new model types in `src/model.py`
3. Update web interface in `templates/index.html`

### Retraining Models
```bash
# After adding new data
python run_feature_analysis.py  # Re-extract features
python train_model.py           # Retrain models
```

## Deployment
See `DEPLOYMENT.md` for cloud deployment instructions.

## Contact
For issues or questions, please check the logs in the console output or create an issue in the repository.