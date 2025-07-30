# Green Crab Molt Detection System

A neural network-based system for detecting the molting phase of green crabs (*Carcinus maenas*) to support sustainable harvesting in New Hampshire and Maine.

## Overview

This project uses computer vision and deep learning to predict when green crabs are about to molt (peeler stage), which is the optimal time for harvesting them for culinary use as soft-shell crabs.

## Features

- Transfer learning using YOLO pre-trained on marine species
- t-SNE visualization of crab images by molt status
- Regression model for molt phase prediction
- Web application for easy field use
- Support for both top and underside crab images

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd green_crabs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
green_crabs/
├── src/                  # Source code
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── feature_extractor.py  # YOLO feature extraction
│   ├── model.py          # Molt phase regression model
│   ├── visualization.py  # t-SNE and other visualizations
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks for exploration
├── models/               # Trained models
├── data/                 # Processed data (raw data in NH Green Crab Project 2016/)
├── static/               # Web app static files
├── templates/            # Web app templates
├── app.py                # Flask web application
└── requirements.txt      # Python dependencies
```

## Usage

### Data Preparation

```python
python src/data_loader.py
```

### Feature Extraction and t-SNE Visualization

```python
python src/visualization.py
```

### Training the Model

```python
python src/train.py
```

### Running the Web Application

```python
python app.py
```

## Model Details

The system uses:
1. A YOLO model pre-trained on FathomNet marine imagery for feature extraction
2. A regression model to predict days until molting
3. Color and texture features specific to crab molt indicators

## Contributing

Please see CLAUDE.md for detailed project context and development guidelines.

## License

[To be determined]