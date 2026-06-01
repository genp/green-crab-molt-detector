# Green Crab Molt Detection System 🦀

AI-powered system for predicting green crab (*Carcinus maenas*) molt phases to support sustainable harvesting in New Hampshire and Maine.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-ready-brightgreen.svg)

## 🎯 Overview

This project uses computer vision and machine learning to help fishermen identify the optimal harvest time for green crabs - just before they molt (the "peeler" stage), when they're most valuable for soft-shell crab markets.

### Key Features

- 🔬 **Neural Network Analysis**: Uses YOLO (marine-trained) and CNN models for feature extraction
- 📊 **t-SNE Visualization**: Visual clustering of crabs by molt phase
- 🎯 **Molt Prediction**: Regression models predict days until molting
- 🌐 **Web Interface**: Easy-to-use drag-and-drop interface for field use
- 📱 **Mobile Friendly**: Responsive design works on phones and tablets
- 🐳 **Docker Support**: Easy deployment with containerization

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/[your-username]/green-crab-molt-detector.git
cd green-crab-molt-detector

# Create a clean Python 3.12 environment outside the repo tree
python3.12 -m venv ~/.venv/green_crabs_local
source ~/.venv/green_crabs_local/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (feature extraction + model training)
python run_pipeline.py

# Start the FastAPI web application
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8080
```

Then open http://127.0.0.1:8080/ui in your browser.

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

## 📸 Screenshots

### Web Interface
The system provides an intuitive interface for molt phase prediction:

![App Interface](static/app%20screenshot.png)
*Clean, mobile-friendly interface with drag-and-drop image upload*

### Prediction Results

#### Peeler Crab (Optimal Harvest Time)
![Peeler Result](static/peeler_result.png)
*Example result showing a crab ready for harvest (0-3 days until molt)*

#### Inter-molt Crab (Not Ready)  
![Inter-molt Result](static/inter-molt%20result.png)
*Example result showing a crab not ready for harvest (>10 days until molt)*

### Key Interface Features
- Upload crab images via drag-and-drop or file selection
- Get instant molt phase predictions with confidence scores
- Receive clear harvest recommendations with color-coded alerts
- Mobile-responsive design works on phones and tablets

## 🧬 The Science

### Molt Cycle Phases
1. **Inter-molt**: Crab is not close to molting (>10 days)
2. **Pre-molt**: Approaching molt (4-10 days)
3. **Peeler**: Optimal harvest window (0-3 days) ⭐
4. **Post-molt**: Recently molted, soft shell

### Visual Indicators
- Color progression: Green → Yellow → Orange → Red
- Shell texture changes
- Behavioral patterns

## 📊 Model Performance

The system achieves molt phase prediction with:
- **Mean Absolute Error**: ~3-5 days (varies by model)
- **Peeler Detection**: High accuracy for 0-3 day window
- **Processing Time**: <2 seconds per image

### Supported Models
- Random Forest (best overall performance)
- Gradient Boosting
- Support Vector Regression
- Neural Networks

## 🌐 Deployment

### Local Development
```bash
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8080
```

### Docker
```bash
docker-compose up -d
```

### Cloud Platforms
See `DEPLOYMENT.md` for detailed instructions:
- Heroku
- AWS EC2
- Google Cloud Run
- DigitalOcean

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Detailed setup and usage guide
- **[CLAUDE.md](CLAUDE.md)**: Project context and development notes
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Production deployment guide

## 🤝 Contributing

Contributions are welcome! This project supports:
- Marine biology research
- Sustainable fisheries development
- Invasive species management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Gabriela Bradt, Ph.D. NH Sea Grant UNH Cooperative Extension
- [FathomNet](https://fathomnet.org/) for marine species detection models
- Green crab fishermen providing domain expertise

---

*Supporting sustainable green crab harvesting through AI* 🌊🦀🤖
