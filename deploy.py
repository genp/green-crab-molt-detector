"""
Deployment script for green crab molt detection web app.

This script helps deploy the application to various platforms.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def create_procfile():
    """Create Procfile for Heroku deployment."""
    content = "web: gunicorn app:app\n"
    with open("Procfile", "w") as f:
        f.write(content)
    print("Created Procfile")

def create_runtime_txt():
    """Create runtime.txt for Python version specification."""
    content = "python-3.10.12\n"
    with open("runtime.txt", "w") as f:
        f.write(content)
    print("Created runtime.txt")

def create_deployment_requirements():
    """Create a deployment-specific requirements file."""
    # Read existing requirements
    with open("requirements.txt", "r") as f:
        requirements = f.read()
    
    # Add deployment-specific packages if not present
    if "gunicorn" not in requirements:
        requirements += "\n# Deployment\ngunicorn==21.2.0\n"
    
    with open("requirements-deploy.txt", "w") as f:
        f.write(requirements)
    print("Created requirements-deploy.txt")

def create_docker_files():
    """Create Docker files for containerized deployment."""
    
    # Dockerfile
    dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p temp_uploads models data/processed

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    print("Created Dockerfile")
    
    # docker-compose.yml
    docker_compose_content = """version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    print("Created docker-compose.yml")
    
    # .dockerignore
    dockerignore_content = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.gitignore
.mypy_cache
.pytest_cache
.hypothesis
*.ipynb
.ipynb_checkpoints
temp_uploads/*
*.jpg
*.jpeg
*.png
*.gif
*.bmp
!static/*.jpg
!static/*.png
"""
    
    with open(".dockerignore", "w") as f:
        f.write(dockerignore_content)
    print("Created .dockerignore")

def create_deployment_guide():
    """Create deployment guide documentation."""
    guide_content = """# Green Crab Molt Detection - Deployment Guide

## Prerequisites

1. Trained models in the `models/` directory
2. Python 3.10 or higher
3. All dependencies installed from requirements.txt

## Local Deployment

### Option 1: Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The app will be available at http://localhost:5000

### Option 2: Docker

```bash
# Build the Docker image
docker build -t green-crab-detector .

# Run the container
docker run -p 5000:5000 -v $(pwd)/models:/app/models green-crab-detector
```

Or using docker-compose:

```bash
docker-compose up -d
```

## Cloud Deployment Options

### Heroku

1. Install Heroku CLI
2. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

3. Deploy:
   ```bash
   git push heroku main
   ```

### AWS EC2

1. Launch an EC2 instance (t2.medium or larger recommended)
2. Install Docker on the instance
3. Copy your code and models to the instance
4. Run using Docker or docker-compose

### Google Cloud Run

1. Build and push Docker image to Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/green-crab-detector
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy --image gcr.io/PROJECT-ID/green-crab-detector --platform managed
   ```

### DigitalOcean App Platform

1. Connect your GitHub repository
2. Configure the app with:
   - Python 3.10 runtime
   - Build command: `pip install -r requirements.txt`
   - Run command: `gunicorn app:app`

## Environment Variables

Set these environment variables for production:

- `FLASK_ENV=production`
- `MODEL_PATH=/path/to/models` (optional, defaults to ./models)

## Performance Considerations

1. **Model Loading**: Models are loaded once at startup. Ensure sufficient memory.
2. **Image Processing**: Consider implementing a queue for high traffic.
3. **Caching**: Add Redis for caching predictions if needed.

## Security Considerations

1. Set up HTTPS with SSL certificates
2. Implement rate limiting
3. Add authentication if needed
4. Validate file uploads thoroughly
5. Keep models in a secure location

## Monitoring

1. Set up logging to a centralized service
2. Monitor memory usage (models can be large)
3. Track prediction latency
4. Set up alerts for errors

## Updating Models

To update the regression model:

1. Train new model using `train_model.py`
2. Copy new model file to `models/` directory
3. Restart the application

## Troubleshooting

### Models not loading
- Check that model files exist in `models/` directory
- Ensure YOLO model path is correct in app.py
- Check Python dependencies are installed

### Out of memory errors
- Use a larger instance
- Reduce batch size in feature extraction
- Consider model quantization

### Slow predictions
- Ensure GPU is available for YOLO (if applicable)
- Use smaller image sizes
- Cache frequent predictions
"""
    
    with open("DEPLOYMENT.md", "w") as f:
        f.write(guide_content)
    print("Created DEPLOYMENT.md")

def main():
    """Run deployment preparation."""
    print("Preparing deployment files...")
    
    # Create all deployment files
    create_procfile()
    create_runtime_txt()
    create_deployment_requirements()
    create_docker_files()
    create_deployment_guide()
    
    print("\nDeployment files created successfully!")
    print("\nNext steps:")
    print("1. Ensure models are trained: python train_model.py")
    print("2. Test locally: python app.py")
    print("3. Choose deployment platform and follow DEPLOYMENT.md")
    
    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists() or not list(models_dir.glob("*.joblib")):
        print("\nWARNING: No trained models found in models/ directory")
        print("Run 'python train_model.py' before deploying")

if __name__ == "__main__":
    main()