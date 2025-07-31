#!/bin/bash

# Script to push green crab molt detection project to GitHub

echo "Green Crab Molt Detection - GitHub Push Helper"
echo "=============================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Error: Git repository not initialized"
    echo "Run: git init"
    exit 1
fi

# Get GitHub username
read -p "Enter your GitHub username: " github_username

# Get repository name
read -p "Enter repository name (default: green-crab-molt-detector): " repo_name
repo_name=${repo_name:-green-crab-molt-detector}

# Ask if repo already exists
echo ""
echo "Does the repository already exist on GitHub?"
echo "1) No, create a new repository"
echo "2) Yes, use existing repository"
read -p "Select option (1 or 2): " repo_exists

if [ "$repo_exists" == "1" ]; then
    echo ""
    echo "To create a new repository on GitHub:"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: $repo_name"
    echo "3. Description: 'AI-powered green crab molt phase detection for sustainable harvesting'"
    echo "4. Choose 'Public' or 'Private'"
    echo "5. DO NOT initialize with README (we already have one)"
    echo "6. Click 'Create repository'"
    echo ""
    read -p "Press Enter after creating the repository..."
fi

# Add remote
echo ""
echo "Adding remote repository..."
git remote add origin "https://github.com/$github_username/$repo_name.git"

# Verify remote was added
echo "Remote added:"
git remote -v

# Create README if it doesn't exist
if [ ! -f "README.md" ]; then
    echo ""
    echo "Creating README.md..."
    cat > README.md << 'EOF'
# Green Crab Molt Detection System 🦀

AI-powered system for predicting green crab (*Carcinus maenas*) molt phases to support sustainable harvesting in New Hampshire and Maine.

## Overview

This project uses computer vision and machine learning to help fishermen identify the optimal harvest time for green crabs - just before they molt (the "peeler" stage), when they're most valuable for soft-shell crab markets.

## Features

- 🔬 **Neural Network Analysis**: Uses YOLO and CNN models for feature extraction
- 📊 **t-SNE Visualization**: Visual clustering of crabs by molt phase
- 🎯 **Molt Prediction**: Regression model predicts days until molting
- 🌐 **Web Interface**: Easy-to-use drag-and-drop interface
- 📱 **Mobile Friendly**: Responsive design for field use

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py

# Start web application
python app.py
```

Then open http://localhost:5000 in your browser.

## Project Structure

```
green_crabs/
├── src/                    # Core ML modules
├── models/                 # Trained models
├── plots/                  # Visualizations
├── templates/              # Web interface
├── app.py                  # Flask application
└── QUICKSTART.md          # Detailed setup guide
```

## Model Performance

The system achieves molt phase prediction with:
- Mean Absolute Error: ~X days
- Identifies "peeler" crabs (0-3 days before molt)
- Processes images in real-time

## Deployment

See `DEPLOYMENT.md` for cloud deployment options (Heroku, AWS, Docker).

## Contributing

This project supports marine biology research and sustainable fisheries. Contributions welcome!

## License

[Your chosen license]

## Acknowledgments

- Marine biologists working on NH/ME coastline
- FathomNet for marine species detection models
EOF
    git add README.md
    git commit -m "Add comprehensive README for GitHub"
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "You may be prompted for your GitHub credentials."
echo ""

# Push main branch
git push -u origin main

# Check if push was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "Your repository is now available at:"
    echo "https://github.com/$github_username/$repo_name"
    echo ""
    echo "Next steps:"
    echo "1. Add a license file if needed"
    echo "2. Set up GitHub Actions for CI/CD (optional)"
    echo "3. Add collaborators if working with a team"
    echo "4. Consider adding GitHub Pages for documentation"
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "1. Authentication failed - set up GitHub credentials"
    echo "2. Repository doesn't exist - create it on GitHub first"
    echo "3. Branch name mismatch - your default might be 'master' not 'main'"
    echo ""
    echo "To fix branch name issue:"
    echo "git branch -m master main  # rename master to main"
    echo "git push -u origin main"
fi