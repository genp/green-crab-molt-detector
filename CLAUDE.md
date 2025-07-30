# Green Crab Molt Detection Project Context

## Project Overview
This project aims to develop a neural network-based regression model to estimate the molting phase of green crabs (*Carcinus maenas*) from top and underside images. The goal is to support the development of a sustainable green crab fisheries industry in New Hampshire and Maine by identifying crabs at the optimal harvest time (just before molting, known as "peeler crabs").

## Business Context
- **Location**: New Hampshire and Maine coastline
- **Purpose**: Enable commercial harvesting of green crabs as a culinary product
- **Critical timing**: Crabs must be harvested just before molting, as they can only be used in cooking just after they've molted (soft-shell stage)
- **Economic impact**: Creating a new fisheries industry from an invasive species

## Green Crab Biology and Molting Cycle

### Molting Overview
- Green crabs molt approximately 18 times in their lifetime (4 larval, 14 post-larval)
- Molting cycle duration: Several weeks
- Actual molting process: 2-3 hours
- Carapace hardening: 3-4 days at 16°C, up to 16 days at 10-11°C

### Molt Cycle Phases
1. **Intermolt/Anecdysis**: Active feeding period, crab gradually changes color from green → yellow → orange → red
2. **Premolt (Peeler stage)**: Crab preparing to molt, highly valuable for harvest
3. **Ecdysis**: Active molting (2-3 hours)
4. **Postmolt**: Soft shell, green color, vulnerable state

### Visual Indicators
- **Color progression**: Green (recently molted) → Yellow → Orange → Red (extended intermolt)
- **Ventral coloration**: Important indicator of molt stage
- **Texture changes**: Shell hardness and appearance change throughout cycle

### Sex Differences
- **Females**: Single population-wide molt (June-November)
- **Males**: Two population-wide molts (April-June, November-June)

## Available Data

### Image Dataset
- **Total images**: ~505 crab images
- **Organization**: By individual crab ID and date
- **Naming convention**: Folders indicate molt dates (e.g., "F1 (molted 9:23)")
- **Image types**: Top and underside views of crabs
- **Time series**: Multiple images per crab over several weeks

### Data Structure
```
NH Green Crab Project 2016/
├── Crabs Aug 26 - Oct 4/
│   ├── F1 (molted 9:23)/
│   │   ├── 8:26/
│   │   ├── 9:1/
│   │   └── ... (dates leading to molt)
│   ├── F2 (molted 9:20)/
│   └── ... (more crabs)
├── Crabs July 22 - Aug 23/
├── Crabs June 28- July 21/
└── Green Crabs September 2016.xlsx (metadata)
```

### Key Observations
- Images are organized chronologically for each crab
- Molt dates are recorded in folder names
- Both male (M) and female (F) crabs included
- Some folders marked "MOLTED" contain post-molt images

## Technical Approach

### Models to Use
1. **Pre-trained YOLO model**: Located at `/Users/genp/BarderryAppliedResearch/FathomNet/qscp/jupyter_notebooks/fathomverse_detector/fathomverse-only-imgs_update_to_FathomNet-NoGameLabels-2024-09-28-model_yolo8_epochs_10_2024-10-22.pt`
   - General crab detector (not green crab specific)
   - Will be used for feature extraction and transfer learning

2. **Off-the-shelf model**: General model that includes crustaceans

### Development Principles
- **DRY (Don't Repeat Yourself)**: Modular, reusable code
- **Type hinting**: Full type annotations for all functions
- **Verbose commenting**: Detailed documentation of logic and decisions
- **Git workflow**: Frequent commits with clear messages

### Technical Tasks
1. Extract penultimate layer features from YOLO model
2. Create t-SNE visualization colored by molt status
3. Build regression model for molt phase prediction
4. Develop web application for deployment
5. Deploy to web hosting service

### Expected Outputs
1. **t-SNE visualization**: 2D plot with points colored by molt status
2. **Molt phase predictor**: Regression model outputting continuous molt phase
3. **Web application**: User-friendly interface for crab image upload and molt phase prediction

## Important Considerations
- Ensure model can handle both top and underside views
- Consider time series nature of data (same crab photographed over time)
- Account for lighting and image quality variations
- Validate model on crabs not seen during training
- Consider environmental factors (temperature effects on molt timing)

## Success Metrics
- Accurate molt phase prediction (days until molt)
- Clear separation of molt phases in t-SNE visualization
- User-friendly web interface
- Deployment accessible to fishermen in the field

## Running the System

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py

# Start web app
python app.py
```

### Key Scripts
- `run_pipeline.py`: Runs entire pipeline automatically
- `run_feature_analysis.py`: Feature extraction and t-SNE visualization
- `train_model.py`: Train and evaluate regression models
- `app.py`: Web application for molt detection
- `deploy.py`: Create deployment files

### Expected Runtime
- Feature extraction: 5-10 minutes (depending on GPU availability)
- Model training: 2-5 minutes
- Web app startup: <30 seconds

### Output Files
- `data/processed/`: Extracted features and dataset CSV
- `models/`: Trained regression models (.joblib files)
- `plots/`: t-SNE visualizations and model comparisons
- `DEPLOYMENT.md`: Cloud deployment instructions