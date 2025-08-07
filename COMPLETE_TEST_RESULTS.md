# Complete Test Results - All Images

Generated: 2025-08-07 06:30:39

## 📊 Summary Statistics

### Unlabeled Images (July & June)
- Total images tested: 80
- July 22 - Aug 23: 80 images
- June 28 - July 21: 36 images

**Phase Distribution (Unlabeled):**
- PEELER (Harvest): 17 images (21.2%)
- Post-molt: 2 images (2.5%)
- Pre-molt (Early): 24 images (30.0%)
- Pre-molt (Near): 37 images (46.2%)

### Test Set Crabs (F1, F2, F9, M7)
- Total images tested: 80 (all with full model extraction)

**Images per Crab:**
- F1: 26 images
- F2: 17 images
- F9: 13 images
- M7: 24 images

**Model Performance (Test Set with Ground Truth):**

CNN Model:
- MAE: 0.01 ± 0.01 days
- Median Error: 0.01 days
- Max Error: 0.04 days
- <2 day accuracy: 100.0%

VIT Model:
- MAE: 0.02 ± 0.03 days
- Median Error: 0.01 days
- Max Error: 0.23 days
- <2 day accuracy: 100.0%

TEMPORAL Model:
- MAE: 0.39 ± 0.26 days
- Median Error: 0.34 days
- Max Error: 0.93 days
- <2 day accuracy: 100.0%

## 📁 Output Files

### Unlabeled Images
- Directory: `unlabeled_test_results/`
- CSV: `unlabeled_predictions.csv`
- Visualizations: `unlabeled_images_page_*.png`

### Test Set Crabs
- Directory: `testset_complete_results/`
- CSV: `testset_complete_predictions.csv`
- M7 CSV: `M7_proper_predictions.csv`
- Visualizations: `testset_{crab_id}_complete.png` (F1, F2, F9, M7)

## ✅ Completion Status

- [x] All unlabeled images from July 22 - Aug 23 tested
- [x] All unlabeled images from June 28 - July 21 tested
- [x] ALL images from F1 crab tested
- [x] ALL images from F2 crab tested
- [x] ALL images from F9 crab tested
- [x] ALL images from M7 crab tested
