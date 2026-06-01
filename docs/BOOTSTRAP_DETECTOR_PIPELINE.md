# Bootstrapped Crab Detector Pipeline

This document describes the reproducible workflow for bootstrapping green crab bounding boxes with SAM3, reviewing candidate boxes, exporting accepted labels to YOLO format, and evaluating a retrained detector against the May 29 blue-cooler dogfood set.

## Goal

The current YOLO detector misses many clear field images, including single crabs held in a gloved hand at close or medium distance. The goal is to use a stronger segmentation model to propose candidate crab boxes, review those proposals quickly, and use accepted boxes to train a field-domain YOLO detector.

The target production pipeline is:

1. Multi-crab detection.
2. Molt phase estimation for each detected crab crop.
3. Additional tags such as view and sex.
4. Calculated outputs such as days to molt and estimated molt date.

## Data Sources

Primary dogfood/evaluation set:

```text
data/raw/Green Crab AI 2026/
```

This is the May 29, 2026 blue-cooler field set. It should be used as a practical field benchmark because the current detector fails on many clear hand-held crabs.

Additional bootstrap/training sources:

```text
data/raw/NH Green Crab Project 2016/
data/raw/NH Green Crab Project -Doyle Fellowship 2017/
data/raw/2018 NH Green Crab-Doyle Fellowship/
data/processed/crate_images/
```

These folders include many hand-held crab images and historical labels encoded in folder names. The detector labels only require crab boxes; molt/date labels are not required for detector training.

## Environment

SAM3 and OpenCLIP are available in the local focal environment:

```bash
/Users/gen/.venv/focal3.12/bin/python
```

Use that interpreter for proposal generation:

```bash
/Users/gen/.venv/focal3.12/bin/python tools/propose_crab_bboxes_sam3.py --help
```

Current local note: run proposal generation with `--device cpu` in this environment. The SAM3 transformer package detects MPS, but its video post-processing path currently mixes CPU and MPS tensors on this Mac. The proposal script catches that failure and falls back to raw SAM3 masks, which are adequate for bootstrapped labels because every candidate is reviewed before training.

## Step 1: Generate Candidate Boxes

Smoke test on a small subset:

```bash
/Users/gen/.venv/focal3.12/bin/python tools/propose_crab_bboxes_sam3.py \
  --input "data/raw/Green Crab AI 2026" \
  --output data/bootstrap_bboxes/blue_cooler_may29 \
  --max-images 10 \
  --device cpu
```

Full run:

```bash
/Users/gen/.venv/focal3.12/bin/python tools/propose_crab_bboxes_sam3.py \
  --input "data/raw/Green Crab AI 2026" \
  --output data/bootstrap_bboxes/blue_cooler_may29 \
  --max-images 0 \
  --device cpu
```

The script uses these default text prompts:

```text
crab
a crab
green crab
crab in hand
ventral crab underside
dorsal crab shell
side view of crab
```

Smoke-test result on the prior YOLO failure case:

```text
input: data/raw/Green Crab AI 2026/IMG_2530.jpeg
output: data/bootstrap_bboxes/smoke_img2530_cpu_noclip/
candidates: 1
candidate bbox: xmin=309, ymin=561, xmax=1131, ymax=1792
```

For each prompt, SAM3 proposes masks. Masks are converted to bounding boxes and filtered by:

- minimum area fraction
- maximum area fraction
- aspect ratio
- duplicate overlap via non-maximum suppression

OpenCLIP then ranks candidate crops using positive and negative prompts:

Positive:

```text
a photo of a green crab
a close-up photo of a crab in a hand
a crab underside with legs and claws
```

Negative:

```text
a photo of a glove without a crab
a photo of a human hand
a photo of a wooden table or wire mesh
```

Outputs:

```text
data/bootstrap_bboxes/blue_cooler_may29/proposals.csv
data/bootstrap_bboxes/blue_cooler_may29/review_overlays/
data/bootstrap_bboxes/blue_cooler_may29/crops/
data/bootstrap_bboxes/blue_cooler_may29/contact_sheets/
```

## Step 2: Create Review Spreadsheet

Create an XLSX review workbook with embedded overlay and crop thumbnails:

```bash
green_crabs_mps/bin/python tools/create_bbox_review_spreadsheet.py \
  --proposals data/bootstrap_bboxes/blue_cooler_may29/proposals.csv \
  --output data/bootstrap_bboxes/blue_cooler_may29/bbox_review.xlsx
```

Review columns:

```text
review_status
review_notes
```

Use these values:

```text
accept
reject
adjust
missed_crab
duplicate
```

Only `accept`, `accepted`, and `keep` are exported to YOLO by the exporter.

If a box is close enough for training, mark `accept`. If a crab is present but no good candidate was proposed, mark the best row for that image as `missed_crab` and note what happened.

## Step 3: Export Accepted Boxes To YOLO

After review, save the workbook as CSV, for example:

```text
data/bootstrap_bboxes/blue_cooler_may29/proposals_reviewed.csv
```

Export accepted rows:

```bash
green_crabs_mps/bin/python tools/export_reviewed_bboxes_to_yolo.py \
  --reviewed data/bootstrap_bboxes/blue_cooler_may29/proposals_reviewed.csv \
  --output data/bootstrap_yolo/green_crab_detector_v1 \
  --force-test-substr "data/raw/Green Crab AI 2026"
```

The output YOLO dataset:

```text
data/bootstrap_yolo/green_crab_detector_v1/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
data/bootstrap_yolo/green_crab_detector_v1.yaml
```

Class list:

```text
0: green_crab
```

## Split Method

The split is deterministic and reproducible.

Rules:

1. Any image whose path contains a `--force-test-substr` value is assigned to `test`.
2. All remaining images are assigned by SHA1 hash of `image_path`.
3. Default fractions are:
   - test: 10%
   - val: 10%
   - train: 80%

The SHA1 rule means another researcher with the same reviewed CSV and same command will get the same train/val/test split. It also prevents accidental ordering effects from filesystem traversal.

For conference reporting, describe this as:

> Accepted SAM3/OpenCLIP pseudo-labels were split deterministically by SHA1 hash of source image path, with the May 29, 2026 blue-cooler field collection optionally held out as a dogfood test set. This avoids image-order dependence and allows exact reproduction from the reviewed proposal CSV.

## Step 4: Train YOLO

Example:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=data/bootstrap_yolo/green_crab_detector_v1.yaml \
  imgsz=1280 \
  epochs=80 \
  batch=8 \
  project=runs/detect \
  name=green_crab_detector_v1
```

Start with one class. Do not train molt phase, sex, or view into the detector. Those should be separate downstream classifiers/regressors.

## Step 5: Dogfood Evaluation

Evaluate the current detector and retrained detector on:

```text
data/raw/Green Crab AI 2026/
```

Report:

- images evaluated
- raw detection count
- filtered detection count
- missed clear single crabs
- multiple-crab detections
- false positives
- whole-image fallback rate in the app

The key practical metric is fallback rate. Before retraining, the blue-cooler label pass had:

```text
whole_image_fallback: 232 / 319
yolo_crop: 87 / 319
```

The first detector improvement target is:

```text
whole_image_fallback < 50 / 319
```

## Production Model Responsibilities

Detector:

- finds one or more crabs
- returns boxes and detection confidence
- should work for ventral, dorsal, and side views
- should handle crab in hand, tray, condo, cooler, and table backgrounds

Molt estimator:

- runs on each detected crab crop
- returns days to molt, estimated molt date, phase, confidence
- can fall back to whole image if detector fails, but should mark that estimate lower-confidence

Taggers:

- view: `ventral`, `dorsal`, `side`, `unknown`
- sex: `male`, `female`, `unknown`
- sex should initially be trusted only on good ventral images

Calculated outputs:

- `days_until_molt`
- `estimated_molt_event_date`
- `phase`
- `estimate_input`: `yolo_crop`, `sam_crop`, `whole_image_fallback`
- quality flags: low light, too far, multiple crabs, detector fallback used

## Reproducibility Checklist

For a publication or shared methods appendix, preserve:

- exact input folders
- proposal script version
- SAM3 model name: `facebook/sam3`
- prompts used
- filter thresholds
- NMS threshold
- OpenCLIP model name: `ViT-H-14`, `laion2b_s32b_b79k`
- reviewed proposal CSV
- export command
- train/val/test split command
- YOLO model seed/config
- final model weights
- dogfood evaluation CSV and gallery
