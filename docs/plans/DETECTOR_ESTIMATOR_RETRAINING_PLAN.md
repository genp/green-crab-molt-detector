# Detector And Estimator Retraining Plan

This plan addresses five current issues:

1. The detector misses small but obvious crabs.
2. Adjacent-frame estimates for nearly identical crops jump around.
3. Expanded SAM3 bootstrap labels should improve the detector.
4. New `data/raw` labels and layouts need review before estimator retraining.
5. Streaming video can keep displaying a stale bbox after the crab is removed.
6. Unlabeled images may contain visible molt cues that can bootstrap estimator
   attribute labels after human review.

The work is staged so detector-only data does not pollute estimator evaluation,
and so all model types use one global split registry.

## Phase 1: Inventory And Global Splits

Build:

```text
data/processed/global_split_registry.csv
data/processed/detector_v2_manifest.csv
data/processed/estimator_v2_manifest.csv
data/processed/field_qa_manifest.csv
```

Follow `docs/DATASET_CONSTRUCTION_PROTOCOL.md`.

Key rule: train/val/test grouping is global over model type. The same
`source_group_id` cannot be detector train and estimator test.

Inventory sources:

```text
data/raw/Green Crab AI 2026/
data/raw/NH Green Crab Project 2016/
data/raw/NH Green Crab Project -Doyle Fellowship 2017/
data/raw/2018 NH Green Crab-Doyle Fellowship/
data/raw/Crate 1 - Completed/
data/raw/Crate 2 - Completed/
data/raw/Crate 3 - Completed/
data/raw/moltmeter_debug_*/
data/bootstrap_bboxes/
```

Parse available `.docx`, `.xlsx`, and `.csv` labels. Keep low-confidence or
ambiguous labels with `label_confidence`, but do not silently mix them into the
estimator training set.

## Phase 2: Detector V2

Goal: improve small-crab recall and reduce false positives in field conditions.

Training data:

- Reviewed bbox labels.
- Accepted SAM3 bootstrap boxes.
- Small obvious crabs.
- Red-colored green crabs.
- Side-view crabs.
- In situ crabs.
- Human/glove/equipment false positives as hard negatives.

Detector settings to test:

- Use YOLO11 as the next detector baseline once the reviewed detector dataset
  is ready. Train `yolo11n` and `yolo11s` before considering larger models.
- Train at `imgsz=960` and `imgsz=1280`.
- Compare stream inference at `STREAM_YOLO_IMGSZ=416`, `640`, and `960`.
- Lower small-object filter candidates from `YOLO_MIN_AREA_PCT=0.01` to
  `0.002` or `0.003`, then tune against false positives.
- Keep `YOLO_MAX_DETECTIONS` high enough for multi-crab frames.

Evaluation:

- Recall by object size: small, medium, large.
- Recall by view: dorsal, ventral, side.
- Recall by color: normal, red.
- Recall for in situ images.
- False positives per image.
- Whole-image fallback rate in the app.
- Latency at each stream image size.

Initial deployment target: reduce the documented whole-image fallback rate on
`data/raw/Green Crab AI 2026/` without unacceptable false positives.

### YOLO11 Migration Gate

Switching the app detector to YOLO11 is planned, but gated on dataset and
benchmark readiness.

Dataset readiness:

- Reviewed SAM3/bootstrap boxes are available for `Green Crab AI Photo
  Database`, `moltmeter_debug_*`, in situ images, red crabs, side views, and
  small obvious crabs.
- Human/glove/equipment false positives are included as hard negatives.
- `data/processed/global_split_registry.csv` validates that detector and
  estimator train/val/test groups are consistent.
- Detector export is built only from accepted/reviewed detector labels, not
  molt-cue proposal rows.

Model comparison:

- Train `yolo11n` and `yolo11s` with the same registry-aware detector dataset.
- Compare against the current YOLOv8/bootstrap detector on the same holdout.
- Report recall by size, view, color state, and in situ status.
- Report false positives per image on field negatives.
- Benchmark Cloud Run CPU latency using ONNX or OpenVINO export at stream image
  sizes `640` and `960`.

Promotion criteria:

- Better small-crab recall than the current detector.
- No unacceptable increase in human/glove/equipment false positives.
- Lower whole-frame fallback rate in the streaming app.
- Meets the target Cloud Run CPU latency budget for the expected traffic.

## Phase 3: SAM3 Bootstrap Expansion

Use the existing SAM3 proposal pipeline and expand prompts for hard cases:

```text
small green crab
red green crab
side view crab
crab in mud
crab underwater
crab on rock
crab partly hidden
```

Review tags to preserve:

```text
species
view
color_state
is_in_situ
negative_type
box_quality
```

Accepted rows feed detector training. Rejected false positives should be
retained as hard-negative evidence and can become detector negative images.

## Phase 4: Estimator V2

Goal: improve molt timing estimates, especially for red crabs, side views, in
situ images with valid labels, detector-crop deployment conditions, and
cross-view consistency for the same crab.

Use only rows with reliable timing labels:

- `capture_date`
- `molt_date` or expert molt estimate
- `days_to_molt`
- `source_group_id`
- `crab_id`
- `observation_id` for the same crab on the same date/session
- `view` for dorsal, ventral, side, or unknown
- registry split

Dataset construction requirement:

- Group dorsal, ventral, and side images of the same crab from the same
  capture event under one `observation_id`.
- Assign the same `days_to_molt` target to all views in an observation.
- Keep all views from an observation in the same split through
  `source_group_id`.
- Prefer balanced multi-view batches or sample weights so the estimator sees
  dorsal, ventral, and side examples for the same target instead of learning
  view-specific target drift.
- Track whether a row is part of a complete multi-view set with
  `has_dorsal_view`, `has_ventral_view`, and `has_side_view`.

Train/evaluate candidates:

- Current ViT features plus RandomForest/GradientBoosting.
- Image embedding plus metadata: `view`, `sex`, `color_state`,
  `bbox_area_pct`, and `crop_source`.
- Temporal model where repeated observations exist.
- Optional side-view specialist if side examples behave differently.
- Multi-view consistency regularization or post-hoc calibration that penalizes
  different predictions for dorsal, ventral, and side images of the same
  `observation_id`.

Evaluate by subgroup:

- Red green crabs.
- Side view.
- In situ.
- Dorsal.
- Ventral.
- Small bbox crops.
- Detector crop vs reviewed crop.
- Same-crab, same-observation view disagreement:
  `max(pred_days) - min(pred_days)` across dorsal/ventral/side views.

Use crab/source-group splits only. Do not split by individual image.

Acceptance target: for complete same-observation multi-view sets, the estimator
should predict a consistent molt window across views. Report MAE by view and
view-disagreement separately; a lower overall MAE is not acceptable if it is
achieved by returning materially different molt windows for different views of
the same crab.

### SAM3 Molt-Cue Attribute Bootstrap

Use SAM3 as a review-first proposal tool for visible pre-molt cues. These
outputs are not molt-time labels and must not be merged into estimator training
until reviewed and joined through the global split registry.

Target output:

```text
data/bootstrap_molt_cues/
```

Initial prompts:

```text
split in crab shell
crack on side of crab shell
blue crab legs
dusky blue crab shell
dull ventral crab plates
pale dull underside of crab
dead-looking crab underside
```

Review labels:

```text
side_shell_split
dusky_blue_dorsal
dusky_blue_legs
dull_ventral_plates
view
species
image_quality
cue_quality
review_status
review_notes
```

Use reviewed cue attributes as auxiliary estimator metadata/features and for
subgroup evaluation. Keep these proposals separate from detector bbox bootstrap
outputs so cue regions do not become detector labels by accident.

## Phase 5: Stable Streaming Estimates

The app should not expose raw frame-by-frame regression directly.

Add track-level smoothing:

- Track primary crab by bbox IoU and center distance.
- Maintain a rolling buffer of recent estimates for the same track.
- Display median days-to-molt and a range, not only one raw value.
- Reject sudden jumps unless supported by consecutive frames.
- Run detection more often than the estimator.
- Reset smoothing when the track changes or bbox IoU drops for several frames.

Recommended display:

```text
estimated 2-4 days
last detected 180 ms ago
```

## Phase 6: Bbox Disappearance Handling

Fix stale stream boxes when a crab leaves the frame.

Rules:

- Bbox overlay disappears quickly when detection is lost.
- Estimate card may remain briefly, but should be marked as based on the last
  detected crab.
- Separate estimate smoothing from bbox display smoothing.
- Clear bbox after 2-3 consecutive no-detection frames or after 300-500 ms.
- Return explicit fields such as:

```text
bbox_stale
bbox_cleared_reason
last_detection_age_ms
track_id
estimate_stale
```

QA case: remove crab from frame and verify the bbox clears in under 500 ms.

## Execution Order

1. Write and validate the dataset construction protocol.
2. Build the global split registry.
3. Build detector, estimator, SAM3 review, and field QA manifests from the
   registry.
4. Run split validation as a hard gate.
5. Expand/review SAM3 labels.
6. Train detector v2.
7. Evaluate detector v1 vs v2.
8. Add stream bbox disappearance handling and estimate smoothing.
9. Build estimator v2 training set from reliable timing labels.
10. Train and evaluate estimator v2 by subgroup.
11. Deploy detector v2 first if detector metrics improve.
12. Deploy estimator v2 only after source-group holdout metrics improve.

## Execution Status - 2026-06-05

Completed:

- Added the dataset construction protocol and global split validation gate.
- Built current manifests:
  - `data/processed/global_split_registry.csv`
  - `data/processed/detector_v2_manifest.csv`
  - `data/processed/estimator_v2_manifest.csv`
- Validated detector/estimator split consistency with
  `tools/validate_dataset_splits.py`.
- Exported a registry-aware detector dataset:
  - `data/bootstrap_yolo/green_crab_detector_v2_registry/`
  - `data/bootstrap_yolo/green_crab_detector_v2_registry.yaml`
- Ran a one-epoch detector smoke train:
  - `runs/detect/runs/detect/green_crab_detector_v2_registry_smoke/weights/best.pt`
- Ran an estimator v2 smoke train from existing ViT features:
  - `models/estimator_v2/random_forest.joblib`
  - `models/estimator_v2/gradient_boosting.joblib`
  - `models/estimator_v2/results.json`
- Updated the streaming app to:
  - clear bboxes when the detector loses the crab
  - keep only the estimate as stale text
  - smooth stream estimates over recent same-track frames
  - expose estimate ranges in the UI
  - default to the bootstrapped detector when available
  - lower the small-object area filter
  - use 640 px stream detector inference by default

Current blockers before production retraining:

- Only 7 reviewed SAM3 boxes across 5 unique images are available in the current
  reviewed detector CSV. The detector smoke checkpoint is not production ready.
- The estimator smoke train matched 230 feature rows and did not produce test
  metrics because precomputed ViT features are missing for many
  `estimator_v2_manifest.csv` rows.
- Human/glove/equipment false positives are not yet present in sufficient
  reviewed detector manifests.
