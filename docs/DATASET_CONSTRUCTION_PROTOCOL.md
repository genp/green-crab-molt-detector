# Dataset Construction Protocol

This protocol defines how datasets are built for detector, estimator, and
supporting model retraining. The central rule is that train/val/test assignment
is global across model types. A source group used for detector training must not
appear in estimator validation or test, and the reverse is also prohibited.

## Canonical Split Registry

Create one canonical split file before exporting any model-specific dataset:

```text
data/processed/global_split_registry.csv
```

Required columns:

```text
source_group_id
split
split_reason
source_dataset
crab_id
session_id
collection_date
is_in_situ
color_state
view
species
has_detector_label
has_estimator_label
has_negative_label
negative_type
notes
```

Allowed `split` values:

```text
train
val
test
field_qa_holdout
```

`source_group_id` is the leakage-safe unit. Use the largest related group that
could leak visual or biological information:

- Known crab sequence: use `crab_id`.
- Video, burst, or field collection session: use `session_id`.
- Unknown identity within a folder collection: use the collection/folder group.
- Individual image: use only when no related images are known.

All downstream manifests must join to this registry and inherit its split.
Do not compute train/val/test independently inside detector, estimator, species,
view, or temporal training scripts.

## Global Split Rules

- A `source_group_id` appears in exactly one split.
- Any shared source group keeps the same split across detector, estimator,
  species classifier, view classifier, and temporal datasets.
- Detector train groups cannot appear in estimator val/test.
- Estimator train groups cannot appear in detector val/test.
- `field_qa_holdout` is never used for training.
- Temporal sequences are never split across train/val/test.
- Hard negatives and false-positive examples get registry rows, even if only the
  detector uses them.
- A source folder can contribute to multiple model products only through
  separate manifests that all reference the same registry split.

## Stratified Global Splitting

The global split should be stratified, not purely random. Important subgroups
must be present in both train and test when enough distinct source groups exist.

Required stratification tags:

- `is_in_situ`
- `color_state`, especially `red_green_crab`
- `view`, especially `side`
- `species`
- `negative_type`, especially human, glove, and equipment false positives
- `source_dataset`
- `crab_id` or `session_id`

Required coverage:

- In situ green crabs must appear in both `train` and `test`.
- Red-colored green crabs must appear in both `train` and `test`.
- Side-view crabs should appear in both `train` and `test` when enough examples
  exist.
- Human/glove/equipment false positives must appear in detector `train`, with a
  reserved subset in detector `test`.

The split is still assigned at `source_group_id`, never by individual image.

## Dataset Products

### Detector V2 Manifest

Target:

```text
data/processed/detector_v2_manifest.csv
```

Purpose: crab localization.

Includes:

- Reviewed bbox labels.
- Accepted SAM3 bootstrap boxes.
- Small obvious crabs.
- Red-colored green crabs.
- Side-view crabs.
- In situ images.
- Human/glove/equipment false positives as hard negatives.
- Empty-label negative images for detector training.

Does not require:

- Molt date.
- Days to molt.
- Crab sequence identity, unless available for split grouping.

Required columns:

```text
image_path
source_group_id
split
bbox_xmin
bbox_ymin
bbox_xmax
bbox_ymax
species
view
color_state
is_in_situ
is_negative
negative_type
label_source
label_confidence
notes
```

### Estimator V2 Manifest

Target:

```text
data/processed/estimator_v2_manifest.csv
```

Purpose: molt timing prediction from crab crops or images.

Includes only examples with trustworthy timing labels:

- Capture date.
- Molt date or expert molt estimate.
- Computed `days_to_molt`.
- Known or reviewable species.
- View, sex, color state, and crop source when available.

Excludes:

- Detector-only negatives.
- Human/glove/equipment false positives.
- Unlabeled in situ examples without molt timing.
- Uncertain species unless explicitly marked and intentionally included.

Required columns:

```text
image_path
source_group_id
split
crab_id
capture_date
molt_date
days_to_molt
species
view
sex
color_state
is_in_situ
crop_source
bbox_xmin
bbox_ymin
bbox_xmax
bbox_ymax
label_confidence
notes
```

### SAM3 Review Manifest

Target:

```text
data/processed/sam3_review_manifest.csv
```

Purpose: proposal review and bbox bootstrapping.

This is not a training dataset until exported through the detector protocol.

Required review columns:

```text
review_status
review_notes
species
view
color_state
is_in_situ
negative_type
box_quality
```

Accepted values for `review_status` include `accept`, `accepted`, and `keep`.
Rejected rows should be retained because they are useful for hard-negative
analysis and future review audits.

### Field QA Manifest

Target:

```text
data/processed/field_qa_manifest.csv
```

Purpose: deployment regression testing.

Includes:

- Small obvious crabs.
- Removed-crab stream cases.
- Red crabs.
- Side views.
- In situ examples.
- False positives.

This data is holdout-only and must never appear in training.

## Validation Gate

Run split validation before training or exporting model datasets:

```bash
python3 tools/validate_dataset_splits.py \
  --registry data/processed/global_split_registry.csv \
  --manifest data/processed/detector_v2_manifest.csv detector \
  --manifest data/processed/estimator_v2_manifest.csv estimator \
  --manifest data/processed/field_qa_manifest.csv field_qa
```

The validation must fail if:

- A `source_group_id` appears in more than one split.
- An image path appears in conflicting splits.
- A manifest split disagrees with the global registry.
- A detector train group appears in estimator val/test.
- An estimator train group appears in detector val/test.
- A `field_qa_holdout` group appears in any training manifest.
- Any `source_group_id` marked as temporal or sequence data is split across
  multiple splits.

Validation should also report subgroup counts for:

```text
red_green_crab
in_situ
side
human false positives
glove false positives
equipment false positives
```

Treat this as a hard pre-training gate.
