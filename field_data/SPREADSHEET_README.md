# Crab Monitoring Spreadsheet Guide

Use `crab_monitoring_template.csv` as the shared tracking sheet for June crab condo monitoring. Keep one row per image, not one row per crab per day. Multiple images of the same crab are useful because camera distance, view angle, sun/shade, and motion all affect model performance.

## File Naming

- `condo_id`: `C01`, `C02`, `C03`
- `image_id`: recommended format:

```text
YYYY-MM-DD_CONDO_VIEW_DISTANCE_LIGHTING_SEQUENCE
```

Example:

```text
2026-06-01_C01_ventral_close_sun_001
```

Do not rename original iCloud images if that is inconvenient. If the iCloud filename stays as `IMG_1234.JPG`, put that exact filename in `image_filename` and put the structured ID in `image_id`.

## Required Columns

Fill these in for every image:

- `date`: date photo was taken, formatted `YYYY-MM-DD`
- `condo_id`: condo identifier, such as `C01`
- `image_id`: structured image identifier
- `icloud_album_name`: shared album name
- `image_filename`: original image filename
- `photographer`: person who took the photo
- `view_angle`: controlled value from the list below
- `distance_category`: controlled value from the list below
- `lighting`: controlled value from the list below
- `background`: controlled value from the list below
- `crab_in_frame_count`: controlled value from the list below
- `crab_position`: controlled value from the list below
- `motion_blur`: controlled value from the list below
- `sex`: `male`, `female`, or `unknown`
- `known_molt_phase`: best known biological phase at time of image
- `review_status`: `new`, `reviewed`, or `exclude`

## Optional But Valuable Columns

Fill these when known:

- `days_until_molt_if_known`: number of days from photo date until observed molt
- `molt_event_date`: actual molt date, formatted `YYYY-MM-DD`
- `app_estimated_days_to_molt`: selected app estimate, using the YOLO crab crop when available and the whole image as fallback
- `app_estimated_molt_event_date`: estimated molt date from the selected app estimate
- `app_phase`: app phase label
- `app_confidence`: app confidence label
- `app_estimate_input`: `yolo_crop`, `whole_image_fallback`, or `not_run`
- `whole_image_estimated_days_to_molt`: app estimate from the whole image
- `whole_image_estimated_molt_event_date`: estimated molt date from the whole-image estimate
- `whole_image_app_phase`: phase from the whole-image estimate
- `yolo_crop_estimated_days_to_molt`: app estimate from the YOLO crop, only when a filtered detection exists
- `yolo_crop_estimated_molt_event_date`: estimated molt date from the YOLO crop estimate
- `yolo_crop_app_phase`: phase from the YOLO crop estimate
- `human_confidence`: confidence in the manual label
- `shell_condition_notes`: visual molt clues, shell softness, color, plates, etc.
- `limb_loss_or_injury`: missing limbs, damaged shell, unusual condition
- `notes`: any other useful context

## Controlled Values

Use these exact values so the spreadsheet can be parsed later.

### `view_angle`

```text
ventral
dorsal
side
unknown
```

### `distance_category`

```text
close_under_6in
medium_6_12in
far_over_12in
unknown
```

### `lighting`

```text
sun
shade
mixed
indoor
low_light
unknown
```

### `background`

```text
hand
tray
condo
cooler
table
other
```

### `crab_in_frame_count`

```text
1
multiple
partial
```

### `crab_position`

```text
centered
edge
partially_cut_off
unknown
```

### `motion_blur`

```text
none
slight
severe
unknown
```

### `sex`

```text
male
female
unknown
```

### `known_molt_phase`

```text
intermolt
pre_molt
peeler_imminent
molted
dead
unknown
```

### `app_phase`

Use the same phase vocabulary as `known_molt_phase`. Current app outputs should be mapped into these shared values, and future estimator versions should also use these values directly.

```text
intermolt
pre_molt
peeler_imminent
molted
dead
unknown
```

### `app_confidence` and `human_confidence`

```text
high
medium
low
unknown
```

### `app_estimate_input`

```text
yolo_crop
whole_image_fallback
not_run
```

Use `whole_image_fallback` when the crab detector fails but the image still contains a usable crab. This is important for current field data because the YOLO detector is missing many clear hand-held crab images.

### `review_status`

```text
new
reviewed
exclude
```

Use `exclude` for images that should not be used for model training, such as the wrong crab, no crab, severely blurry photos, duplicate accidental shots, or an image where the image context cannot be trusted.

## Photo Set Per Crab

Ideal photo set for each crab on each monitoring day:

- ventral close
- dorsal close
- side close
- one medium-distance image

If time allows, include both sun and shade examples. If time is limited, prioritize clear close images over large numbers of poor images.

## Monitoring Schedule

For Monday/Wednesday/Friday monitoring in June:

1. Open the iCloud album for the correct date and condo.
2. Use a consistent `image_id` pattern for the current condo, view, distance, lighting, and sequence number.
3. Take the minimum photo set for that crab.
4. Add one spreadsheet row per image.
5. If the crab molts later, update prior rows with `molt_event_date` and `days_until_molt_if_known`.
6. Mark rows `reviewed` after labels have been checked.

## Data Quality Rules

- Do not mix multiple crabs in one image unless deliberately testing multi-crab detection.
- Avoid filling free-text variants in controlled columns. Use the exact values above.
- Leave unknown values blank or use `unknown`; do not guess.
- The actual molt date is more important than the app estimate for training.
- Side-view images are important. Do not skip them unless the crab cannot be safely handled.
