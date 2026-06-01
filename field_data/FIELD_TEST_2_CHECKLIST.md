# Field Test #2 Checklist

Goal: test whether the app can support fast field scanning, identify the best physical workflow, and collect labeled images for improving detection and molt estimation.

## Before The Test

- [ ] Confirm the FastAPI app runs on the test device or server.
- [ ] Confirm camera access works on the phone.
- [ ] Confirm `/predict_stream` returns detection boxes and molt estimates.
- [ ] Confirm session export works, or prepare a manual note sheet if export is not ready.
- [ ] Print or open `SPREADSHEET_README.md`.
- [ ] Create the shared spreadsheet from `crab_monitoring_template.csv`.
- [ ] Create iCloud albums by date and condo, for example `2026-06-03 Condo C01`.
- [ ] Prepare stable crab IDs for each condo: `C01-01` through `C01-36`.
- [ ] Bring a phone charger or battery pack.
- [ ] Bring a tripod or phone stand.
- [ ] Bring a tray or marked scan area.
- [ ] Bring a ruler, caliper, or known-size reference marker if size estimation will be tested.
- [ ] Decide who scans, who handles crabs, and who records notes.

## Test Workflows

Run each workflow separately and record timing, failures, and user comments.

### Workflow A: Handheld Crab, Handheld Phone

- [ ] One person holds one crab.
- [ ] One person scans with phone.
- [ ] Capture ventral, dorsal, and side views.
- [ ] Record whether the crab was too wiggly, too close, too far, or hard to focus.
- [ ] Record approximate crabs processed per minute.

### Workflow B: Fixed Phone Stand

- [ ] Mount the phone on a stand.
- [ ] Move each crab into the marked scan zone.
- [ ] Capture ventral, dorsal, and side views.
- [ ] Record whether detection is more stable than handheld scanning.
- [ ] Record approximate crabs processed per minute.

### Workflow C: Tray Or Table Scan

- [ ] Place one crab in a tray or on a consistent background.
- [ ] Keep the camera distance consistent.
- [ ] Capture ventral, dorsal, and side views.
- [ ] Test sun, shade, and mixed light if available.
- [ ] Record whether this produces the most repeatable images.

## Required Test Cases

- [ ] Close distance, under 6 inches.
- [ ] Medium distance, 6-12 inches.
- [ ] Far distance, over 12 inches.
- [ ] Direct sun.
- [ ] Shade.
- [ ] Mixed shadow.
- [ ] Ventral view.
- [ ] Dorsal view.
- [ ] Side-left or side-right view.
- [ ] Moving crab.
- [ ] Stationary crab.
- [ ] Crab partly out of frame.
- [ ] Multiple crabs in frame.
- [ ] Single isolated crab in frame.

## Per-Crab Minimum Photo Set

For monitored condo crabs, collect at least:

- [ ] Ventral close image.
- [ ] Dorsal close image.
- [ ] Side close image.
- [ ] One medium-distance image.

If time allows:

- [ ] Repeat one view in sun.
- [ ] Repeat one view in shade.
- [ ] Include a known-size reference marker.

## What To Record In The Spreadsheet

For each image:

- [ ] `date`
- [ ] `condo_id`
- [ ] `image_id`
- [ ] `icloud_album_name`
- [ ] `image_filename`
- [ ] `photographer`
- [ ] `view_angle`
- [ ] `distance_category`
- [ ] `lighting`
- [ ] `background`
- [ ] `crab_in_frame_count`
- [ ] `crab_position`
- [ ] `motion_blur`
- [ ] `sex`
- [ ] `known_molt_phase`
- [ ] `review_status`

If the app is used on the image:

- [ ] `app_estimated_days_to_molt`
- [ ] `app_phase`
- [ ] `app_confidence`

If the crab later molts:

- [ ] `molt_event_date`
- [ ] `days_until_molt_if_known`

## App Behavior To Evaluate

- [ ] Detection box appears on the crab.
- [ ] Detection box follows the crab smoothly.
- [ ] Estimate does not flicker badly between categories.
- [ ] App gives useful guidance when the frame is poor.
- [ ] App warns when multiple crabs are in frame.
- [ ] App warns when the crab is too far away.
- [ ] App warns when lighting is too dark or shadowed.
- [ ] Results are readable outdoors.
- [ ] Scanner is usable while holding a crab.
- [ ] Scanner is usable with a fixed phone stand.
- [ ] Session history/export is useful for later review.

## Metrics To Capture

Record these for each workflow:

- [ ] Number of crabs attempted.
- [ ] Number of successful scans.
- [ ] Number of no-detection failures.
- [ ] Number of poor-quality warnings.
- [ ] Number of multiple-crab warnings.
- [ ] Average seconds per crab.
- [ ] Best physical setup.
- [ ] Worst physical setup.
- [ ] Main user frustration.
- [ ] Main requested improvement.

## End Of Test

- [ ] Confirm all photos are uploaded to the shared iCloud album.
- [ ] Confirm spreadsheet rows exist for each useful image.
- [ ] Mark unusable images as `exclude`, not deleted.
- [ ] Save app session export, if available.
- [ ] Write a short summary of what workflow felt fastest.
- [ ] Write a short summary of what workflow produced the most trustworthy images.
- [ ] List the top three UI changes needed before Field Test #3.
