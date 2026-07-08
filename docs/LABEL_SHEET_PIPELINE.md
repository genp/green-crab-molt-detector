# Centralized Green Crab Label Sheet Pipeline

This document is the authoritative plan and reference for the `label_sheet/`
package. It is written so that a different engineer or model can pick the work up
without re-deriving context.

## 1. Goal

Produce a single, human-reviewable Excel workbook,
`data/processed/green_crab_label_sheet.xlsx`, with **one worksheet per study
year** (2016, 2017, 2018, "crate" study, 2026). Every worksheet uses the **same
column layout** even when a given year is missing some data. Each row is **one
image** and contains:

1. An **embedded thumbnail** of the image.
2. A **relative path** to the full-resolution image (relative to repo root).
3. **Ground-truth labels** extracted from that year's source (folder names, Word
   `.docx` tables, auxiliary `.xlsx`/`.csv` files, or the 2026 condo data sheet).
4. **Metadata / aux-labels** from the same sources (sex, condo/cell, color,
   carapace width, outcome, free-text molt-cue notes, etc.).
5. **Model estimates** from two models — the most-recent molt estimator and a
   "bootstrapped" estimator — for `days_to_molt` plus aux labels (sex, view,
   position, visible molt indicators such as shell transparency/color and seam /
   side cracking). Model columns carry an Excel **cell comment** that records
   exactly which model, weights file, and feature extractor produced them.
6. **Expert review columns** (dropdown-validated) so a biologist can
   confirm / reject / update every estimated label.

Model estimates are generated for **every image in every year, including years
that were in a model's training set** (the request is explicit about this — it
lets the expert see train-vs-field behavior). A per-row `*_in_training_set`
flag marks likely training leakage so the expert can weight those rows
accordingly.

## 2. Why the design is split into three layers

Model inference over several thousand images is the expensive part and needs GPU
/ heavy feature extractors. Label extraction and spreadsheet assembly are cheap.
So the pipeline is deliberately three decoupled stages that each read/write flat
files and can be run independently:

```
  extract  ->  data/processed/label_sheet/records.parquet   (cheap, no ML)
  predict  ->  data/processed/label_sheet/predictions.csv   (expensive, cached)
  assemble ->  data/processed/green_crab_label_sheet.xlsx    (cheap)
```

- **Extract** never imports torch. Re-run it freely as label sources improve.
- **Predict** is incremental: it only runs models on `(image, model_id)` pairs
  not already in the cache, so adding a year or a model does not re-run
  everything. Safe to interrupt and resume.
- **Assemble** joins records + prediction cache + thumbnails into the workbook.
  Regenerating the sheet after an expert tweaks column widths costs seconds.

## 3. Package layout

```
label_sheet/
  config.py            Paths, YearConfig, ModelConfig, and the default registry.
  schema.py            The single canonical column schema + controlled vocab.
  records.py           ImageRecord dataclass (one normalized row) + helpers.
  thumbnails.py        Cached EXIF-correct thumbnail generation (PIL).
  extractors/
    base.py            Extractor protocol; shared path/date parsing helpers.
    manifest_years.py  2016 / 2017 / 2018 from processed CSVs + folder names.
    crate_docs.py      Crate 1-3 from data/processed/crate_docs.csv (docx).
    year_2026.py       Green Crab AI 2026 condo sheet + capture-order mapping.
  models/
    registry.py        ModelRunner protocol + registry; prediction dataclass.
    features.py        ViT (torchvision) and OpenCLIP feature extractors.
    estimator.py       days_to_molt regressor runner (primary + bootstrap).
    auxlabels.py       Heuristic sex / view / molt-indicator taggers.
    cache.py           Incremental prediction cache (CSV keyed by image+model).
  xlsx_writer.py       openpyxl multi-sheet writer: thumbnails, comments,
                       frozen header, expert dropdowns, consistent formatting.
  pipeline.py          Orchestration for the three stages.

tools/
  build_label_sheet.py Thin CLI over label_sheet.pipeline (extract/predict/
                       assemble/all).
  sync_photos_album.py Copy the "Green Crab AI 2026" shared album from Apple
                       Photos into data/raw (osxphotos or AppleScript fallback).
```

## 4. Canonical schema

`label_sheet/schema.py` is the single source of truth. Columns are grouped:

- **Identity / paths**: `year`, `dataset`, `crab_id`, `observation_id`,
  `session_id`, `original_filename`, `image_relpath`. (`thumbnail` is column A,
  rendered as an image, not stored as text.)
- **Ground truth**: `capture_date`, `molt_date`, `days_until_molt`,
  `known_molt_phase`, `sex`, `view`, `color_state`, `carapace_width_mm`,
  `outcome`, `label_source`, `label_confidence`, `source_notes`.
- **Aux metadata**: `condo_id`, `cell_id`, `degree_of_molt`, `is_in_situ`.
- **Model A (primary estimator)** prefix `m1_`: `m1_days_to_molt`, `m1_phase`,
  `m1_confidence`, `m1_sex`, `m1_view`, `m1_molt_indicators`,
  `m1_estimate_input`, `m1_in_training_set`.
- **Model B (bootstrap estimator)** prefix `m2_`: same fields plus
  `m2_days_to_molt_std` (bootstrap uncertainty).
- **Expert review** prefix `expert_`: `expert_status` (confirm/reject/update),
  `expert_days_to_molt`, `expert_molt_phase`, `expert_sex`, `expert_view`,
  `expert_molt_indicators`, `expert_notes`.

Controlled vocabularies reuse `field_data/SPREADSHEET_README.md` verbatim
(phase: intermolt / pre_molt / peeler_imminent / molted / dead / unknown; view:
ventral / dorsal / side / unknown; sex: male / female / unknown; confidence:
high / medium / low / unknown). Keeping these identical means the new sheet is
parseable by the same downstream tooling.

## 5. Per-year label sources — RAW ONLY

**The pipeline reads only from `data/raw`.** No extractor reads a pre-existing
`data/processed` label file. The only things written under `data/processed` are
this pipeline's own outputs (`data/processed/label_sheet/…`, including crate
images that the crate extractor re-extracts from the raw `.docx` on each run).
Model *weights* under `models/` are used for the predict stage — those are
trained artifacts, not labels.

| Sheet    | Images from                                     | Labels from (all raw)                                                        |
|----------|-------------------------------------------------|------------------------------------------------------------------------------|
| `2016`   | `data/raw/NH Green Crab Project 2016/`          | Folder names (crab id, capture date `8:26`, molt tag) + the raw workbook `Green Crabs September 2016.xlsx` ("General Data": per-crab molt date + carapace). Workbook molt date/carapace override folder parsing. |
| `2017`   | `data/raw/NH Green Crab Project -Doyle Fellowship 2017/` | Folder names (`(molted 8.x)`, crab id, capture date). The raw `molted crabs.xlsx` is a free-form grid and is not parsed; folder names carry the per-crab molt dates (~546 labeled). |
| `2018`   | `data/raw/2018 NH Green Crab-Doyle Fellowship/` | Folder names. The raw `2018 Crab Data.xlsx` holds only aggregate daily molt/mortality counts (no per-crab per-image molt date), so it is not joined. Sparse ground truth by design. |
| `2019_crate` | Re-extracted from `data/raw/Crate {1,2,3} - Completed/Crab *.docx` into `data/processed/label_sheet/crate_images/` | The same raw `.docx`, parsed in place with stdlib `zipfile`+`xml.etree` (no `python-docx`): table row (crab #, carapace, color, degree-of-molt) + body date paragraphs. `degree_of_molt`→`days_until_molt` via a documented heuristic. **Year = 2019** (from `Salinity and Temp Monitoring Crate 1.docx`: 6/21/2019–7/8/2019; Observations: "Experiment Begins June 19th"). |
| `2026`   | `data/raw/Green Crab AI 2026/Green Crab AI Photo Database/**` | **Self-describing filenames** `MMDDYY_JEL_<A/B/C>_<cell>_<view>` (e.g. `052926_JEL_A_A1_Dorsal.jpeg`) give date + condo + cell (A1–F6) + view directly, joined by (condo, cell) to the **typed** `Green Crab AI Condo Data Sheet.xlsx` (molt phase, Moltmeter predictor, handwritten notes). Condo blocks: JEL-A rows 1–36, JEL_B 37–72, JEL_C 73–108. `Red Crabs/`→`red_green_crab`, `In Situ Crabs/`→`is_in_situ`. **No OCR of the photographed paper-notebook pages is performed — those pages are not on disk; only the typed .xlsx ships in the album.** |

### 2026 photo-to-label mapping

The 2026 collection turned out to be organized under
`Green Crab AI Photo Database/<MM_DD_2026>/JEL_<A|B|C>/` with **self-describing
filenames** (`052926_JEL_A_A1_Dorsal.jpeg`), so the image→cell→view mapping is
read directly from the filename rather than guessed from capture order. Cells are
joined by `(condo, cell)` to the typed condo data sheet; these rows get
`label_confidence = high` and `label_source = "filename + condo_sheet (condo,cell join)"`.
`Seam`/`Molt` close-ups keep `view = unknown` but record the feature in
`source_notes`. `Red Crabs/` and `In Situ Crabs/` folders are tagged via
`color_state` / `is_in_situ`. Flat top-level `IMG_*.jpeg` are unsorted camera
originals — kept for a complete inventory but not cell-mapped.

An older EXIF-capture-order heuristic (used before the organized folders were
available) has been removed. The photographed **paper notebook** pages are the
only source that could reveal transcription errors in the typed sheet, and they
are not on disk here — reconciling them requires the shared-album sync
(§9). This is where OCR would apply.

## 6. Models

Two runners are registered in `label_sheet/config.py`. Both emit `days_to_molt`
plus aux labels, and both are swappable by editing the registry.

- **`m1` primary estimator** — `vit_base` torchvision features (768-d) +
  `models/molt_regressor_vit_random_forest.joblib`. This is the model the
  deployed FastAPI app loads, so "most recent molt estimator" = what ships.
- **`m2` bootstrap estimator** — OpenCLIP `ViT-H-14 / laion2b_s32b_b79k`
  features (1024-d) + `models/openclip_regressor.joblib` (newest estimator
  artifact, Dec 2025). Bootstrap uncertainty (`m2_days_to_molt_std`) is produced
  by resampling the random-forest tree predictions. If OpenCLIP weights are not
  available in the environment, `m2` degrades gracefully to a bootstrap over the
  ViT features so the column group is still populated and clearly labeled.

Aux labels (sex, view, molt indicators, position) come from
`models/auxlabels.py`. Sex/view use lightweight image heuristics today (color,
aspect, symmetry) and are structured so a trained tagger can drop in behind the
same interface. Molt indicators (shell transparency/color, seam/side cracking)
are derived from color statistics of the crab crop and expressed as short
human-readable tags. All of this is explicitly provisional and flagged for
expert confirmation — the point of the sheet is to collect those confirmations.

`m*_estimate_input` records whether the estimate used a detector crop
(`yolo_crop`) or the whole image (`whole_image_fallback`), matching the
vocabulary already used in the field-data spreadsheet.

## 7. Model provenance comments

`xlsx_writer.py` attaches an Excel cell comment to the header of each model
column group. The comment text is generated from the `ModelConfig` (id, human
name, weights path, feature extractor, git commit, run timestamp) so it always
matches what actually ran. This satisfies "a comment on the spreadsheet to
indicate what model was used for what estimated labels."

## 8. Running it

```bash
# One-shot: extract labels, run model inference (cached), build the workbook.
venv/bin/python tools/build_label_sheet.py all

# Or stage by stage:
venv/bin/python tools/build_label_sheet.py extract
venv/bin/python tools/build_label_sheet.py predict           # heavy; resumable
venv/bin/python tools/build_label_sheet.py assemble

# Cheap sheet without ML (thumbnails + labels only), useful for review setup:
venv/bin/python tools/build_label_sheet.py all --no-models

# Limit inference while validating the pipeline:
venv/bin/python tools/build_label_sheet.py predict --limit 50
```

The Apple Photos sync is a separate, explicit step (it touches the Photos
library, so it is never run implicitly by the build):

```bash
venv/bin/python tools/sync_photos_album.py --album "Green Crab AI 2026" \
    --dest "data/raw/Green Crab AI 2026"
```

## 9. Current status / handoff notes

- **Apple Photos sync: DONE.** The shared album was synced into
  `data/raw/Green Crab AI 2026/` (now ~2,800 image files incl. 1,479 HEIC
  live-photos + `.mov`). `osxphotos` needed Full Disk Access first (it's an iCloud
  *Shared Album*). The 2026 extractor **excludes HEIC as image rows** because they
  largely duplicate the jpeg set (e.g. `IMG_2543.jpeg` vs `IMG_2543 (1).HEIC`);
  revisit with a content-hash dedup if the HEIC-only originals are needed. `.mov`
  videos are ignored (not in `IMAGE_EXTENSIONS`).
- **2026 notebook OCR: DONE.** The photographed paper-notebook data pages are
  `IMG_3637/3638/3639.HEIC` (captured 2026-06-05 11:10) — per-cell molt-cue notes
  for JEL-A/B/C, cells A1–F6. Transcribed into
  `data/curated/notebook_2026-06-05_transcription.csv` (96 cells; JEL-C E/F not
  photographed). The transcription is regenerable by
  `tools/ocr_notebook_pages.py` (Claude vision + Structured Outputs; needs
  `pip install anthropic` + an API key / `ant auth login`); it stays a reviewable
  artifact because handwriting OCR is model-based, not deterministic.
  - The jpeg `IMG_27xx` "pages" are **column-divider markers**: a divider photo of
    "JEL-B / C" means the photos captured immediately after it (by EXIF time) are
    that condo + column. Verified against the labeled photo-database filenames —
    every 5/29 divider sits exactly at its column boundary — which independently
    confirms the filename cell labels. This is the scheme that produced the
    `JEL_X_cellN` filenames. The dense `IMG_363x` pages instead describe the crabs
    themselves (a later 6/5 session).
  - The notebook is a **6/5 follow-up**; the condo `.xlsx` is the **5/29 initial**
    assessment (Date Added 5/29). So the notebook is an added time-point, not a
    duplicate. **No systematic cell misalignment** was found: cells the notebook
    marks "empty" on 6/5 coincide positionally with cells the xlsx marks removed
    before 6/5. Four cells (JEL-A C2, JEL-A F6, JEL-B B1, JEL-B F4) are xlsx-removed
    yet have a 6/5 note — flagged for expert review (restock? recording gap?).
  - Per the data owner, the 96 notebook observations are added to the **2026 sheet
    as image-less rows** (blank `image_relpath`/`abs_path` → no thumbnail, no model
    inference), via `year_2026._notebook_records`. `view = notebook_observation`,
    raw cue text in `degree_of_molt` + `source_notes`, phase best-effort in
    `known_molt_phase`. Red-crab (308) and in-situ (41) images stay as inventory
    rows.
- `openpyxl`, `torch`, `ultralytics`, `open_clip`, `PIL`, `sklearn` are all
  present in `venv/`. `python-docx` is NOT needed: the crate extractor parses raw
  `.docx` with stdlib `zipfile`+`xml.etree`.
- Reproducibility: `tools/build_label_sheet.py all` regenerates the entire
  workbook from `data/raw` alone. Deleting `data/processed/label_sheet/` (records,
  predictions cache, thumbnails, re-extracted crate images) and re-running
  reproduces byte-for-byte-equivalent labels; only the model predict stage needs
  the `models/*.joblib` weights.
- Model joblibs were pickled with scikit-learn 1.7.1; `venv/` has 1.8.0 and
  emits an `InconsistentVersionWarning`. Predictions still load and run; pin
  `scikit-learn==1.7.1` if you want to silence it.
- Open decisions a reviewer may want to revisit: (a) exact "bootstrap" model
  choice; (b) the 2026 view-triple grouping rule; (c) whether the crate study
  belongs to a specific calendar year. All three are isolated to single modules.
