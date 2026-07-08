# `label_sheet` package

Builds the centralized, per-year green crab **label workbook**
(`data/processed/green_crab_label_sheet.xlsx`) — one worksheet per study year,
identical columns across years, each row an image with an embedded thumbnail,
extracted ground-truth labels + metadata, two models' `days_to_molt` and
aux-label estimates (with a provenance comment), and expert confirm/reject/update
columns.

Full design and rationale: [`docs/LABEL_SHEET_PIPELINE.md`](../docs/LABEL_SHEET_PIPELINE.md).

## Quick start

```bash
venv/bin/python tools/build_label_sheet.py all          # extract + predict + assemble
venv/bin/python tools/build_label_sheet.py all --no-models   # labels + thumbnails only
```

## Three decoupled stages

| Stage      | Command                                     | Output                                        | Cost   |
|------------|---------------------------------------------|-----------------------------------------------|--------|
| extract    | `build_label_sheet.py extract`              | `data/processed/label_sheet/records.csv`      | cheap  |
| predict    | `build_label_sheet.py predict [--limit N]`  | `data/processed/label_sheet/predictions.csv`  | heavy, resumable/cached |
| assemble   | `build_label_sheet.py assemble [--no-models]` | `data/processed/green_crab_label_sheet.xlsx` | cheap  |

## Where things live

- `schema.py` — the one canonical column list + controlled vocab. Change columns here.
- `config.py` — repo paths, `YEARS` (worksheets), `MODELS` (the two estimators).
- `extractors/` — one module per label source; registered in `extractors/__init__.py`.
- `models/` — feature extractors, estimator runner, aux taggers, prediction cache.
- `xlsx_writer.py` — openpyxl multi-sheet writer (thumbnails, comments, dropdowns).
- `pipeline.py` — orchestration.

## Extending

- **New year / source**: add a module in `extractors/`, register it, add a
  `YearConfig` to `config.YEARS`.
- **Swap a model**: edit the matching `ModelConfig` in `config.MODELS` (weights,
  feature extractor, bootstrap). The provenance comment updates automatically.
- **Better sex/view tagging**: replace `models/auxlabels.py:AuxTagger.tag` — the
  `AuxResult` contract is stable.

The separate `tools/sync_photos_album.py` copies the "Green Crab AI 2026" shared
Apple Photos album into `data/raw` (osxphotos or AppleScript). It is never run
implicitly by the build.
