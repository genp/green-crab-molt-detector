# AGENTS.md - Green Crabs repo working notes

## Purpose
Green crab molt phase prediction from images to support harvest timing (peeler stage 0-3 days before molt). System includes single-shot models (YOLO/CNN/ViT) and temporal models; temporal achieves sub-1-day MAE and is commercially viable.

## Primary references
- Project overview and quick start: `README.md`, `QUICKSTART.md`
- Deployment guides: `DEPLOYMENT.md`, `HEROKU_DEPLOYMENT.md`
- Research context/results: `cvpr_paper_simple.tex`, `cvpr_workshop_paper.tex`, `PRESENTATION_README.md`, `TEST_RESULTS_SUMMARY.md`
- App entry points: `app.py` (Flask), `app_fastapi.py` (FastAPI)

## Current app + model status (from prior agent notes)
- FastAPI app exists in `app_fastapi.py` with endpoints:
  - `/predict` for single images
  - `/predict_stream` for frame uploads
  - `/health` for health checks
  - `/ui` serving `templates/index.html`
- Optional YOLO bounding boxes via `YOLO_MODEL_PATH` env var.
- Concurrency guard via `MAX_CONCURRENT_INFERENCES` env var (default 2).
- A front-end camera stream was added to `templates/index.html` to post frames to `/predict_stream` and display returned thumbnails.

## Models and performance (from papers)
- Dataset: 230 images, 11 crabs, time-series with dorsal/ventral views; class imbalance (81.7% female).
- Single-shot results (MAE days):
  - YOLO+SVR: 5.01
  - CNN+SVR: 5.28
  - ViT+NN: 4.77 (best single-shot)
- Temporal results:
  - Temporal RF: 0.48 MAE (94% @ 3 days) -> commercially viable
  - Temporal GB: 0.52 MAE
- Best temporal window: k=7 observations (0.48 MAE).

## Model files (local)
- Trained models under `models/` (e.g., `best_vit_regressor.joblib`, `best_yolo_regressor.joblib`, `best_temporal_model.pkl`).

## Development principles (from CLAUDE.md)
- DRY, modular design.
- Full type hints.
- Verbose commenting for complex logic.
- Frequent, clear git commits.

## Local setup
- Install deps: `pip install -r requirements.txt`
- Run full pipeline: `python run_pipeline.py`
- Run Flask app: `python app.py`
- Run FastAPI app: `uvicorn app_fastapi:app --host 0.0.0.0 --port 5001`

## Deployment
- See `DEPLOYMENT.md` for AWS/Cloud Run/DO/Docker.
- `HEROKU_DEPLOYMENT.md` covers Heroku + S3 model storage.

## Notes for future agents
- If choosing a model for real-time use, temporal models are most accurate but require sequences; ViT or YOLO regressors are likely faster per frame.
- Validate latency before selecting the “fastest acceptable” model for live video.
