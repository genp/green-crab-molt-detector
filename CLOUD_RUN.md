# Cloud Run (CPU) Deployment

This is the lowest-cost deployment path for short burst usage.

## Build and deploy

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/green-crabs

gcloud run deploy green-crabs \
  --image gcr.io/YOUR_PROJECT_ID/green-crabs \
  --region us-central1 \
  --platform managed \
  --cpu 2 \
  --memory 4Gi \
  --timeout 300 \
  --concurrency 4 \
  --allow-unauthenticated
```

## Recommended env vars

```bash
gcloud run services update green-crabs \
  --set-env-vars INFERENCE_MODE=cpu \
  --set-env-vars DETECTION_ENABLED=true \
  --set-env-vars MODEL_LOAD_ASYNC=true \
  --set-env-vars MAX_CONCURRENT_INFERENCES=2
```

Notes:
- If `DETECTION_ENABLED=true` and no `YOLO_MODEL_PATH` is set, the service uses
  `yolov8n.pt` if present (fast CPU detector) and falls back to the baseline
  model in `models/` if not.
- For faster cold starts, keep models inside the container image.

## Local smoke test (Docker)

```bash
docker build -t green-crabs .
docker run -p 8080:8080 green-crabs
```

Open: http://localhost:8080/ui

The container runs `app_fastapi:app` on port `8080`, which matches the local
`python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8080` flow.
