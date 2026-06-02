# Cloud Run (CPU) Deployment

This is the lowest-cost deployment path for short burst usage.

## Build and deploy

```bash
gcloud auth login
gcloud config set project greencrabmoltestimator

gcloud builds submit --tag gcr.io/greencrabmoltestimator/green-crabs

gcloud run deploy green-crabs \
  --project greencrabmoltestimator \
  --image gcr.io/greencrabmoltestimator/green-crabs \
  --region us-central1 \
  --platform managed \
  --min-instances 1 \
  --cpu 4 \
  --memory 8Gi \
  --timeout 300 \
  --concurrency 1 \
  --allow-unauthenticated
```

To update the runtime settings on the current service:

```bash
gcloud run services update green-crabs \
  --project greencrabmoltestimator \
  --region us-central1 \
  --min-instances 1 \
  --cpu 4 \
  --memory 8Gi \
  --concurrency 1
```

## Recommended env vars

```bash
gcloud run services update green-crabs \
  --project greencrabmoltestimator \
  --region us-central1 \
  --set-env-vars INFERENCE_MODE=cpu \
  --set-env-vars DETECTION_ENABLED=true \
  --set-env-vars YOLO_MODEL_PATH=models/fathomnet_mvp_yolov8_1280_20240914.pt \
  --set-env-vars YOLO_CONF_MIN=0.25 \
  --set-env-vars YOLO_MAX_DETECTIONS=5 \
  --set-env-vars MODEL_LOAD_ASYNC=true \
  --set-env-vars MAX_CONCURRENT_INFERENCES=1
```

Notes:
- `YOLO_MODEL_PATH` pins the detector to the FathomNet MVP YOLOv8 model. If it
  is not set, the app chooses the FathomNet model when present, then falls back
  to `yolov8n.pt` if present.
- `YOLO_CONF_MIN=0.25` and `YOLO_MAX_DETECTIONS=5` show the top five crab
  detections above 25% confidence.
- For faster cold starts, keep models inside the container image.

## Local smoke test (Docker)

```bash
docker build -t green-crabs .
docker run -p 8080:8080 green-crabs
```

Open: http://localhost:8080/ui

The container runs `app_fastapi:app` on port `8080`, which matches the local
`python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8080` flow.
