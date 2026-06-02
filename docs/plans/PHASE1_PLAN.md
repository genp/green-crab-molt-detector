# Phase 1: Video UI Overhaul - Implementation Plan

## Context from Field Test (May 29, 2026)

**Environment**: Processing ~100 male green crabs in a cooler, dynamic field conditions

**Key Problems Identified**:
1. Video and results displayed in separate areas → hard to focus on crab
2. UI feels "janky" - not smooth or interactive
3. Hard to get real-time feedback on crab positioning
4. Need to process large volumes quickly
5. Results not fast enough for interactive workflow

**User Need**: "The advice of the app has to be fast, easier to consume, more lightweight, more useful for the candidate user who is a marine fisheries worker, new to green crabs, who wants to process a large amount of crabs in a dynamic environment quickly while learning to discern the molt phase."

## Phase 1 Goals

Create a real-time, interactive video experience with:
1. Detection overlays directly on video stream
2. Color-coded molt phase categories
3. Days-to-molt estimates on overlay
4. Smooth, responsive feel
5. Session export for validation

## Technical Architecture

### Current System (app_fastapi.py)
- **Frontend**: templates/index.html with getUserMedia video streaming
- **Backend**: FastAPI with `/predict_stream` endpoint
- **Pipeline**:
  ```
  Image → YOLO Detection → Crop to bbox → ViT Feature Extraction → Molt Regression
  ```
- **Models**:
  - YOLO: models/fathomnet_mvp_yolov8_1280_20240914.pt
  - ViT: models/M7_vit_RMSE_1.07_R2_0.99.joblib
  - Feature extractor: GeneralCrustaceanFeatureExtractor (ViT-B/16)

### Key Functions (app_fastapi.py:261-316)
- `run_detection()`: YOLO object detection
- `filter_bboxes()`: Filter by confidence, area, aspect ratio
- `select_primary_bbox()`: Pick highest confidence detection
- `crop_to_bbox()`: Crop image to detected region

## Molt Phase Categories

Based on codebase analysis (tools/label_blue_cooler_field_images.py, test_crab_with_images.py):

| Phase | Days to Molt | Description | Priority |
|-------|--------------|-------------|----------|
| **Peeler** | < 1 day | Currently or just about to molt - HARVEST NOW | Critical |
| **Imminent** | < 3 days | Approaching molt - monitor closely | High |
| **Pre-molt Near** | < 5 days | Near pre-molt phase | Medium |
| **Pre-molt Later** | < 14 days | Early pre-molt / feeding phase | Medium |
| **Intermolt** | > 14 days | Extended intermolt / recently molted | Low |

**Note**: Current codebase uses 0-3 days as "peeler" window, but field experience suggests < 1 day is more accurate for harvest timing.

### Color Coding for Overlay

```python
def categorize_molt_phase(days_to_molt: float) -> dict:
    """Categorize molt phase and assign color."""
    if days_to_molt < 1:
        return {
            "category": "peeler",
            "color": "#ff0000",  # Red - URGENT
            "label": "PEELER - HARVEST NOW"
        }
    elif days_to_molt < 3:
        return {
            "category": "imminent",
            "color": "#ff6600",  # Orange-red
            "label": "IMMINENT"
        }
    elif days_to_molt < 5:
        return {
            "category": "premolt_near",
            "color": "#ffaa00",  # Orange
            "label": "PRE-MOLT (NEAR)"
        }
    elif days_to_molt < 14:
        return {
            "category": "premolt_later",
            "color": "#ffdd00",  # Yellow
            "label": "PRE-MOLT"
        }
    else:
        return {
            "category": "intermolt",
            "color": "#44ff44",  # Green
            "label": "INTER-MOLT"
        }
```

## Implementation Plan

### 1. Real-Time Video Overlay with Color-Coded Detections

**Current State**:
- Video in one area, results in separate section
- `/predict_stream` returns JSON with predictions
- Frontend manually updates results section

**Target State**:
- Canvas overlay on top of video stream
- Bounding boxes drawn directly on video
- Color-coded by molt phase category
- Days-to-molt displayed inside bbox
- Confidence scores shown

**Technical Approach**:

#### Backend Changes (app_fastapi.py):

1. Add molt phase categorization function (after line 260):
   ```python
   def categorize_molt_phase(days_to_molt: float) -> dict:
       """Categorize molt phase and assign color for display."""
       if days_to_molt < 1:
           return {
               "category": "peeler",
               "color": "#ff0000",
               "label": "PEELER - HARVEST NOW"
           }
       elif days_to_molt < 3:
           return {
               "category": "imminent",
               "color": "#ff6600",
               "label": "IMMINENT"
           }
       elif days_to_molt < 5:
           return {
               "category": "premolt_near",
               "color": "#ffaa00",
               "label": "PRE-MOLT (NEAR)"
           }
       elif days_to_molt < 14:
           return {
               "category": "premolt_later",
               "color": "#ffdd00",
               "label": "PRE-MOLT"
           }
       else:
           return {
               "category": "intermolt",
               "color": "#44ff44",
               "label": "INTER-MOLT"
           }
   ```

2. Enhance `/predict_stream` response to include detection metadata:
   ```python
   # After prediction is made (around line 450)
   phase_info = categorize_molt_phase(prediction['days_to_molt'])

   return {
       "days_to_molt": prediction['days_to_molt'],
       "molt_date": prediction['molt_date'],
       "phase": phase_info['category'],
       "phase_label": phase_info['label'],
       "phase_color": phase_info['color'],
       "bbox": bbox_coords,  # [x1, y1, x2, y2]
       "confidence": bbox_confidence,
       "frame_width": original_width,
       "frame_height": original_height,
       "timestamp": datetime.now().isoformat()
   }
   ```

#### Frontend Changes (templates/index.html):

1. Add canvas overlay on video element:
   ```html
   <div id="videoContainer" style="position: relative; display: inline-block;">
     <video id="videoElement" autoplay playsinline></video>
     <canvas id="overlayCanvas" style="position: absolute; top: 0; left: 0; pointer-events: none;"></canvas>
   </div>
   ```

2. Sync canvas size with video:
   ```javascript
   video.addEventListener('loadedmetadata', () => {
     canvas.width = video.videoWidth;
     canvas.height = video.videoHeight;
   });
   ```

3. Draw detections on canvas:
   ```javascript
   function drawDetection(prediction, canvas, video) {
     const ctx = canvas.getContext('2d');
     ctx.clearRect(0, 0, canvas.width, canvas.height);

     if (!prediction.bbox) return;

     // Scale bbox to canvas dimensions
     const scaleX = canvas.width / prediction.frame_width;
     const scaleY = canvas.height / prediction.frame_height;
     const [x1, y1, x2, y2] = prediction.bbox;
     const x = x1 * scaleX;
     const y = y1 * scaleY;
     const w = (x2 - x1) * scaleX;
     const h = (y2 - y1) * scaleY;

     // Draw bbox with color coding
     ctx.strokeStyle = prediction.phase_color;
     ctx.lineWidth = 4;
     ctx.strokeRect(x, y, w, h);

     // Draw label background
     const labelText = `${prediction.days_to_molt.toFixed(1)}d - ${prediction.phase_label}`;
     ctx.font = 'bold 18px Arial';
     const textWidth = ctx.measureText(labelText).width;

     ctx.fillStyle = prediction.phase_color;
     ctx.fillRect(x, y - 35, textWidth + 20, 35);

     // Draw text
     ctx.fillStyle = 'white';
     ctx.fillText(labelText, x + 10, y - 10);

     // Draw confidence if available
     if (prediction.confidence) {
       ctx.font = '14px Arial';
       ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
       ctx.fillText(`${(prediction.confidence * 100).toFixed(0)}%`, x + 10, y + h - 10);
     }
   }
   ```

4. Update frame capture loop (reduce interval for smoothness):
   ```javascript
   const FRAME_INTERVAL_MS = 200; // 5 FPS for smooth feel

   async function captureAndSendFrame() {
     if (!isStreaming) return;

     // Show processing indicator
     document.getElementById('processingIndicator').style.display = 'block';

     // Capture frame
     const canvas = document.createElement('canvas');
     canvas.width = video.videoWidth;
     canvas.height = video.videoHeight;
     canvas.getContext('2d').drawImage(video, 0, 0);

     // Send to backend
     const blob = await new Promise(resolve =>
       canvas.toBlob(resolve, 'image/jpeg', 0.8)
     );
     const formData = new FormData();
     formData.append('file', blob);

     try {
       const response = await fetch('/predict_stream', {
         method: 'POST',
         body: formData
       });
       const prediction = await response.json();

       // Draw detection on overlay
       drawDetection(prediction, overlayCanvas, video);

       // Record for session export
       recordFrame(blob, prediction);

     } catch (error) {
       console.error('Prediction error:', error);
     } finally {
       // Hide processing indicator
       document.getElementById('processingIndicator').style.display = 'none';

       // Schedule next frame
       setTimeout(captureAndSendFrame, FRAME_INTERVAL_MS);
     }
   }
   ```

### 2. Smooth Video Streaming with Lower Latency

**Performance Optimizations**:

1. **Reduce frame interval**: 800ms → 200ms (5 FPS) for interactive feel
2. **Optimize image encoding**: JPEG quality 0.8, resize if > 640px width
3. **Add temporal smoothing** to reduce prediction jitter:

   ```python
   # Add to app_fastapi.py
   from collections import deque
   import numpy as np

   # Global smoothing buffer (per-session in production)
   prediction_history = deque(maxlen=5)

   def smooth_prediction(days_to_molt: float) -> float:
       """Apply temporal smoothing to reduce jitter."""
       prediction_history.append(days_to_molt)
       return float(np.mean(prediction_history))

   # In /predict_stream endpoint:
   raw_prediction = model.predict(features)
   smoothed_prediction = smooth_prediction(raw_prediction)
   ```

4. **Add visual feedback during processing**:
   ```html
   <div id="processingIndicator" class="processing-indicator">
     <div class="spinner"></div> Processing...
   </div>
   ```

   ```css
   .processing-indicator {
     position: absolute;
     top: 10px;
     right: 10px;
     background: rgba(0, 0, 0, 0.7);
     color: white;
     padding: 8px 12px;
     border-radius: 4px;
     display: none;
   }

   .spinner {
     display: inline-block;
     width: 12px;
     height: 12px;
     border: 2px solid rgba(255,255,255,.3);
     border-radius: 50%;
     border-top-color: white;
     animation: spin 0.6s linear infinite;
   }

   @keyframes spin {
     to { transform: rotate(360deg); }
   }
   ```

### 3. Redesigned Photo UI with Inline Detection

**Layout Changes**:

```html
<!-- BEFORE: Photo upload at top -->
<!-- AFTER: Video first, photo at bottom -->

<div class="container">
  <!-- PRIMARY: Video Section -->
  <section id="videoSection" class="main-section">
    <h2>Live Camera Detection</h2>
    <div id="videoContainer">
      <video id="videoElement"></video>
      <canvas id="overlayCanvas"></canvas>
    </div>
    <div class="controls">
      <button id="startCamera">Start Camera</button>
      <button id="stopCamera">Stop Camera</button>
      <button id="exportSession">Export Session</button>
    </div>
  </section>

  <!-- SECONDARY: Photo Section (at bottom) -->
  <section id="photoSection" class="secondary-section">
    <h2>Upload Photo (Optional)</h2>
    <input type="file" id="photoUpload" accept="image/*">

    <!-- Results shown inline with detection overlay -->
    <div id="photoResultContainer" style="display: none;">
      <div style="position: relative; display: inline-block;">
        <img id="uploadedPhoto" style="max-width: 100%;">
        <canvas id="photoOverlay" style="position: absolute; top: 0; left: 0;"></canvas>
      </div>
    </div>
  </section>
</div>
```

**Photo Upload Handler**:
```javascript
document.getElementById('photoUpload').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Display image
  const img = document.getElementById('uploadedPhoto');
  const reader = new FileReader();
  reader.onload = (e) => {
    img.src = e.target.result;
    img.onload = () => {
      // Size canvas to match image
      const canvas = document.getElementById('photoOverlay');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
    };
  };
  reader.readAsDataURL(file);

  // Send to backend
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/predict', {
    method: 'POST',
    body: formData
  });
  const prediction = await response.json();

  // Draw detection on photo overlay
  drawDetection(prediction, document.getElementById('photoOverlay'), img);

  document.getElementById('photoResultContainer').style.display = 'block';
});
```

### 4. About Us Page

**File**: `templates/about.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About MoltMeter - Green Crab Molt Detection</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">MoltMeter</a>
            <a class="nav-link text-white" href="/about-page">About</a>
        </div>
    </nav>

    <div class="container mt-5">
        <h1>About MoltMeter</h1>

        <section class="mt-4">
            <h2>Our Mission</h2>
            <p class="lead">
                Enable sustainable green crab harvesting through AI-powered molt detection.
            </p>
            <p>
                MoltMeter helps fisheries workers identify green crabs at optimal harvest timing
                by predicting molt phase from images. Our technology supports the development
                of a sustainable soft-shell crab industry while managing invasive green crab populations.
            </p>
        </section>

        <section class="mt-4">
            <h2>How It Works</h2>
            <ul>
                <li>Upload or stream live video of green crabs</li>
                <li>AI analyzes visual indicators to predict days until molt</li>
                <li>Color-coded overlays show molt phase categories</li>
                <li>Identify "peeler" crabs ready for immediate harvest</li>
            </ul>
        </section>

        <section class="mt-4">
            <h2>Use Cases</h2>
            <ul>
                <li><strong>Commercial Fisheries</strong>: Optimize harvest timing for soft-shell crab operations</li>
                <li><strong>Marine Research</strong>: Study molt cycles and green crab biology</li>
                <li><strong>Invasive Species Management</strong>: Support sustainable utilization of invasive populations</li>
            </ul>
        </section>

        <section class="mt-4">
            <h2>Technology</h2>
            <p>
                MoltMeter leverages computer vision for ecology technologies, trained on hundreds
                of green crab molt cycles to provide accurate predictions in field conditions.
            </p>
        </section>

        <section class="mt-4">
            <h2>Contact</h2>
            <p>Interested in using MoltMeter for your operation or research?</p>
            <div class="card p-3">
                <p class="mb-2">
                    <strong>Gabriela Brandt</strong><br>
                    UNH Sea Grant
                </p>
                <p class="mb-0">
                    <strong>Genevieve Patterson</strong><br>
                    Barderry Applied Research, LLC
                </p>
            </div>
        </section>

        <section class="mt-5 mb-5">
            <a href="/" class="btn btn-primary">Try MoltMeter</a>
        </section>
    </div>
</body>
</html>
```

**Backend Endpoint** (add to app_fastapi.py):
```python
@app.get("/about-page", response_class=HTMLResponse)
async def about_page():
    """About page for customers and stakeholders."""
    with open(BASE_PATH / "templates" / "about.html") as f:
        return f.read()
```

**Update Navigation** in index.html:
```html
<nav class="navbar navbar-dark bg-dark">
    <div class="container">
        <a class="navbar-brand" href="/">MoltMeter</a>
        <a class="nav-link text-white" href="/about-page">About</a>
    </div>
</nav>
```

### 5. Session Export Feature

**Purpose**: Export all frames and predictions from a video session for validation and analysis.

**Implementation** (Client-Side using JSZip):

1. **Add JSZip library** to templates/index.html:
   ```html
   <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
   ```

2. **Session storage**:
   ```javascript
   const sessionData = {
     sessionId: Date.now().toString(),
     startTime: new Date().toISOString(),
     frames: []
   };

   function recordFrame(imageBlob, prediction) {
     sessionData.frames.push({
       timestamp: new Date().toISOString(),
       image: imageBlob,
       prediction: prediction,
       frameNumber: sessionData.frames.length
     });
   }
   ```

3. **Export button handler**:
   ```javascript
   document.getElementById('exportSession').addEventListener('click', async () => {
     if (sessionData.frames.length === 0) {
       alert('No frames recorded yet. Start camera to begin recording.');
       return;
     }

     const zip = new JSZip();

     // Add metadata
     zip.file('session_metadata.json', JSON.stringify({
       sessionId: sessionData.sessionId,
       startTime: sessionData.startTime,
       endTime: new Date().toISOString(),
       frameCount: sessionData.frames.length,
       deviceInfo: navigator.userAgent
     }, null, 2));

     // Add frames with predictions
     for (let i = 0; i < sessionData.frames.length; i++) {
       const frame = sessionData.frames[i];
       const frameNum = i.toString().padStart(4, '0');

       // Add image
       zip.file(`frames/frame_${frameNum}.jpg`, frame.image);

       // Add prediction JSON
       zip.file(`predictions/frame_${frameNum}.json`, JSON.stringify({
         frameNumber: i,
         timestamp: frame.timestamp,
         prediction: frame.prediction
       }, null, 2));
     }

     // Generate and download
     const content = await zip.generateAsync({
       type: 'blob',
       compression: 'DEFLATE',
       compressionOptions: { level: 6 }
     });

     const url = URL.createObjectURL(content);
     const a = document.createElement('a');
     a.href = url;
     a.download = `moltmeter_session_${sessionData.sessionId}.zip`;
     a.click();
     URL.revokeObjectURL(url);

     alert(`Exported ${sessionData.frames.length} frames successfully!`);
   });
   ```

4. **Clear session button** (optional):
   ```javascript
   function clearSession() {
     if (confirm('Clear all recorded frames?')) {
       sessionData.frames = [];
       sessionData.sessionId = Date.now().toString();
       sessionData.startTime = new Date().toISOString();
       alert('Session cleared.');
     }
   }
   ```

## File Structure After Phase 1

```
green_crabs/
├── app_fastapi.py              # Updated with categorization, enhanced endpoints
├── templates/
│   ├── index.html              # Redesigned UI with video overlay
│   └── about.html              # New About Us page
├── static/
│   ├── css/
│   │   └── main.css           # Updated styles
│   ├── js/
│   │   ├── video_overlay.js   # New: Canvas overlay logic
│   │   ├── session_export.js  # New: Export functionality (or inline)
│   │   └── utils.js           # Shared utilities
│   └── robots.txt
└── docs/plans/PHASE1_PLAN.md  # This document
```

## Current Implementation Status (2026-06-01)

### Complete or Mostly Complete
- [x] FastAPI stream endpoint exists: `/predict_stream` returns the enriched prediction payload used by `/predict`.
- [x] YOLO detection path exists with bbox detection, confidence/area/aspect filtering, primary bbox selection, and crop-to-bbox regression input.
- [x] Molt phase categorization exists via `get_molt_phase_category()` with phase, color, recommendation, and harvest-ready fields.
- [x] Prediction responses include detection metadata: bbox counts, primary bbox, all bboxes, image dimensions, crop/fallback flags, and detection debug fields.
- [x] Live camera stream is the primary UI section, above photo upload.
- [x] Canvas overlay is layered on the live video stream.
- [x] Video overlay drawing exists for bbox, molt phase, days-to-molt, and recommendation text.
- [x] Photo upload is secondary and has inline overlay support via `photoOverlay`.
- [x] About page exists at `templates/about.html` and is served by `/about-page`.
- [x] Home page navigation links to the About page.

### Partial
- [ ] 5 FPS target is configured with `streamIntervalMs = 200`, but actual throughput is backend-latency-bound because the frontend avoids overlapping requests with `sendingFrame`.
- [ ] Response shape differs from the original draft contract: current fields are `phase`, `color`, `primary_bbox`, and `bboxes`, rather than `phase_label`, `phase_color`, and `bbox`.
- [ ] Processing feedback exists through stream status and detection info, but the planned explicit `processingIndicator` spinner is not implemented.
- [ ] Frontend code remains inline in `templates/index.html`; planned `static/js/video_overlay.js`, `session_export.js`, and `utils.js` modules have not been split out.

### Not Yet Complete
- [ ] Session export is not implemented: no JSZip import, session frame store, export button, or ZIP creation flow.
- [ ] Temporal smoothing is not implemented: no prediction history buffer or `smooth_prediction()` path.
- [ ] Performance validation is still needed for frame latency, practical FPS, mobile behavior, and stutter/freezing.
- [ ] CPU detect+estimate latency is too slow for live field testing: current 4-vCPU gcloud instance takes about 5 seconds from known-crab frame upload to bbox overlay; target is 0.5 seconds or lower for this week's testing.
- [ ] Deployment validation is still needed for Docker/Cloud Run/moltmeter.ai and monitoring.
- [ ] Detector upgrade is still needed: train/deploy a new detector bootstrapped from reviewed SAM3 bbox outputs.
- [ ] Qualitatively test the SAM3-bootstrapped detector on `data/raw/Green Crab AI 2026` images, including hands, coolers, dorsal/ventral views, lighting variation, field negatives, and side-ish views.
- [ ] Multi-crab detection and estimation output is not implemented: current app selects one primary bbox, but field workflow needs per-crab boxes, per-crab days-to-molt estimates, and multi-crab overlay/result output.

## Testing Plan

### 1. Video Overlay Testing
- [ ] Color coding matches molt phase thresholds
- [ ] Bbox scales correctly at different resolutions
- [ ] Text labels are readable
- [ ] Overlay doesn't interfere with video playback

### 2. Performance Testing
- [ ] Frame processing latency < 250ms (target 200ms)
- [ ] CPU live-stream round-trip latency < 500ms on 4-vCPU gcloud test instance for a known single-crab frame.
- [ ] Add request timing instrumentation for upload decode, YOLO detect, bbox filter/crop, feature extraction, regression, JSON response, and frontend canvas draw.
- [ ] Test smaller stream inputs: client-side downscale to 640px or 416px max dimension before upload; JPEG quality 0.6-0.75.
- [ ] Benchmark detector-only latency for `yolov8n`/bootstrap detector at `imgsz=320`, `416`, and `640`.
- [ ] Benchmark estimate-only latency on a known crop to determine whether ViT feature extraction/regression or YOLO detection is the bottleneck.
- [ ] Add fast path for video stream: return bbox/phase metadata only, no server-generated thumbnail/base64 image.
- [ ] Avoid running full regression on every frame: detect every frame if cheap, estimate every N frames or only when bbox changes enough.
- [ ] Cache/reuse last good bbox and last estimate for short windows to keep the overlay responsive while inference catches up.
- [ ] Consider ONNX/OpenVINO/CoreML/TorchScript export for detector and feature extractor if Python/PyTorch CPU inference remains above target.
- [ ] 5 FPS feels smooth and interactive
- [ ] No stuttering or freezing
- [ ] Works on mobile devices

### 3. Session Export Testing
- [ ] Small session (10 frames) exports correctly
- [ ] Large session (500+ frames) exports without errors
- [ ] ZIP file structure is correct
- [ ] Predictions match exported images

### 4. About Page Testing
- [ ] All links work
- [ ] Contact info is accurate
- [ ] Mobile-responsive design

### 5. Field Testing
- [ ] Test with actual crabs in cooler
- [ ] Verify usability improvement vs. current UI
- [ ] Measure time to process 100 crabs
- [ ] Gather user feedback

### 6. Detector Bootstrap Testing
- [ ] Review a first batch of SAM3 proposals from `data/bootstrap_bboxes/all_raw_sam3_bootstrap_v1`.
- [ ] Export accepted reviewed boxes to a YOLO dataset with `tools/export_reviewed_bboxes_to_yolo.py`.
- [ ] Train or fine-tune a green crab detector from the reviewed SAM3-bootstrap labels.
- [ ] Run qualitative detector review on `data/raw/Green Crab AI 2026` images.
- [ ] Check failure modes: false positives on hands/gloves/cooler edges, missed crabs, poor ventral/dorsal coverage, side-view misses, and lighting/distance sensitivity.
- [ ] Keep the bootstrap effort split into two tracks:
  - detector track: bootstrap a stronger YOLO v2 from reviewed boxes and field negatives
  - estimator track: keep a separate MVP v2 discussion for the molt estimator and auxiliary tags
- [ ] If SAM3 bootstrapping is restarted, add review-time tags for `view` (`dorsal`, `ventral`, `side`, `unknown`), `sex` (`male`, `female`, `unknown`), and image quality/negative categories such as `human`, `oyster`, `cage`, `equipment`, `table`, `tray`, `vegetation`, `partial crab`, and `legs only`.
- [ ] Keep the detector target simple: one crab class for YOLO, with the negative diversity coming from the reviewed bootstrap set.
- [ ] Treat aux tags as downstream signals first; only move them into the detector if the tagged data shows a measurable gain in either detection quality or molt estimation.
- [ ] After bootstrap v2 updates land, deploy the corrected detector + params locally first, compare dogfood results on original vs downscaled 416px/Q0.65 frames, and only then promote to Cloud Run.
- [ ] After detector v2 is stable, decide whether to restart SAM3 with the tighter prompt/tag set or to begin a separate estimator MVP v2 run using `view`/`sex` tags.
- [ ] Retrain the estimator against the actual streaming-frame resolution and compression settings so molt predictions stay good on the frames sent by the video stream, not just on original high-resolution photos.

### 7. Multi-Crab Output Testing
- [ ] Update backend response to include a `crab_predictions` list: bbox, detection confidence, crop-used flag, days-to-molt, phase, color, and recommendation for each confident crab.
- [ ] Run molt regression independently on each accepted crab crop instead of only the selected primary bbox.
- [ ] Render multiple video/photo overlay boxes with stable labels so users can distinguish Crab 1, Crab 2, etc.
- [ ] Add a compact multi-crab result table for fast field review.
- [ ] Test with images containing multiple crabs and verify that per-crab labels do not overlap or obscure the video feed.

## Known Limitations & Future Work

### Current Limitations
1. **Distance sensitivity**: Poor performance > 6 inches from camera
2. **Lighting sensitivity**: Poor performance in shade/low light
3. **Multi-crab handling**: Currently processes one crab at a time
4. **No side-view support**: Trained only on top/bottom views

### Phase 2 Features
- Male/female sex detection
- Carapace size estimation
- Estimated market value
- Multi-crab detection and per-crab molt estimation output
- Multi-crab tracking (SORT/DeepSORT)
- Side-view training data collection
- Branding refresh

## Deployment Checklist

- [x] Update app_fastapi.py with molt phase categorization
- [x] Enhance /predict_stream response format
- [x] Update templates/index.html with canvas overlay
- [x] Create templates/about.html
- [ ] Add session export functionality
- [x] Update navigation to include About link
- [ ] Add temporal smoothing for live predictions
- [ ] Add explicit processing indicator/spinner for frame inference
- [ ] Add CPU latency instrumentation to `/predict_stream` and frontend frame loop.
- [ ] Optimize `/predict_stream` fast path to avoid thumbnail/base64 generation for live video.
- [ ] Downscale client video frames before upload and benchmark 320/416/640 detector image sizes.
- [ ] Profile YOLO detection versus crop feature extraction/regression on the 4-vCPU gcloud instance.
- [ ] Add frame-skipping/cached-estimate strategy so overlays update within 0.5 seconds even if full estimate runs slower.
- [ ] Qualitatively test current detector on `data/raw/Green Crab AI 2026`
- [ ] Train/fine-tune SAM3-bootstrapped detector from reviewed bbox outputs
- [ ] Test locally: `uvicorn app_fastapi:app --reload`
- [ ] Deploy SAM3-bootstrapped detector with `YOLO_MODEL_PATH`
- [ ] Qualitatively test SAM3-bootstrapped detector on `data/raw/Green Crab AI 2026`
- [ ] Add multi-crab detection and estimation output for all confident crab detections
- [ ] Build Docker image
- [ ] Deploy to Cloud Run
- [ ] Test on moltmeter.ai
- [ ] Monitor performance metrics

## Success Metrics

1. **Usability**: Process 100 crabs in < 20 minutes (vs. current struggles)
2. **Responsiveness**: CPU stream overlay latency < 500ms on a 4-vCPU gcloud instance; stretch target < 250ms
3. **Learning**: Color coding helps users identify molt phases
4. **Validation**: Collect 5+ session exports from field tests
5. **Adoption**: Generate interest via About page contact info

## References

- Field test feedback: User message 2026-06-01
- Molt phase code: tools/label_blue_cooler_field_images.py, test_crab_with_images.py
- Current detection pipeline: app_fastapi.py:261-316
- Frontend streaming: templates/index.html:500-630
- Models: models/fathomnet_mvp_yolov8_1280_20240914.pt, models/M7_vit_RMSE_1.07_R2_0.99.joblib
