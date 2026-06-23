const video = document.getElementById('demoVideo');
const overlay = document.getElementById('demoOverlay');
const statusEl = document.getElementById('demoStatus');
const strip = document.getElementById('thumbnailStrip');
const captureCanvas = document.createElement('canvas');
let clips = [];
let currentIndex = 0;
let autoRotate = true;
let activePrediction = null;
let streamInterval = null;
let sendingFrame = false;
let resetStreamOnNextFrame = true;
let activeTimeline = [];
const streamIntervalMs = 200;
const streamMaxDimension = 416;
const streamJpegQuality = 0.65;

function formatDays(value) {
    return value === undefined || value === null ? 'N/A' : `${Number(value).toFixed(1)} days`;
}

function updateResult(clip) {
    const prediction = clip.prediction || {};
    activePrediction = prediction;
    document.getElementById('clipTitle').textContent = `Demo Video ${currentIndex + 1}`;
    document.getElementById('demoDays').textContent = formatDays(prediction.days_until_molt);
    document.getElementById('demoPhase').textContent = prediction.phase || 'N/A';
    document.getElementById('demoRecommendation').textContent = prediction.recommendation || '';
    document.getElementById('demoModel').textContent = prediction.model_display_name || prediction.feature_type || 'Cached demo result';
    renderExplainability(prediction, 'demoExplainability');
    const thumb = document.getElementById('demoThumb');
    if (clip.thumbnail || prediction.thumbnail) {
        thumb.src = clip.thumbnail || prediction.thumbnail;
        thumb.style.display = 'block';
    } else {
        thumb.removeAttribute('src');
        thumb.style.display = 'none';
    }
    thumb.alt = `${clip.title || 'Demo clip'} cached MoltMeter detection`;
    Array.from(strip.children).forEach((button, index) => {
        button.classList.toggle('active', index === currentIndex);
    });
    document.getElementById('rotationMode').textContent = autoRotate ? 'Auto-rotating' : 'Repeating selected clip';
    drawOverlay();
}

function updateResultFromPrediction(prediction) {
    activePrediction = prediction || {};
    document.getElementById('demoDays').textContent = formatDays(activePrediction.days_until_molt);
    document.getElementById('demoPhase').textContent = activePrediction.phase || 'N/A';
    document.getElementById('demoRecommendation').textContent = activePrediction.recommendation || '';
    document.getElementById('demoModel').textContent = activePrediction.model_display_name || activePrediction.feature_type || 'Streaming video result';
    renderExplainability(activePrediction, 'demoExplainability');
    drawOverlay();
}

function renderExplainability(prediction, elementId) {
    const list = document.getElementById(elementId);
    if (!list) return;
    const reasons = [];
    if (prediction?.crop_used) {
        reasons.push('The estimate comes from a crab crop instead of the full frame, which usually gives the model a cleaner view.');
    } else if (prediction?.whole_image_fallback_used) {
        reasons.push('The detector did not find a clean crab crop, so the estimate came from the whole image.');
    }
    if (typeof prediction?.bbox_count === 'number' && prediction.bbox_count > 1) {
        reasons.push(`The frame had ${prediction.bbox_count} detections; MoltMeter used the strongest crab box.`);
    }
    if (prediction?.estimate_smoothed) {
        reasons.push('The displayed number is smoothed across nearby frames to reduce jitter.');
    }
    if (prediction?.bbox_stale) {
        reasons.push('The crab box was reused from nearby frames because the detector briefly dropped it.');
    }
    if (prediction?.estimate_stale) {
        reasons.push('The estimate was reused briefly while the app waited for a fresher frame.');
    }
    reasons.push('Look for the same visual cues in the field: a side split, pale halos around joints, and duller shell or limb opacity.');
    list.innerHTML = reasons.map((reason) => `<li>${reason}</li>`).join('');
}

function playClip(index, manual = false) {
    if (!clips.length) return;
    if (manual) autoRotate = false;
    currentIndex = index % clips.length;
    const clip = clips[currentIndex];
    resetStreamOnNextFrame = true;
    activePrediction = null;
    video.src = clip.video;
    video.load();
    updateResult(clip);
    loadTimeline(clip);
    if (statusEl) statusEl.textContent = `Playing ${currentIndex + 1} of ${clips.length}`;
    video.play().catch(() => {
        if (statusEl) statusEl.textContent = 'Tap video to start';
    });
}

async function loadTimeline(clip) {
    activeTimeline = [];
    if (!clip.timeline) return;
    try {
        const response = await fetch(clip.timeline, { cache: 'no-store' });
        if (!response.ok) throw new Error(`timeline ${response.status}`);
        const data = await response.json();
        activeTimeline = Array.isArray(data.samples) ? data.samples : [];
    } catch (error) {
        console.warn('Demo timeline unavailable, using live fallback', error);
        activeTimeline = [];
    }
}

function startVideoProcessing() {
    if (streamInterval) clearInterval(streamInterval);
    streamInterval = setInterval(captureAndSendVideoFrame, streamIntervalMs);
}

function stopVideoProcessing() {
    if (streamInterval) {
        clearInterval(streamInterval);
        streamInterval = null;
    }
    sendingFrame = false;
}

function drawOverlay() {
    if (!overlay || !video || !activePrediction) return;
    const rect = video.getBoundingClientRect();
    const displayWidth = Math.max(1, Math.round(rect.width));
    const displayHeight = Math.max(1, Math.round(rect.height));
    overlay.width = displayWidth;
    overlay.height = displayHeight;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const bbox = activePrediction.primary_bbox;
    if (!bbox || !activePrediction.image_width || !activePrediction.image_height) {
        return;
    }
    const scaleX = overlay.width / activePrediction.image_width;
    const scaleY = overlay.height / activePrediction.image_height;
    const x = bbox.xmin * scaleX;
    const y = bbox.ymin * scaleY;
    const w = (bbox.xmax - bbox.xmin) * scaleX;
    const h = (bbox.ymax - bbox.ymin) * scaleY;
    const color = activePrediction.color || '#173d22';
    const label = activePrediction.days_until_molt !== undefined
        ? `${Number(activePrediction.days_until_molt).toFixed(1)} days | ${activePrediction.phase || 'Molt estimate'}`
        : activePrediction.phase || 'Molt estimate';

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(4, Math.round(overlay.width / 180));
    ctx.strokeRect(x, y, w, h);

    const fontSize = Math.max(16, Math.round(overlay.width / 32));
    ctx.font = `900 ${fontSize}px system-ui, sans-serif`;
    const padding = Math.round(fontSize * 0.45);
    const labelWidth = Math.min(ctx.measureText(label).width + padding * 2, overlay.width - 12);
    const labelHeight = fontSize + padding * 1.4;
    const labelX = Math.min(Math.max(6, x), overlay.width - labelWidth - 6);
    const labelY = y - labelHeight - 6 > 0 ? y - labelHeight - 6 : y + 6;
    ctx.fillStyle = color;
    ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, labelX + padding, labelY + fontSize + padding * 0.15);
}

async function captureAndSendVideoFrame() {
    if (activeTimeline.length) return;
    if (!video || video.paused || video.ended || sendingFrame) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    sendingFrame = true;
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    const scale = Math.min(1, streamMaxDimension / Math.max(sourceWidth, sourceHeight));
    captureCanvas.width = Math.max(1, Math.round(sourceWidth * scale));
    captureCanvas.height = Math.max(1, Math.round(sourceHeight * scale));
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const started = performance.now();

    captureCanvas.toBlob(async (blob) => {
        if (!blob) {
            sendingFrame = false;
            return;
        }
        const formData = new FormData();
        formData.append('file', blob, 'demo_frame.jpg');
        if (resetStreamOnNextFrame) {
            formData.append('stream_reset', 'true');
            resetStreamOnNextFrame = false;
        }
        try {
            const response = await fetch('/predict_stream', { method: 'POST', body: formData });
            const data = await response.json();
            data.client_round_trip_ms = performance.now() - started;
            updateResultFromPrediction(data);
            if (statusEl) {
                statusEl.textContent = autoRotate
                    ? `Playing ${currentIndex + 1} of ${clips.length} - ${data.client_round_trip_ms.toFixed(0)} ms`
                    : `Repeating selected clip - ${data.client_round_trip_ms.toFixed(0)} ms`;
            }
        } catch (error) {
            console.warn('Demo stream frame failed', error);
            if (statusEl) statusEl.textContent = 'Video frame failed';
        } finally {
            sendingFrame = false;
        }
    }, 'image/jpeg', streamJpegQuality);
}

function predictionForCurrentTime() {
    if (!activeTimeline.length) return null;
    const t = video.currentTime || 0;
    const index = Math.min(activeTimeline.length - 1, Math.max(0, Math.round(t / (streamIntervalMs / 1000))));
    return activeTimeline[index]?.prediction || null;
}

function buildStrip() {
    strip.innerHTML = '';
    clips.forEach((clip, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        const img = document.createElement('img');
        img.alt = clip.title || `Clip ${index + 1}`;
        img.src = clip.thumbnail || '/static/GreenCrab.png';
        button.appendChild(img);
        button.addEventListener('click', () => playClip(index, true));
        strip.appendChild(button);
    });
}

async function loadDemo() {
    try {
        const response = await fetch('/static/demo_videos/manifest.json', { cache: 'no-store' });
        const data = await response.json();
        clips = data.clips || [];
        if (!clips.length) {
            if (statusEl) statusEl.textContent = 'No demo clips found';
            return;
        }
        buildStrip();
        playClip(0);
    } catch (error) {
        if (statusEl) statusEl.textContent = 'Demo manifest missing';
        console.warn(error);
    }
}

video.addEventListener('loadedmetadata', () => {
    drawOverlay();
    startVideoProcessing();
});
video.addEventListener('play', startVideoProcessing);
video.addEventListener('pause', stopVideoProcessing);
video.addEventListener('timeupdate', () => {
    const timelinePrediction = predictionForCurrentTime();
    if (timelinePrediction) {
        updateResultFromPrediction(timelinePrediction);
    } else {
        drawOverlay();
    }
});
window.addEventListener('resize', drawOverlay);
video.addEventListener('ended', () => {
    stopVideoProcessing();
    playClip(autoRotate ? currentIndex + 1 : currentIndex);
});
video.addEventListener('click', () => video.play());
document.getElementById('resumeRotationBtn').addEventListener('click', () => {
    autoRotate = true;
    playClip(currentIndex + 1);
});
loadDemo();
