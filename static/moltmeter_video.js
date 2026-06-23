const cameraPreview = document.getElementById('cameraPreview');
const overlayCanvas = document.getElementById('overlayCanvas');
const captureCanvas = document.getElementById('captureCanvas');
const startStreamBtn = document.getElementById('startStreamBtn');
const stopStreamBtn = document.getElementById('stopStreamBtn');
const streamStatus = document.getElementById('streamStatus');

let mediaStream = null;
let streamInterval = null;
let sendingFrame = false;

const streamIntervalMs = 200;
const streamMaxDimension = 416;
const streamJpegQuality = 0.65;

function formatDays(value) {
    return value === undefined || value === null ? 'N/A' : `${Number(value).toFixed(1)} days`;
}

function setStatus(text) {
    streamStatus.textContent = text;
}

function getPredictionBboxes(prediction) {
    if (Array.isArray(prediction.bboxes) && prediction.bboxes.length > 0) {
        return prediction.bboxes;
    }
    return prediction.primary_bbox ? [prediction.primary_bbox] : [];
}

function isPrimaryBbox(prediction, bbox) {
    const primary = prediction.primary_bbox;
    if (!primary || !bbox) return false;
    return ['xmin', 'ymin', 'xmax', 'ymax'].every(key => Math.abs(Number(primary[key]) - Number(bbox[key])) < 0.5);
}

function drawBboxLabel(ctx, text, x, y, color, fontSize, canvasWidth) {
    ctx.font = `900 ${fontSize}px system-ui, sans-serif`;
    const padding = Math.round(fontSize * 0.45);
    const width = Math.min(ctx.measureText(text).width + padding * 2, canvasWidth - 12);
    const height = fontSize + padding * 1.4;
    const labelX = Math.min(Math.max(6, x), canvasWidth - width - 6);
    const labelY = y - height - 6 > 0 ? y - height - 6 : y + 6;
    ctx.fillStyle = color;
    ctx.fillRect(labelX, labelY, width, height);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, labelX + padding, labelY + fontSize + padding * 0.15);
}

function drawDetection(prediction) {
    const rect = cameraPreview.getBoundingClientRect();
    overlayCanvas.width = Math.max(1, Math.round(rect.width));
    overlayCanvas.height = Math.max(1, Math.round(rect.height));
    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const bboxes = getPredictionBboxes(prediction);
    if (bboxes.length === 0 || !prediction.image_width || !prediction.image_height) {
        ctx.fillStyle = 'rgba(16, 33, 22, 0.82)';
        ctx.fillRect(12, 12, Math.min(340, overlayCanvas.width - 24), 70);
        ctx.fillStyle = '#ffffff';
        ctx.font = '900 20px system-ui, sans-serif';
        ctx.fillText('No crab detected', 24, 42);
        ctx.font = '600 14px system-ui, sans-serif';
        ctx.fillText('Point camera at a crab', 24, 64);
        return;
    }

    const scaleX = overlayCanvas.width / prediction.image_width;
    const scaleY = overlayCanvas.height / prediction.image_height;
    bboxes.forEach((bbox) => {
        const x = bbox.xmin * scaleX;
        const y = bbox.ymin * scaleY;
        const w = (bbox.xmax - bbox.xmin) * scaleX;
        const h = (bbox.ymax - bbox.ymin) * scaleY;
        const color = prediction.color || '#173d22';
        const label = prediction.days_until_molt !== undefined
            ? `${Number(prediction.days_until_molt).toFixed(1)} days | ${prediction.phase || 'Molt estimate'}`
            : prediction.phase || 'Molt estimate';
        ctx.strokeStyle = color;
        ctx.lineWidth = isPrimaryBbox(prediction, bbox) ? 6 : 4;
        ctx.strokeRect(x, y, w, h);
        drawBboxLabel(ctx, label, x, y, color, isPrimaryBbox(prediction, bbox) ? 24 : 19, overlayCanvas.width);
    });
}

function showResult(data) {
    document.getElementById('daysUntilMolt').textContent = formatDays(data.days_until_molt);
    document.getElementById('phase').textContent = data.phase || 'N/A';
    document.getElementById('recommendation').textContent = data.recommendation || '';
    document.getElementById('modelInfo').textContent = data.model_display_name || data.feature_type || '';
}

async function startCameraStream() {
    if (mediaStream) return;
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        cameraPreview.srcObject = mediaStream;
        await cameraPreview.play();
        startStreamBtn.disabled = true;
        stopStreamBtn.disabled = false;
        setStatus('Camera Active');
        streamInterval = setInterval(captureAndSendFrame, streamIntervalMs);
    } catch (error) {
        alert(`Unable to access camera: ${error.message}`);
    }
}

function stopCameraStream() {
    if (streamInterval) {
        clearInterval(streamInterval);
        streamInterval = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    sendingFrame = false;
    startStreamBtn.disabled = false;
    stopStreamBtn.disabled = true;
    setStatus('Camera Off');
    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

async function captureAndSendFrame() {
    if (!mediaStream || sendingFrame) return;
    if (cameraPreview.videoWidth === 0 || cameraPreview.videoHeight === 0) return;

    sendingFrame = true;
    const sourceWidth = cameraPreview.videoWidth;
    const sourceHeight = cameraPreview.videoHeight;
    const scale = Math.min(1, streamMaxDimension / Math.max(sourceWidth, sourceHeight));
    captureCanvas.width = Math.max(1, Math.round(sourceWidth * scale));
    captureCanvas.height = Math.max(1, Math.round(sourceHeight * scale));
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(cameraPreview, 0, 0, captureCanvas.width, captureCanvas.height);
    const started = performance.now();

    captureCanvas.toBlob(async (blob) => {
        if (!blob) {
            sendingFrame = false;
            return;
        }
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        try {
            const response = await fetch('/predict_stream', { method: 'POST', body: formData });
            const data = await response.json();
            data.client_round_trip_ms = performance.now() - started;
            drawDetection(data);
            showResult(data);
            setStatus(`Camera Active - ${data.client_round_trip_ms.toFixed(0)} ms`);
        } catch (error) {
            console.warn('Stream frame failed', error);
            setStatus('Camera Active - frame failed');
        } finally {
            sendingFrame = false;
        }
    }, 'image/jpeg', streamJpegQuality);
}

startStreamBtn.addEventListener('click', startCameraStream);
stopStreamBtn.addEventListener('click', stopCameraStream);
window.addEventListener('resize', () => {
    if (overlayCanvas.width && overlayCanvas.height) {
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
});
