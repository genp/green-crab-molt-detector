const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewWrap = document.getElementById('previewWrap');
const previewImg = document.getElementById('previewImg');
const statusEl = document.getElementById('status');
const resultPanel = document.getElementById('resultPanel');
const newImageBtn = document.getElementById('newImageBtn');

function setStatus(text) {
    statusEl.textContent = text;
}

function formatDays(value) {
    return value === undefined || value === null ? 'N/A' : `${Number(value).toFixed(1)} days`;
}

function showResult(data) {
    resultPanel.style.display = 'block';
    document.getElementById('daysUntilMolt').textContent = formatDays(data.days_until_molt);
    document.getElementById('phase').textContent = data.phase || 'N/A';
    document.getElementById('recommendation').textContent = data.recommendation || '';
    document.getElementById('modelInfo').textContent = data.model_display_name || data.feature_type || '';
    renderExplainability(data, 'imageExplainability');
    const thumb = document.getElementById('thumbnail');
    if (data.thumbnail) {
        thumb.src = data.thumbnail;
        thumb.style.display = 'block';
    } else {
        thumb.removeAttribute('src');
        thumb.style.display = 'none';
    }
}

function renderExplainability(prediction, elementId) {
    const list = document.getElementById(elementId);
    if (!list) return;
    const reasons = [];
    if (prediction?.crop_used) {
        reasons.push('The estimate comes from a crab crop instead of the whole frame, so the model can focus on the shell and joints.');
    } else if (prediction?.whole_image_fallback_used) {
        reasons.push('The detector did not find a clean crab crop, so the estimate came from the whole image.');
    }
    if (typeof prediction?.bbox_count === 'number' && prediction.bbox_count > 1) {
        reasons.push(`The frame had ${prediction.bbox_count} detections; MoltMeter selected the strongest crab.`);
    }
    if (prediction?.estimate_smoothed) {
        reasons.push('The displayed number is smoothed to reduce frame-to-frame jitter.');
    }
    if (prediction?.bbox_stale) {
        reasons.push('The crab box was reused from nearby frames because the detector briefly lost it.');
    }
    if (prediction?.estimate_stale) {
        reasons.push('The estimate was reused briefly while the app waited for a fresher crop.');
    }
    reasons.push('Training cue reminder: side split, halo at joints, and duller shell or limb opacity.');
    list.innerHTML = reasons.map((reason) => `<li>${reason}</li>`).join('');
}

async function analyzeFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please choose an image file.');
        return;
    }
    const reader = new FileReader();
    reader.onload = event => {
        previewImg.src = event.target.result;
        uploadZone.style.display = 'none';
        previewWrap.style.display = 'block';
    };
    reader.readAsDataURL(file);

    setStatus('Analyzing');
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Prediction failed');
        }
        setStatus('Complete');
        showResult(data);
    } catch (error) {
        setStatus('Error');
        alert(`Prediction failed: ${error.message}`);
    }
}

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', event => {
    event.preventDefault();
    uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', event => {
    event.preventDefault();
    uploadZone.classList.remove('dragover');
    analyzeFile(event.dataTransfer.files[0]);
});
fileInput.addEventListener('change', event => analyzeFile(event.target.files[0]));
newImageBtn.addEventListener('click', () => fileInput.click());
