// HeliosAI Frontend Engine - State & Controller
let state = {
  activeTab: 'presets',
  selectedPreset: null,
  selectedFile: null,
  viewMode: 'split',
  confidence: 0.40,
  minArea: 5,
  isProcessing: false,
  latestResult: null
};

// DOM Elements
const presetListEl = document.getElementById('preset-list');
const dropzoneEl = document.getElementById('dropzone');
const fileInputEl = document.getElementById('file-input');
const runBtnEl = document.getElementById('run-btn');
const btnSpinnerEl = document.getElementById('btn-spinner');
const btnTextEl = document.getElementById('btn-text');

const emptyStateEl = document.getElementById('empty-state');
const activeDisplayEl = document.getElementById('active-display');

const downloadImgBtn = document.getElementById('download-img-btn');
const exportGeojsonBtn = document.getElementById('export-geojson-btn');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  fetchPresets();
  setupDropzone();
  setupSplitSlider();
});

// 1. Fetch & Render Benchmark Presets
async function fetchPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    presetListEl.innerHTML = '';
    
    if (data.presets && data.presets.length > 0) {
      data.presets.forEach((preset, index) => {
        const card = document.createElement('div');
        card.className = `preset-card ${index === 0 ? 'selected' : ''}`;
        card.onclick = () => selectPreset(preset, card);
        card.innerHTML = `
          <img src="${preset.thumbnail}" alt="${preset.title}" class="preset-thumb">
          <div class="preset-info">
            <div class="preset-title">${preset.title}</div>
            <div class="preset-sub">${preset.subtitle}</div>
            <div class="preset-meta">${preset.resolution} · ${preset.location}</div>
          </div>
        `;
        presetListEl.appendChild(card);
        
        if (index === 0) {
          state.selectedPreset = preset.filename;
        }
      });
    }
  } catch (err) {
    console.error('Failed to load presets:', err);
    presetListEl.innerHTML = '<div style="padding: 1rem; color: #ef4444; font-size: 0.8rem;">Failed to load presets from server.</div>';
  }
}

function selectPreset(preset, cardEl) {
  document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));
  cardEl.classList.add('selected');
  state.selectedPreset = preset.filename;
  state.selectedFile = null;
}

// 2. Tab Switching
function switchInputTab(tab) {
  state.activeTab = tab;
  document.getElementById('tab-presets').classList.toggle('active', tab === 'presets');
  document.getElementById('tab-upload').classList.toggle('active', tab === 'upload');
  
  document.getElementById('view-presets').classList.toggle('hidden', tab !== 'presets');
  document.getElementById('view-upload').classList.toggle('hidden', tab !== 'upload');
}

// 3. Dropzone & Custom Upload Handling
function setupDropzone() {
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzoneEl.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzoneEl.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzoneEl.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzoneEl.classList.remove('dragover');
    }, false);
  });

  dropzoneEl.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });
}

function triggerFileInput() {
  fileInputEl.click();
}

function handleFileSelected(event) {
  if (event.target.files.length > 0) {
    handleFile(event.target.files[0]);
  }
}

function handleFile(file) {
  state.selectedFile = file;
  state.selectedPreset = null;
  
  const previewContainer = document.getElementById('upload-preview-container');
  const previewImg = document.getElementById('upload-preview-img');
  const fileNameEl = document.getElementById('upload-file-name');
  const fileSizeEl = document.getElementById('upload-file-size');
  
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewContainer.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

// 4. Threshold Controls
function updateConfValue(val) {
  state.confidence = parseFloat(val);
  document.getElementById('conf-val').textContent = `${Math.round(val * 100)}%`;
}

function updateAreaValue(val) {
  state.minArea = parseInt(val);
  document.getElementById('area-val').textContent = `${val} m²`;
}

// 5. Execute Model Inference
async function runInference() {
  if (state.isProcessing) return;
  
  const formData = new FormData();
  if (state.activeTab === 'upload' && state.selectedFile) {
    formData.append('file', state.selectedFile);
  } else if (state.selectedPreset) {
    formData.append('preset_filename', state.selectedPreset);
  } else {
    alert('Please select a benchmark preset or upload an image tile.');
    return;
  }
  
  formData.append('confidence_threshold', state.confidence);
  formData.append('min_area', state.minArea);
  
  setLoading(true);
  
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Inference execution failed');
    }
    
    const data = await res.json();
    state.latestResult = data;
    
    renderResults(data);
  } catch (err) {
    console.error('Inference error:', err);
    alert(`Inference failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  state.isProcessing = isLoading;
  btnSpinnerEl.classList.toggle('hidden', !isLoading);
  btnTextEl.textContent = isLoading ? 'Processing Aerial Tensor...' : 'Execute Model Inference';
  runBtnEl.disabled = isLoading;
}

// 6. Render Inference Results & Telemetry
function renderResults(data) {
  // Hide empty state, show active display
  emptyStateEl.classList.add('hidden');
  activeDisplayEl.classList.remove('hidden');
  
  // Enable download action buttons
  downloadImgBtn.disabled = false;
  exportGeojsonBtn.disabled = false;
  
  // 1. Update Images in All Views
  document.getElementById('split-img-before').src = data.images.original;
  document.getElementById('split-img-after').src = data.images.overlay;
  
  document.getElementById('single-overlay-img').src = data.images.overlay;
  document.getElementById('side-img-raw').src = data.images.original;
  document.getElementById('side-img-pred').src = data.images.overlay;
  
  document.getElementById('single-mask-img').src = data.images.mask;
  document.getElementById('single-heatmap-img').src = data.images.heatmap;
  
  // 2. Update Telemetry Summary Bar
  document.getElementById('stat-count').textContent = data.total_arrays;
  document.getElementById('stat-area').textContent = `${data.total_area_m2.toLocaleString()} m²`;
  document.getElementById('stat-kw').textContent = `${data.total_capacity_kw.toLocaleString()} kW`;
  document.getElementById('stat-kwh').textContent = `${data.annual_yield_kwh.toLocaleString()} kWh/yr`;
  document.getElementById('stat-co2').textContent = `${data.co2_saved_tons} Tons/yr`;
  document.getElementById('stat-latency').textContent = `${data.inference_time_ms} ms`;
  
  // 3. Update Detailed Arrays Table if present
  const tbody = document.getElementById('arrays-table-body');
  const badge = document.getElementById('table-badge');
  if (badge) badge.textContent = `${data.arrays.length} Features`;
  
  if (tbody) {
    if (data.arrays.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="table-empty-row">No solar arrays detected above current confidence threshold.</td></tr>';
    } else {
      tbody.innerHTML = data.arrays.map(arr => `
        <tr>
          <td><strong>#${arr.id}</strong></td>
          <td><span class="conf-badge">${arr.confidence}%</span></td>
          <td>${arr.area_m2} m²</td>
          <td>${arr.area_px.toLocaleString()} px</td>
          <td>${arr.est_kw} kW</td>
          <td>[${arr.bbox.join(', ')}]</td>
        </tr>
      `).join('');
    }
  }
}

// 7. View Mode Switcher
function setViewMode(mode) {
  state.viewMode = mode;
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
  });
  
  document.getElementById('view-mode-split').classList.toggle('hidden', mode !== 'split');
  document.getElementById('view-mode-overlay').classList.toggle('hidden', mode !== 'overlay');
  document.getElementById('view-mode-side').classList.toggle('hidden', mode !== 'side');
  document.getElementById('view-mode-mask').classList.toggle('hidden', mode !== 'mask');
  document.getElementById('view-mode-heatmap').classList.toggle('hidden', mode !== 'heatmap');
  
  // Show/hide wipe control strip
  const wipeBar = document.getElementById('slider-quick-bar');
  if (wipeBar) {
    wipeBar.style.display = (mode === 'split') ? 'flex' : 'none';
  }
}

// 8. Interactive Split Slider Controller
function setSplitPosition(percent) {
  percent = Math.max(0, Math.min(100, parseFloat(percent)));
  const container = document.getElementById('view-mode-split');
  const handle = document.getElementById('split-slider-handle');
  const rangeInput = document.getElementById('split-range-input');
  const wipeLabel = document.getElementById('wipe-percent-label');
  
  if (container) {
    container.style.setProperty('--slider-pos', `${percent}%`);
  }
  if (handle) {
    handle.style.left = `${percent}%`;
  }
  if (rangeInput && rangeInput.value !== String(percent)) {
    rangeInput.value = percent;
  }
  if (wipeLabel) {
    wipeLabel.textContent = `${Math.round(percent)}%`;
  }
  
  // Update quick pills active state
  document.querySelectorAll('.pill-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  if (percent === 0 && document.querySelectorAll('.pill-btn')[0]) {
    document.querySelectorAll('.pill-btn')[0].classList.add('active');
  } else if (percent === 50 && document.querySelectorAll('.pill-btn')[1]) {
    document.querySelectorAll('.pill-btn')[1].classList.add('active');
  } else if (percent === 100 && document.querySelectorAll('.pill-btn')[2]) {
    document.querySelectorAll('.pill-btn')[2].classList.add('active');
  }
}

function setupSplitSlider() {
  const container = document.getElementById('view-mode-split');
  if (!container) return;
  
  let isDragging = false;
  
  function updateFromPointer(clientX) {
    const rect = container.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const percent = (x / rect.width) * 100;
    setSplitPosition(percent);
  }
  
  container.addEventListener('pointerdown', (e) => {
    isDragging = true;
    container.setPointerCapture(e.pointerId);
    updateFromPointer(e.clientX);
  });
  
  container.addEventListener('pointermove', (e) => {
    if (!isDragging) return;
    updateFromPointer(e.clientX);
  });
  
  container.addEventListener('pointerup', (e) => {
    if (isDragging) {
      container.releasePointerCapture(e.pointerId);
      isDragging = false;
    }
  });
  
  container.addEventListener('pointercancel', () => {
    isDragging = false;
  });
}

// 9. Export & Download Handlers
function downloadResultImage() {
  if (!state.latestResult) return;
  const link = document.createElement('a');
  link.href = state.latestResult.images.overlay;
  link.download = `solar_detection_overlay_${Date.now()}.jpg`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function downloadGeoJSON() {
  window.open('/api/export_geojson', '_blank');
}
