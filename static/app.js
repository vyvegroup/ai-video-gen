/* ============================================================
   AI Video Gen — app.js
   Vanilla JavaScript — no dependencies
   ============================================================ */

(function () {
  'use strict';

  // ============================
  // CONFIGURATION
  // ============================
  const CONFIG = {
    get serverUrl() {
      return localStorage.getItem('serverUrl') || window.location.origin;
    },
    set serverUrl(v) {
      localStorage.setItem('serverUrl', v.replace(/\/+$/, ''));
    },
    get allowNsfw() {
      return localStorage.getItem('allowNsfw') === 'true';
    },
    set allowNsfw(v) {
      localStorage.setItem('allowNsfw', String(v));
    },
    get defaultFrames() {
      return parseInt(localStorage.getItem('defaultFrames') || '14', 10);
    },
    set defaultFrames(v) {
      localStorage.setItem('defaultFrames', String(v));
    },
    get defaultSteps() {
      return parseInt(localStorage.getItem('defaultSteps') || '25', 10);
    },
    set defaultSteps(v) {
      localStorage.setItem('defaultSteps', String(v));
    },
    get defaultFps() {
      return parseInt(localStorage.getItem('defaultFps') || '6', 10);
    },
    set defaultFps(v) {
      localStorage.setItem('defaultFps', String(v));
    },
  };

  // ============================
  // STATE
  // ============================
  const state = {
    currentTab: 'create',
    genMode: 'image',         // 'image' | 'text'
    downloadSource: 'huggingface',
    loadedModel: null,
    models: [],
    uploadedImagePath: null,
    generating: false,
    currentVideoId: null,
    wsConnection: null,
    pollingInterval: null,
    elapsedInterval: null,
    startTime: null,
    deleteCallback: null,
    galleryItems: [],
    // Chat state
    chatSessionId: null,
    chatMessages: [],
    chatOpen: false,
    chatSending: false,
    chatCharacter: { name: 'AI Assistant', age: null, personality: '', appearance: '', scenario: '', nsfw: false },
    chatUnreadCount: 0,
    // Persistent generation state
    activeGenerations: {},
  };

  // ============================
  // UTILITY HELPERS
  // ============================
  function api(path) {
    return `${CONFIG.serverUrl}${path}`;
  }

  async function apiFetch(path, options = {}) {
    const url = api(path);
    try {
      const resp = await fetch(url, {
        ...options,
        headers: {
          'ngrok-skip-browser-warning': '1',
          ...(options.headers || {}),
        },
      });
      // Check if ngrok returned HTML warning page instead of JSON
      const contentType = resp.headers.get('content-type') || '';
      if (contentType.includes('text/html') && !path.endsWith('.html') && path.startsWith('/api/')) {
        throw new Error('ngrok: Click "Visit Site" first, or the tunnel is disconnected');
      }
      if (!resp.ok) {
        const errorBody = await resp.text();
        let msg = `HTTP ${resp.status}`;
        try { msg = JSON.parse(errorBody).detail || msg; } catch (_) { /* ignore */ }
        throw new Error(msg);
      }
      if (resp.status === 204) return null;
      return resp.json();
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        throw new Error(`Cannot connect to server at ${CONFIG.serverUrl}`);
      }
      throw err;
    }
  }

  function debounce(fn, ms = 300) {
    let timer;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function formatDuration(ms) {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  }

  function $(sel, ctx = document) {
    return ctx.querySelector(sel);
  }

  function $$(sel, ctx = document) {
    return Array.from(ctx.querySelectorAll(sel));
  }

  function el(tag, attrs = {}, children = []) {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'className') e.className = v;
      else if (k === 'textContent') e.textContent = v;
      else if (k === 'innerHTML') e.innerHTML = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'dataset') Object.assign(e.dataset, v);
      else e.setAttribute(k, v);
    });
    children.forEach(c => {
      if (typeof c === 'string') e.appendChild(document.createTextNode(c));
      else if (c) e.appendChild(c);
    });
    return e;
  }

  // ============================
  // TOAST SYSTEM
  // ============================
  function showToast(type, title, message, duration = 5000) {
    const container = $('#toastContainer');
    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    const colors = { success: 'var(--green)', error: 'var(--red)', info: 'var(--blue)', warning: 'var(--yellow)' };

    const toast = el('div', { className: `toast toast-${type}` }, [
      el('span', { className: 'toast-icon', textContent: icons[type] || 'ℹ️' }),
      el('div', { className: 'toast-content' }, [
        el('div', { className: 'toast-title', textContent: title }),
        el('div', { className: 'toast-message', textContent: message }),
      ]),
      el('button', {
        className: 'toast-close',
        textContent: '✕',
        onClick: () => removeToast(toast),
      }),
      el('div', {
        className: 'toast-progress',
        style: `background: ${colors[type] || colors.info}`,
      }),
    ]);

    container.appendChild(toast);

    const timer = setTimeout(() => removeToast(toast), duration);
    toast._timer = timer;
  }

  function removeToast(toast) {
    if (!toast || toast._removing) return;
    toast._removing = true;
    clearTimeout(toast._timer);
    toast.classList.add('removing');
    toast.addEventListener('animationend', () => toast.remove());
  }

  // ============================
  // TAB NAVIGATION
  // ============================
  function initTabs() {
    $$('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        switchTab(tab);
      });
    });
  }

  function switchTab(tabName) {
    state.currentTab = tabName;
    $$('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
    $$('.tab-content').forEach(c => c.classList.toggle('active', c.id === `tab-${tabName}`));

    if (tabName === 'models') {
      loadModels();
      loadSuggestedModels();
    }
    if (tabName === 'gallery') {
      loadGallery();
    }
  }

  // ============================
  // GENERATION MODE TOGGLE
  // ============================
  function initGenModeToggle() {
    const toggleBtns = $$('#tab-create .toggle-group:not(.mb-3) .toggle-btn');
    // Actually we need to be more specific — the mode toggle group is the second one in create tab
    const modeToggleCard = $('#imageUploadCard').previousElementSibling;
    const modeBtns = modeToggleCard ? $$('.toggle-btn', modeToggleCard) : [];

    // Better: target by parent that contains data-mode
    $$('#tab-create [data-mode]').forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        state.genMode = mode;
        $$('#tab-create [data-mode]').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
        updateModeUI();
      });
    });
  }

  function updateModeUI() {
    const isImage = state.genMode === 'image';
    $('#imageUploadCard').classList.toggle('hidden', !isImage);
    $('#textPromptCard').classList.toggle('hidden', isImage);

    // Toggle SVD / Text2Video specific settings
    $('#motionBucketItem').classList.toggle('hidden', !isImage);
    $('#noiseAugItem').classList.toggle('hidden', !isImage);
    $('#cfgScaleItem').classList.toggle('hidden', isImage);
  }

  // ============================
  // SOURCE TOGGLE (Models tab)
  // ============================
  function initSourceToggle() {
    $$('#tab-models [data-source]').forEach(btn => {
      btn.addEventListener('click', () => {
        const source = btn.dataset.source;
        state.downloadSource = source;
        $$('#tab-models [data-source]').forEach(b => b.classList.toggle('active', b.dataset.source === source));
        updateSourceUI();
      });
    });
  }

  function updateSourceUI() {
    const isHF = state.downloadSource === 'huggingface';
    const input = $('#modelUrlInput');
    const hint = $('#modelUrlHint');
    if (isHF) {
      input.placeholder = 'stabilityai/stable-video-diffusion-img2vid-xt';
      hint.textContent = 'Enter HuggingFace model ID (e.g. stabilityai/stable-video-diffusion-img2vid-xt)';
    } else {
      input.placeholder = 'https://civarchive.com/api/download/models/...';
      hint.textContent = 'Enter a direct download URL for the model';
    }
  }

  // ============================
  // COLLAPSIBLES
  // ============================
  function initCollapsibles() {
    $$('.collapsible-header').forEach(header => {
      header.addEventListener('click', () => {
        const body = header.nextElementSibling;
        if (!body || !body.classList.contains('collapsible-body')) return;
        const isOpen = !body.classList.contains('hidden');
        body.classList.toggle('hidden', isOpen);
        header.classList.toggle('open', !isOpen);
      });
    });
  }

  // ============================
  // IMAGE UPLOAD
  // ============================
  function initImageUpload() {
    const zone = $('#uploadZone');
    const input = $('#imageInput');
    const placeholder = $('#uploadPlaceholder');
    const preview = $('#uploadPreview');
    const previewImg = $('#previewImage');
    const removeBtn = $('#removeImageBtn');

    // Click to open
    zone.addEventListener('click', (e) => {
      if (e.target.closest('#removeImageBtn')) return;
      input.click();
    });

    // File selected
    input.addEventListener('change', () => {
      if (input.files && input.files[0]) {
        handleImageFile(input.files[0]);
      }
    });

    // Drag & drop
    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
      zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      if (files && files[0] && files[0].type.startsWith('image/')) {
        handleImageFile(files[0]);
      } else {
        showToast('warning', 'Invalid file', 'Please drop an image file (PNG, JPG, WEBP)');
      }
    });

    // Remove
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      state.uploadedImagePath = null;
      input.value = '';
      placeholder.classList.remove('hidden');
      preview.classList.add('hidden');
      previewImg.src = '';
    });
  }

  async function handleImageFile(file) {
    const placeholder = $('#uploadPlaceholder');
    const preview = $('#uploadPreview');
    const previewImg = $('#previewImage');

    // Show preview immediately
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      placeholder.classList.add('hidden');
      preview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Upload to server
    try {
      showToast('info', 'Uploading', `Uploading ${file.name}...`);
      const formData = new FormData();
      formData.append('file', file);
      const result = await apiFetch('/api/upload/image', {
        method: 'POST',
        body: formData,
      });
      state.uploadedImagePath = result.path || result.file_path || result.filename;
      showToast('success', 'Uploaded', 'Image uploaded successfully');
    } catch (err) {
      showToast('error', 'Upload failed', err.message);
      state.uploadedImagePath = null;
      placeholder.classList.remove('hidden');
      preview.classList.add('hidden');
    }
  }

  // ============================
  // SLIDERS & INPUTS
  // ============================
  function initSliders() {
    const sliders = [
      { id: 'framesSlider', display: 'framesValue' },
      { id: 'stepsSlider', display: 'stepsValue' },
      { id: 'fpsSlider', display: 'fpsValue' },
      { id: 'motionBucketSlider', display: 'motionBucketValue' },
      { id: 'noiseAugSlider', display: 'noiseAugValue' },
      { id: 'cfgScaleSlider', display: 'cfgScaleValue' },
    ];

    sliders.forEach(({ id, display }) => {
      const slider = $(`#${id}`);
      const valueEl = $(`#${display}`);
      if (!slider || !valueEl) return;

      slider.addEventListener('input', () => {
        valueEl.textContent = slider.value;
      });
    });
  }

  function initSeedButton() {
    const randomBtn = $('#randomSeedBtn');
    const seedInput = $('#seedInput');
    if (randomBtn && seedInput) {
      randomBtn.addEventListener('click', () => {
        seedInput.value = Math.floor(Math.random() * 2147483647);
      });
    }
  }

  function initCharCount() {
    const textarea = $('#promptInput');
    const counter = $('#charCount');
    if (textarea && counter) {
      textarea.addEventListener('input', () => {
        counter.textContent = textarea.value.length;
      });
    }
  }

  // ============================
  // MODEL MANAGEMENT
  // ============================
  async function loadModels() {
    const select = $('#modelSelect');
    const listContainer = $('#modelList');

    try {
      const models = await apiFetch('/api/models');
      state.models = Array.isArray(models) ? models : (models.models || []);

      // Populate select
      select.innerHTML = '';
      if (state.models.length === 0) {
        select.innerHTML = '<option value="">No models installed</option>';
      } else {
        select.innerHTML = '<option value="">Select a model...</option>';
        state.models.forEach(m => {
          const name = m.name || m.id || m;
          const opt = el('option', { value: name, textContent: name });
          if (m.loaded || state.loadedModel === name) {
            opt.selected = true;
            state.loadedModel = name;
          }
          select.appendChild(opt);
        });
      }

      // Populate model list
      renderModelList();

    } catch (err) {
      select.innerHTML = '<option value="">Error loading models</option>';
      listContainer.innerHTML = `<div class="empty-state"><div class="empty-state-icon">⚠️</div><h3>Connection Error</h3><p>${err.message}</p></div>`;
    }
  }

  function renderModelList() {
    const container = $('#modelList');
    if (!state.models || state.models.length === 0) {
      container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📦</div><h3>No models installed</h3><p>Download a model to get started</p></div>`;
      return;
    }

    container.innerHTML = '';
    state.models.forEach(m => {
      const name = m.name || m.id || m;
      const type = m.type || 'unknown';
      const size = m.size_mb ? `${Number(m.size_mb).toFixed(1)} MB` : (m.size ? formatBytes(m.size) : 'Unknown');
      const source = m.source || 'huggingface';
      const loaded = m.loaded || state.loadedModel === name;

      const card = el('div', { className: `model-card ${loaded ? 'loaded' : ''}` });
      card.innerHTML = `
        <div class="model-card-header">
          <div class="model-name">${escapeHtml(name)}</div>
          <span class="model-badge ${loaded ? 'loaded-badge' : 'unloaded-badge'}">${loaded ? '⚡ Loaded' : 'Idle'}</span>
        </div>
        <div class="model-meta">
          <span>🏷️ ${escapeHtml(type)}</span>
          <span>💾 ${size}</span>
          <span>🔗 ${escapeHtml(source)}</span>
        </div>
        <div class="model-actions">
          ${loaded
            ? `<button class="btn btn-sm btn-outline" data-action="unload" data-name="${escapeHtml(name)}">⏹️ Unload</button>`
            : `<button class="btn btn-sm btn-success" data-action="load" data-name="${escapeHtml(name)}">⚡ Load</button>`
          }
          <button class="btn btn-sm btn-danger" data-action="delete" data-name="${escapeHtml(name)}">🗑️</button>
        </div>
      `;
      container.appendChild(card);
    });

    // Bind model actions
    container.querySelectorAll('[data-action]').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        const name = btn.dataset.name;
        if (action === 'load') loadModel(name);
        else if (action === 'unload') unloadModel(name);
        else if (action === 'delete') confirmDeleteModel(name);
      });
    });
  }

  async function loadModel(name) {
    try {
      showToast('info', 'Loading model', `Loading ${name}...`);
      await apiFetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `model_name=${encodeURIComponent(name)}&allow_nsfw=${CONFIG.allowNsfw}`,
      });
      state.loadedModel = name;
      showToast('success', 'Model loaded', `${name} is ready`);
      loadModels();
    } catch (err) {
      showToast('error', 'Load failed', err.message);
    }
  }

  async function unloadModel(name) {
    try {
      showToast('info', 'Unloading', `Unloading ${name}...`);
      await apiFetch('/api/models/unload', { method: 'POST' });
      state.loadedModel = null;
      showToast('success', 'Unloaded', `${name} has been unloaded`);
      loadModels();
    } catch (err) {
      showToast('error', 'Unload failed', err.message);
    }
  }

  function confirmDeleteModel(name) {
    state.deleteCallback = async () => {
      try {
        showToast('info', 'Deleting', `Deleting ${name}...`);
        await apiFetch(`/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (state.loadedModel === name) state.loadedModel = null;
        showToast('success', 'Deleted', `${name} has been removed`);
        loadModels();
      } catch (err) {
        showToast('error', 'Delete failed', err.message);
      }
    };
    $('#deleteMessage').textContent = `Are you sure you want to delete "${name}"? This cannot be undone.`;
    $('#deleteModal').classList.remove('hidden');
  }

  // ============================
  // SUGGESTED MODELS
  // ============================
  async function loadSuggestedModels() {
    const container = $('#suggestedModels');
    try {
      const data = await apiFetch('/api/models/registry');
      const models = Array.isArray(data) ? data : (data.models || data.suggestions || []);
      if (models.length === 0) {
        container.innerHTML = `<div class="empty-state"><p>No suggested models available</p></div>`;
        return;
      }
      container.innerHTML = '';
      models.forEach(m => {
        const name = m.name || m.id || m;
        const desc = m.description || m.desc || '';
        const source = m.source || 'huggingface';
        const card = el('div', { className: 'suggested-card' }, [
          el('div', { className: 'suggested-info' }, [
            el('div', { className: 'suggested-name', textContent: name }),
            el('div', { className: 'suggested-desc', textContent: desc }),
          ]),
          el('button', {
            className: 'suggested-btn',
            textContent: '📥 Add',
            onClick: () => downloadModel(name, source),
          }),
        ]);
        container.appendChild(card);
      });
    } catch (err) {
      container.innerHTML = `<div class="empty-state"><p>Could not load suggestions</p></div>`;
    }
  }

  // ============================
  // MODEL DOWNLOAD
  // ============================
  async function downloadModel(modelId, source) {
    const progressDiv = $('#downloadProgress');
    const progressBar = $('#downloadProgressBar');
    const progressText = $('#downloadProgressText');
    const btn = $('#downloadModelBtn');

    if (!modelId) {
      modelId = $('#modelUrlInput').value.trim();
      source = state.downloadSource;
    }

    if (!modelId) {
      showToast('warning', 'Missing input', 'Please enter a model ID or URL');
      return;
    }

    try {
      btn.disabled = true;
      progressDiv.classList.remove('hidden');
      progressBar.style.width = '0%';
      progressText.textContent = 'Starting download...';
      showToast('info', 'Downloading', `Starting download of ${modelId}`);

      const formData = new URLSearchParams();
      formData.append('model_id', modelId);
      formData.append('source', source);
      const hfToken = $('#hfTokenInput').value.trim();
      if (hfToken) formData.append('hf_token', hfToken);

      // Simulate progress (the API may not stream progress for downloads)
      let progress = 0;
      const fakeProgress = setInterval(() => {
        progress = Math.min(progress + Math.random() * 3, 90);
        progressBar.style.width = progress + '%';
        progressText.textContent = `Downloading... ${Math.round(progress)}%`;
      }, 800);

      await apiFetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData.toString(),
      });

      clearInterval(fakeProgress);
      progressBar.style.width = '100%';
      progressText.textContent = 'Download complete!';
      showToast('success', 'Download complete', `${modelId} has been downloaded`);

      setTimeout(() => {
        progressDiv.classList.add('hidden');
        loadModels();
      }, 2000);

    } catch (err) {
      showToast('error', 'Download failed', err.message);
      progressText.textContent = 'Download failed';
    } finally {
      btn.disabled = false;
    }
  }

  // ============================
  // GENERATION
  // ============================
  function getGenerateParams() {
    return {
      model_name: $('#modelSelect').value,
      prompt: $('#promptInput').value,
      negative_prompt: $('#negativePrompt').value,
      image_path: state.uploadedImagePath || '',
      num_frames: parseInt($('#framesSlider').value, 10),
      num_inference_steps: parseInt($('#stepsSlider').value, 10),
      fps: parseInt($('#fpsSlider').value, 10),
      seed: parseInt($('#seedInput').value, 10),
      motion_bucket_id: parseInt($('#motionBucketSlider').value, 10),
      noise_aug_strength: parseFloat($('#noiseAugSlider').value),
      cfg_scale: parseFloat($('#cfgScaleSlider').value),
      width: parseInt($('#widthInput').value, 10),
      height: parseInt($('#heightInput').value, 10),
      allow_nsfw: $('#nsfwToggle').checked || CONFIG.allowNsfw,
    };
  }

  function validateGenerateParams(params) {
    if (!params.model_name) {
      showToast('warning', 'No model selected', 'Please select a model before generating');
      return false;
    }
    if (state.genMode === 'image' && !params.image_path) {
      showToast('warning', 'No image', 'Please upload a source image');
      return false;
    }
    if (state.genMode === 'text' && !params.prompt.trim()) {
      showToast('warning', 'No prompt', 'Please enter a text prompt');
      return false;
    }
    return true;
  }

  async function startGeneration() {
    if (state.generating) return;

    const params = getGenerateParams();
    if (!validateGenerateParams(params)) return;

    const btn = $('#generateBtn');
    const progressPanel = $('#progressPanel');
    const resultPanel = $('#resultPanel');

    try {
      state.generating = true;
      btn.disabled = true;
      resultPanel.classList.add('hidden');
      progressPanel.classList.remove('hidden');

      // Reset progress
      updateProgress(0, 'Sending request...');
      state.startTime = Date.now();
      state.elapsedInterval = setInterval(updateElapsedTime, 1000);

      // Build form
      const formData = new URLSearchParams();
      Object.entries(params).forEach(([k, v]) => {
        if (v !== undefined && v !== null && v !== '') {
          formData.append(k, v);
        }
      });

      const result = await apiFetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData.toString(),
      });

      const videoId = result.video_id || result.id || result.task_id;
      if (!videoId) {
        throw new Error('Server did not return a video ID');
      }

      state.currentVideoId = videoId;
      showToast('info', 'Generation started', `Video ID: ${videoId}`);

      // Try WebSocket first, fall back to polling
      connectWebSocket(videoId);
      startPolling(videoId);

    } catch (err) {
      showToast('error', 'Generation failed', err.message);
      stopGeneration();
    }
  }

  function connectWebSocket(videoId) {
    if (state.wsConnection) {
      state.wsConnection.close();
    }

    try {
      const wsUrl = CONFIG.serverUrl.replace(/^http/, 'ws') + `/ws/generate/${videoId}`;
      const ws = new WebSocket(wsUrl);
      state.wsConnection = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.progress !== undefined) {
            const stepText = data.message || data.step || data.current_step || '';
            updateProgress(data.progress, stepText);
          }
          if (data.status === 'completed') {
            onGenerationComplete(data, videoId);
          }
          if (data.status === 'failed' || data.status === 'error') {
            onGenerationFailed(data.message || data.error || 'Generation failed');
          }
        } catch (_) { /* ignore parse errors */ }
      };

      ws.onopen = () => {
        console.log('WebSocket connected for progress');
      };

      ws.onerror = () => {
        console.log('WebSocket error, relying on polling');
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
      };
    } catch (err) {
      console.log('WebSocket not available, using polling only');
    }
  }

  function startPolling(videoId) {
    if (state.pollingInterval) clearInterval(state.pollingInterval);

    const poll = async () => {
      try {
        const data = await apiFetch(`/api/generate/${videoId}`);
        if (data.progress !== undefined) {
          const stepText = data.message || data.step || data.current_step || '';
          updateProgress(data.progress, stepText);
        }
        if (data.status === 'completed') {
          onGenerationComplete(data, videoId);
        }
        if (data.status === 'failed' || data.status === 'error') {
          onGenerationFailed(data.message || data.error || 'Generation failed');
        }
      } catch (err) {
        // Don't stop polling on temporary errors
        console.warn('Poll error:', err.message);
      }
    };

    // Poll every 2 seconds
    state.pollingInterval = setInterval(poll, 2000);
    // Also poll immediately
    poll();
  }

  function updateProgress(percent, stepText) {
    const clampedPercent = Math.max(0, Math.min(100, Math.round(percent)));
    $('#progressBar').style.width = clampedPercent + '%';
    $('#progressPercent').textContent = clampedPercent + '%';
    if (stepText) {
      $('#progressStep').textContent = stepText;
    }
  }

  function updateElapsedTime() {
    if (!state.startTime) return;
    const elapsed = Date.now() - state.startTime;
    $('#elapsedTime').textContent = formatDuration(elapsed);
  }

  function onGenerationComplete(data, videoId) {
    stopTracking();
    const elapsed = state.startTime ? formatDuration(Date.now() - state.startTime) : '--';
    state.generating = false;

    $('#generateBtn').disabled = false;
    $('#progressPanel').classList.add('hidden');

    const resultPanel = $('#resultPanel');
    resultPanel.classList.remove('hidden');

    // Set video source
    const videoUrl = data.video_url || data.output_url || `${api('/api/outputs')}/${videoId}/video`;
    const gifUrl = data.gif_url || `${api('/api/outputs')}/${videoId}/gif`;
    const resultVideo = $('#resultVideo');
    resultVideo.src = videoUrl;
    resultVideo.load();
    resultVideo.play().catch(() => { /* autoplay may be blocked */ });

    // Set info
    $('#resultTime').textContent = elapsed;
    $('#resultFrames').textContent = data.frames || data.num_frames || '--';
    $('#resultSeed').textContent = data.seed || '--';

    // Download buttons
    $('#downloadMp4Btn').onclick = () => window.open(videoUrl, '_blank');
    $('#downloadGifBtn').onclick = () => window.open(gifUrl, '_blank');

    showToast('success', 'Generation complete', 'Your video is ready! ✨');
  }

  function onGenerationFailed(errorMsg) {
    stopTracking();
    state.generating = false;
    $('#generateBtn').disabled = false;
    $('#progressPanel').classList.add('hidden');
    showToast('error', 'Generation failed', errorMsg);
  }

  function cancelGeneration() {
    stopTracking();
    state.generating = false;
    state.currentVideoId = null;
    $('#generateBtn').disabled = false;
    $('#progressPanel').classList.add('hidden');
    showToast('warning', 'Cancelled', 'Generation has been cancelled');
  }

  function stopGeneration() {
    stopTracking();
    state.generating = false;
    $('#generateBtn').disabled = false;
  }

  function stopTracking() {
    if (state.wsConnection) {
      try { state.wsConnection.close(); } catch (_) {}
      state.wsConnection = null;
    }
    if (state.pollingInterval) {
      clearInterval(state.pollingInterval);
      state.pollingInterval = null;
    }
    if (state.elapsedInterval) {
      clearInterval(state.elapsedInterval);
      state.elapsedInterval = null;
    }
    state.startTime = null;
  }

  // ============================
  // GALLERY
  // ============================
  async function loadGallery() {
    const grid = $('#galleryGrid');
    const emptyState = $('#galleryEmpty');

    try {
      const data = await apiFetch('/api/outputs');
      const outputs = Array.isArray(data) ? data : (data.outputs || data.videos || []);

      // Clear existing gallery items (keep empty state)
      const existingCards = grid.querySelectorAll('.gallery-card');
      existingCards.forEach(c => c.remove());

      if (outputs.length === 0) {
        emptyState.classList.remove('hidden');
        return;
      }

      emptyState.classList.add('hidden');
      state.galleryItems = outputs;

      outputs.forEach((item, i) => {
        const id = item.video_id || item.id || i;
        const videoUrl = item.video_url || item.url || `${api('/api/outputs')}/${id}/video`;
        const gifUrl = item.gif_url || `${api('/api/outputs')}/${id}/gif`;
        const prompt = item.prompt || '';
        const genTime = item.generation_time ? formatDuration(item.generation_time * 1000) : '';
        const frames = item.frames || item.num_frames || '';
        const fps = item.fps || '';
        const seed = item.seed || '';

        const card = el('div', {
          className: 'gallery-card',
          style: `animation-delay: ${i * 0.05}s`,
        });
        card.innerHTML = `
          <div class="gallery-thumb">
            <video src="${escapeHtml(videoUrl)}" muted loop preload="metadata" playsinline></video>
            <div class="play-overlay"><span>▶️</span></div>
          </div>
          <div class="gallery-info">
            <div class="gallery-prompt">${escapeHtml(prompt) || '<em class="text-muted">No prompt</em>'}</div>
            <div class="gallery-meta">
              ${genTime ? `<span>⏱️ ${genTime}</span>` : ''}
              ${frames ? `<span>🎞️ ${frames}f</span>` : ''}
              ${fps ? `<span>📹 ${fps}fps</span>` : ''}
              ${seed ? `<span>🎲 ${seed}</span>` : ''}
            </div>
            <div class="gallery-actions">
              <button class="btn btn-sm btn-outline" data-action="view" data-url="${escapeHtml(videoUrl)}">👁️ View</button>
              <button class="btn btn-sm btn-outline" data-action="download" data-url="${escapeHtml(videoUrl)}">📥 MP4</button>
              <button class="btn btn-sm btn-danger" data-action="delete-gallery" data-id="${escapeHtml(String(id))}">🗑️</button>
            </div>
          </div>
        `;

        // Hover to play
        const video = card.querySelector('video');
        card.addEventListener('mouseenter', () => {
          video.play().catch(() => {});
        });
        card.addEventListener('mouseleave', () => {
          video.pause();
          video.currentTime = 0;
        });

        grid.appendChild(card);
      });

      // Bind actions
      grid.querySelectorAll('[data-action]').forEach(btn => {
        btn.addEventListener('click', () => {
          const action = btn.dataset.action;
          if (action === 'view') {
            window.open(btn.dataset.url, '_blank');
          } else if (action === 'download') {
            window.open(btn.dataset.url, '_blank');
          } else if (action === 'delete-gallery') {
            deleteGalleryItem(btn.dataset.id);
          }
        });
      });

      // Setup lazy loading with Intersection Observer
      initGalleryLazyLoad();

    } catch (err) {
      grid.innerHTML = `<div class="empty-state"><div class="empty-state-icon">⚠️</div><h3>Error</h3><p>${err.message}</p></div>`;
    }
  }

  function initGalleryLazyLoad() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    $$('.gallery-card').forEach(card => {
      card.style.opacity = '0';
      card.style.transition = 'opacity 0.4s ease';
      observer.observe(card);
    });
  }

  async function deleteGalleryItem(id) {
    state.deleteCallback = async () => {
      try {
        await apiFetch(`/api/outputs/${encodeURIComponent(id)}`, { method: 'DELETE' });
        showToast('success', 'Deleted', 'Video removed from gallery');
        loadGallery();
      } catch (err) {
        showToast('error', 'Delete failed', err.message);
      }
    };
    $('#deleteMessage').textContent = 'Are you sure you want to delete this video?';
    $('#deleteModal').classList.remove('hidden');
  }

  // ============================
  // SYSTEM INFO
  // ============================
  async function fetchSystemInfo() {
    const badge = $('#systemBadge');
    const dot = $('.badge-dot', badge);
    const text = $('.badge-text', badge);

    try {
      dot.className = 'badge-dot loading';
      text.textContent = 'Checking...';

      const info = await apiFetch('/api/system/info');
      dot.className = 'badge-dot online';

      // Determine display text
      if (info.gpu_name || info.gpu) {
        const gpuName = info.gpu_name || info.gpu || 'GPU';
        const vram = info.vram_total ? ` (${Math.round(info.vram_total / 1024)}GB)` : '';
        text.textContent = `${gpuName}${vram}`;
      } else if (info.device) {
        text.textContent = info.device;
      } else {
        text.textContent = 'Online';
      }

      // Store for settings display
      state.systemInfo = info;

    } catch (err) {
      dot.className = 'badge-dot error';
      text.textContent = 'Offline';
    }
  }

  async function renderSystemInfo() {
    const container = $('#systemInfo');
    try {
      const info = await apiFetch('/api/system/info');
      const fields = [
        { label: 'Device', value: info.gpu_name || info.gpu || info.device || 'N/A' },
        { label: 'VRAM', value: info.vram_total ? `${formatBytes(info.vram_total * 1024 * 1024)}` : 'N/A' },
        { label: 'VRAM Free', value: info.vram_free ? `${formatBytes(info.vram_free * 1024 * 1024)}` : 'N/A' },
        { label: 'RAM', value: info.ram_total ? formatBytes(info.ram_total * 1024 * 1024) : 'N/A' },
        { label: 'CPU', value: info.cpu || 'N/A' },
        { label: 'Python', value: info.python_version || 'N/A' },
        { label: 'PyTorch', value: info.torch_version || info.pytorch_version || 'N/A' },
        { label: 'CUDA', value: info.cuda_version || 'N/A' },
      ];

      container.innerHTML = fields
        .map(f => `<div class="system-info-row"><span class="system-info-label">${f.label}</span><span class="system-info-value">${escapeHtml(f.value)}</span></div>`)
        .join('');
    } catch (err) {
      container.innerHTML = `<div class="system-info-row"><span class="system-info-label">Status</span><span class="system-info-value" style="color:var(--red)">Offline</span></div>`;
    }
  }

  // ============================
  // SETTINGS MODAL
  // ============================
  function initSettings() {
    const modal = $('#settingsModal');
    const openBtn = $('#settingsBtn');
    const closeBtn = $('#settingsClose');
    const saveBtn = $('#saveSettingsBtn');

    openBtn.addEventListener('click', () => {
      // Populate current values
      $('#serverUrlInput').value = CONFIG.serverUrl;
      $('#globalNsfwToggle').checked = CONFIG.allowNsfw;
      $('#defaultFrames').value = CONFIG.defaultFrames;
      $('#defaultSteps').value = CONFIG.defaultSteps;
      $('#defaultFps').value = CONFIG.defaultFps;
      renderSystemInfo();
      modal.classList.remove('hidden');
    });

    closeBtn.addEventListener('click', () => {
      modal.classList.add('hidden');
    });

    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.add('hidden');
    });

    saveBtn.addEventListener('click', () => {
      const url = $('#serverUrlInput').value.trim();
      if (url) CONFIG.serverUrl = url;
      CONFIG.allowNsfw = $('#globalNsfwToggle').checked;
      const frames = parseInt($('#defaultFrames').value, 10);
      const steps = parseInt($('#defaultSteps').value, 10);
      const fps = parseInt($('#defaultFps').value, 10);
      if (frames > 0) CONFIG.defaultFrames = frames;
      if (steps > 0) CONFIG.defaultSteps = steps;
      if (fps > 0) CONFIG.defaultFps = fps;

      // Update UI sliders
      $('#framesSlider').value = CONFIG.defaultFrames;
      $('#framesValue').textContent = CONFIG.defaultFrames;
      $('#stepsSlider').value = CONFIG.defaultSteps;
      $('#stepsValue').textContent = CONFIG.defaultSteps;
      $('#fpsSlider').value = CONFIG.defaultFps;
      $('#fpsValue').textContent = CONFIG.defaultFps;

      showToast('success', 'Settings saved', 'Configuration updated');
      modal.classList.add('hidden');
      fetchSystemInfo();
    });
  }

  // ============================
  // DELETE MODAL
  // ============================
  function initDeleteModal() {
    const modal = $('#deleteModal');
    const cancelBtn = $('#deleteCancelBtn');
    const confirmBtn = $('#deleteConfirmBtn');

    cancelBtn.addEventListener('click', () => {
      modal.classList.add('hidden');
      state.deleteCallback = null;
    });

    confirmBtn.addEventListener('click', () => {
      modal.classList.add('hidden');
      if (state.deleteCallback) {
        state.deleteCallback();
        state.deleteCallback = null;
      }
    });

    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.add('hidden');
        state.deleteCallback = null;
      }
    });
  }

  // ============================
  // REFRESH BUTTONS
  // ============================
  function initRefreshButtons() {
    $('#refreshModelsBtn').addEventListener('click', () => {
      loadModels();
    });
    $('#refreshGalleryBtn').addEventListener('click', () => {
      loadGallery();
    });
  }

  // ============================
  // GENERATE BUTTON
  // ============================
  function initGenerateButton() {
    $('#generateBtn').addEventListener('click', startGeneration);
    $('#cancelBtn').addEventListener('click', cancelGeneration);
  }

  // ============================
  // DOWNLOAD MODEL BUTTON
  // ============================
  function initDownloadButton() {
    $('#downloadModelBtn').addEventListener('click', () => {
      downloadModel();
    });
  }

  // ============================
  // HTML ESCAPE
  // ============================
  function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ============================
  // KEYBOARD SHORTCUTS
  // ============================
  function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Escape closes modals and chat
      if (e.key === 'Escape') {
        $('#settingsModal').classList.add('hidden');
        $('#deleteModal').classList.add('hidden');
        $('#charSettingsModal').classList.add('hidden');
        if (state.chatOpen) chatTogglePanel();
      }
      // Ctrl/Cmd+Enter to generate
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && state.currentTab === 'create') {
        e.preventDefault();
        startGeneration();
      }
    });
  }

  // ============================
  // HEALTH CHECK
  // ============================
  async function healthCheck() {
    try {
      await apiFetch('/api/system/health');
      return true;
    } catch {
      return false;
    }
  }

  // ============================
  // AI CHAT — Session, Messaging, Character
  // ============================
  function formatChatTimestamp(isoStr) {
    if (!isoStr) return '';
    const d = new Date(isoStr);
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return sameDay ? `${hh}:${mm}` : `${d.getMonth()+1}/${d.getDate()} ${hh}:${mm}`;
  }

  function chatScrollToBottom() {
    const container = $('#chatMessages');
    if (container) container.scrollTop = container.scrollHeight;
  }

  function chatRenderMessages() {
    const container = $('#chatMessages');
    const emptyEl = $('#chatEmpty');
    if (!container) return;

    if (state.chatMessages.length === 0) {
      emptyEl.classList.remove('hidden');
      return;
    }
    emptyEl.classList.add('hidden');

    container.innerHTML = '';
    state.chatMessages.forEach(msg => {
      const isUser = msg.role === 'user';
      const div = el('div', { className: `chat-msg ${isUser ? 'chat-msg-user' : 'chat-msg-ai'}` });
      div.innerHTML = `
        <div class="chat-msg-bubble">${escapeHtml(msg.content)}</div>
        <div class="chat-msg-time">${formatChatTimestamp(msg.created_at || msg.timestamp)}</div>
      `;
      container.appendChild(div);
    });

    chatScrollToBottom();
  }

  function chatUpdateBadge() {
    const badge = $('#chatBadge');
    if (!badge) return;
    if (state.chatUnreadCount > 0 && !state.chatOpen) {
      badge.textContent = state.chatUnreadCount > 99 ? '99+' : state.chatUnreadCount;
      badge.classList.remove('hidden');
    } else {
      badge.classList.add('hidden');
    }
  }

  function chatSetTyping(show) {
    const el = $('#chatTyping');
    if (el) el.classList.toggle('hidden', !show);
    if (show) chatScrollToBottom();
  }

  async function chatCreateSession() {
    try {
      const result = await apiFetch('/api/chat/session', { method: 'POST' });
      state.chatSessionId = result.session_id || result.id;
      return state.chatSessionId;
    } catch (err) {
      console.warn('Failed to create chat session:', err);
      showToast('error', 'Chat error', 'Could not create chat session');
      return null;
    }
  }

  async function chatLoadSession(sessionId) {
    try {
      const result = await apiFetch(`/api/chat/session/${encodeURIComponent(sessionId)}`);
      state.chatSessionId = sessionId;
      state.chatMessages = result.messages || [];

      if (result.character) {
        state.chatCharacter = { ...state.chatCharacter, ...result.character };
        chatUpdateCharacterUI();
      }

      chatRenderMessages();
    } catch (err) {
      console.warn('Failed to load chat session:', err);
    }
  }

  async function chatSendMessage(text) {
    if (!text.trim() || state.chatSending) return;

    const trimmed = text.trim();

    // Ensure session exists
    if (!state.chatSessionId) {
      const sid = await chatCreateSession();
      if (!sid) return;
      localStorage.setItem('chatSessionId', state.chatSessionId);
    }

    state.chatSending = true;
    $('#chatSendBtn').disabled = true;
    $('#chatInput').value = '';

    // Optimistically add user message
    const userMsg = { role: 'user', content: trimmed, created_at: new Date().toISOString() };
    state.chatMessages.push(userMsg);
    chatRenderMessages();
    chatSetTyping(true);

    try {
      const result = await apiFetch('/api/chat/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed, session_id: state.chatSessionId }),
      });

      chatSetTyping(false);

      const aiMsg = {
        role: 'assistant',
        content: result.response || result.message || result.content || '(No response)',
        created_at: result.created_at || new Date().toISOString(),
      };
      state.chatMessages.push(aiMsg);
      chatRenderMessages();

      // Update unread if panel is closed
      if (!state.chatOpen) {
        state.chatUnreadCount++;
        chatUpdateBadge();
      }

    } catch (err) {
      chatSetTyping(false);
      showToast('error', 'Chat error', err.message);
    } finally {
      state.chatSending = false;
      $('#chatSendBtn').disabled = false;
    }
  }

  async function chatUpdateCharacter() {
    if (!state.chatSessionId) return;
    try {
      const char = { ...state.chatCharacter };
      await apiFetch(`/api/chat/session/${encodeURIComponent(state.chatSessionId)}/character`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(char),
      });
      showToast('success', 'Character saved', `${char.name || 'AI Assistant'} updated`);
    } catch (err) {
      showToast('error', 'Save failed', err.message);
    }
  }

  function chatUpdateCharacterUI() {
    const nameEl = $('#chatCharName');
    if (nameEl) nameEl.textContent = state.chatCharacter.name || 'AI Assistant';
  }

  function chatPopulateCharSettings() {
    const c = state.chatCharacter;
    const nameInput = $('#charNameInput');
    const ageInput = $('#charAgeInput');
    const persInput = $('#charPersonalityInput');
    const appInput = $('#charAppearanceInput');
    const scenInput = $('#charScenarioInput');
    const nsfwInput = $('#charNsfwToggle');
    if (nameInput) nameInput.value = c.name || '';
    if (ageInput) ageInput.value = c.age || '';
    if (persInput) persInput.value = c.personality || '';
    if (appInput) appInput.value = c.appearance || '';
    if (scenInput) scenInput.value = c.scenario || '';
    if (nsfwInput) nsfwInput.checked = !!c.nsfw;
  }

  function chatReadCharSettings() {
    return {
      name: ($('#charNameInput') || {}).value || 'AI Assistant',
      age: parseInt(($('#charAgeInput') || {}).value, 10) || null,
      personality: ($('#charPersonalityInput') || {}).value || '',
      appearance: ($('#charAppearanceInput') || {}).value || '',
      scenario: ($('#charScenarioInput') || {}).value || '',
      nsfw: (($('#charNsfwToggle') || {}).checked) || false,
    };
  }

  function chatTogglePanel() {
    state.chatOpen = !state.chatOpen;
    const panel = $('#chatPanel');
    const toggle = $('#chatToggle');
    if (panel) panel.classList.toggle('hidden', !state.chatOpen);
    if (toggle) toggle.classList.toggle('active', state.chatOpen);

    if (state.chatOpen) {
      state.chatUnreadCount = 0;
      chatUpdateBadge();
      setTimeout(() => { if (state.chatOpen) $('#chatInput').focus(); }, 100);
    }
  }

  function initChat() {
    // Toggle panel
    $('#chatToggle').addEventListener('click', chatTogglePanel);

    // Close button
    $('#chatCloseBtn').addEventListener('click', () => {
      if (state.chatOpen) chatTogglePanel();
    });

    // Send message
    $('#chatSendBtn').addEventListener('click', () => {
      const input = $('#chatInput');
      if (input && input.value.trim()) chatSendMessage(input.value);
    });

    // Enter to send
    const chatInput = $('#chatInput');
    if (chatInput) {
      chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (chatInput.value.trim()) chatSendMessage(chatInput.value);
        }
      });
    }

    // Character settings open
    $('#chatSettingsBtn').addEventListener('click', () => {
      chatPopulateCharSettings();
      $('#charSettingsModal').classList.remove('hidden');
    });

    // Character settings close
    $('#charSettingsClose').addEventListener('click', () => {
      $('#charSettingsModal').classList.add('hidden');
    });

    $('#charSettingsCancelBtn').addEventListener('click', () => {
      $('#charSettingsModal').classList.add('hidden');
    });

    $('#charSettingsModal').addEventListener('click', (e) => {
    if (e.target === $('#charSettingsModal')) $('#charSettingsModal').classList.add('hidden');
    });

    // Character settings save
    $('#charSettingsSaveBtn').addEventListener('click', () => {
      state.chatCharacter = chatReadCharSettings();
      chatUpdateCharacterUI();
      chatUpdateCharacter();
      $('#charSettingsModal').classList.add('hidden');
    });

    // Try to restore session from localStorage
    const savedSession = localStorage.getItem('chatSessionId');
    if (savedSession) {
      chatLoadSession(savedSession);
    }

    chatUpdateBadge();
  }

  // ============================
  // PERSISTENT GENERATION STATE
  // ============================
  async function restoreGenerations() {
    try {
      const result = await apiFetch('/api/generations');
      const gens = Array.isArray(result) ? result : (result.generations || []);

      if (gens.length === 0) return;

      let hasActive = false;

      gens.forEach(gen => {
        const id = gen.video_id || gen.id || gen.task_id;
        if (!id) return;

        const status = gen.status || 'completed';

        if (status === 'processing' || status === 'generating' || status === 'pending' || status === 'queued') {
          // Active generation — reconnect WebSocket + polling
          hasActive = true;
          state.activeGenerations[id] = gen;

          // If this is the generation the user was watching, show progress
          if (!state.generating) {
            state.generating = true;
            state.currentVideoId = id;
            $('#generateBtn').disabled = true;
            $('#progressPanel').classList.remove('hidden');
            const statusMsg = gen.message || gen.step || gen.current_step || 'Generating...';
            updateProgress(gen.progress || 0, statusMsg);
            state.startTime = gen.started_at ? new Date(gen.started_at).getTime() : (gen.created_at ? new Date(gen.created_at * 1000).getTime() : Date.now());
            state.elapsedInterval = setInterval(updateElapsedTime, 1000);

            connectWebSocket(id);
            startPolling(id);
          }
        }
      });

      if (hasActive) {
        showToast('info', 'Active generations', 'Found in-progress generations');
      }

      // Make sure completed ones show in gallery
      const completedGens = gens.filter(g => (g.status === 'completed') && (g.video_id || g.id));
      if (completedGens.length > 0 && state.currentTab === 'gallery') {
        loadGallery();
      }

    } catch (err) {
      // Silently fail — generations endpoint may not exist yet
      console.warn('Could not restore generations:', err.message);
    }
  }

  // ============================
  // INIT
  // ============================
  async function init() {
    initTabs();
    initGenModeToggle();
    initSourceToggle();
    initCollapsibles();
    initImageUpload();
    initSliders();
    initSeedButton();
    initCharCount();
    initSettings();
    initDeleteModal();
    initRefreshButtons();
    initGenerateButton();
    initDownloadButton();
    initKeyboardShortcuts();
    updateModeUI();
    updateSourceUI();

    // Apply saved defaults
    $('#framesSlider').value = CONFIG.defaultFrames;
    $('#framesValue').textContent = CONFIG.defaultFrames;
    $('#stepsSlider').value = CONFIG.defaultSteps;
    $('#stepsValue').textContent = CONFIG.defaultSteps;
    $('#fpsSlider').value = CONFIG.defaultFps;
    $('#fpsValue').textContent = CONFIG.defaultFps;

    // Check server health
    const isHealthy = await healthCheck();
    if (!isHealthy) {
      showToast('error', 'Server offline', `Cannot connect to ${CONFIG.serverUrl}. Check settings.`);
    }

    // Load initial data
    fetchSystemInfo();
    loadModels();

    // Init chat
    initChat();

    // Restore persistent generation state
    restoreGenerations();
  }

  // Start the app
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
