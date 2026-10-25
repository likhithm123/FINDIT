/**
 * ChromaDB Multimodal Vector Search - Frontend Controller
 */

let state = {
  currentTab: 'text',
  catalogItems: [],
  activeQueryImageId: null,
  activeUploadedFile: null,
  lastQueryVector: null,
  isSearching: false
};

// Initialize Application on Page Load
document.addEventListener('DOMContentLoaded', async () => {
  setupDragAndDrop();
  await loadDatabaseStats();
  await loadCatalogExplorer();
  
  // Execute initial showcase search
  document.getElementById('text-query-input').value = 'sunset beach with warm golden waves';
  executeTextSearch();
});

/**
 * Fetch and update database telemetry pills
 */
async function loadDatabaseStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error('Stats API failed');
    const data = await res.json();
    
    document.getElementById('stat-vector-count').textContent = data.total_vectors;
    const statusPill = document.getElementById('db-status-pill');
    statusPill.innerHTML = `<span class="status-dot"></span><span>ChromaDB Active (${data.total_vectors} items)</span>`;
  } catch (err) {
    console.error('Failed to load DB stats:', err);
    document.getElementById('db-status-pill').innerHTML = `<span class="status-dot" style="background-color:#ef4444"></span><span>DB Offline</span>`;
  }
}

/**
 * Switch tabs: 'text', 'image', 'explorer', 'index'
 */
function switchTab(tabName) {
  state.currentTab = tabName;

  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  const activeBtn = document.getElementById(`tab-${tabName}`);
  const activeContent = document.getElementById(`section-${tabName}`);

  if (activeBtn) activeBtn.classList.add('active');
  if (activeContent) activeContent.classList.add('active');

  // Toggle results section visibility
  const resultsSection = document.getElementById('results-section');
  if (tabName === 'explorer' || tabName === 'index' || tabName === 'map') {
    resultsSection.style.display = 'none';
  } else {
    resultsSection.style.display = 'block';
  }

  if (tabName === 'explorer') {
    loadCatalogExplorer();
  } else if (tabName === 'map') {
    renderVectorSpaceMap();
  }
}

/**
 * Render 2D Vector Space Projection on HTML5 Canvas
 */
async function renderVectorSpaceMap() {
  const canvas = document.getElementById('vector-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  // Clear canvas
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  // Subtle coordinate grid
  ctx.strokeStyle = '#f1f5f9';
  ctx.lineWidth = 1;
  for (let x = 40; x < w; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 40; y < h; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw axes
  ctx.strokeStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.moveTo(w / 2, 20);
  ctx.lineTo(w / 2, h - 20);
  ctx.moveTo(20, h / 2);
  ctx.lineTo(w - 20, h / 2);
  ctx.stroke();

  try {
    const res = await fetch('/api/visualize');
    const data = await res.json();
    const points = data.points;

    if (!points || points.length === 0) return;

    // Find min and max for auto-scaling
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach(p => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });

    const spanX = Math.max(maxX - minX, 0.1);
    const spanY = Math.max(maxY - minY, 0.1);
    const pad = 60;

    const catColors = {
      nature: '#2563eb',
      architecture: '#7c3aed',
      urban: '#db2777',
      objects: '#ea580c'
    };

    // Draw connection lines to center or nearest
    points.forEach(p => {
      const cx = pad + ((p.x - minX) / spanX) * (w - 2 * pad);
      const cy = h - (pad + ((p.y - minY) / spanY) * (h - 2 * pad));

      const color = catColors[p.category] || '#64748b';

      // Outer glow circle
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, 2 * Math.PI);
      ctx.fillStyle = color + '15';
      ctx.fill();

      // Main point
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.fillStyle = '#0f172a';
      ctx.font = '600 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(p.title.split(' ')[0] + ' ' + (p.title.split(' ')[1] || ''), cx, cy - 14);

      ctx.fillStyle = '#64748b';
      ctx.font = '400 9px monospace';
      ctx.fillText(`(${p.x.toFixed(2)}, ${p.y.toFixed(2)})`, cx, cy + 22);
    });
  } catch (err) {
    console.error('Failed to render 2D map:', err);
  }
}

/**
 * Text-to-Image Search Execution
 */
async function executeTextSearch(e) {
  if (e) e.preventDefault();
  const query = document.getElementById('text-query-input').value.trim();
  if (!query) return;

  const category = document.getElementById('filter-category-text').value;
  const topK = parseInt(document.getElementById('filter-topk-text').value, 10);

  setSearchLoading(true);

  try {
    const res = await fetch('/api/search/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK, category })
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    state.lastQueryVector = data.query_vector_sample;
    renderTelemetry(data.latency_ms);
    renderResults(data.results, `Text Query: "${query}"`);
  } catch (err) {
    console.error('Search error:', err);
    alert('Error running search: ' + err.message);
  } finally {
    setSearchLoading(false);
  }
}

/**
 * Preset chip clicked
 */
function applyPresetQuery(text) {
  document.getElementById('text-query-input').value = text;
  executeTextSearch();
}

/**
 * Image-to-Image search by existing asset ID
 */
async function selectPickerAsset(imageId) {
  state.activeQueryImageId = imageId;
  state.activeUploadedFile = null;

  // Clear upload dropzone UI
  document.getElementById('upload-preview-container').style.display = 'none';
  document.getElementById('upload-prompt').style.display = 'block';

  // Highlight selected thumb
  document.querySelectorAll('.picker-thumb').forEach(img => {
    img.classList.toggle('selected', img.dataset.id === imageId);
  });

  await reRunImageSearch();
}

/**
 * Execute or re-run Image-to-Image search
 */
async function reRunImageSearch() {
  const category = document.getElementById('filter-category-image').value;
  const topK = parseInt(document.getElementById('filter-topk-image').value, 10);

  if (state.activeUploadedFile) {
    // Search using uploaded file
    setSearchLoading(true);
    const formData = new FormData();
    formData.append('file', state.activeUploadedFile);
    formData.append('category', category);
    formData.append('top_k', topK);

    try {
      const res = await fetch('/api/search/upload', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      state.lastQueryVector = data.query_vector_sample;
      renderTelemetry(data.latency_ms);
      renderResults(data.results, `Image Query: "${state.activeUploadedFile.name}"`);
    } catch (err) {
      console.error(err);
      alert('Error searching image: ' + err.message);
    } finally {
      setSearchLoading(false);
    }
  } else if (state.activeQueryImageId) {
    // Search using existing database ID
    setSearchLoading(true);
    try {
      const res = await fetch('/api/search/image-id', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: state.activeQueryImageId,
          category,
          top_k: topK
        })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      state.lastQueryVector = data.query_vector_sample;
      renderTelemetry(data.latency_ms);
      renderResults(data.results, `Image Query ID: ${state.activeQueryImageId}`);
    } catch (err) {
      console.error(err);
      alert('Error searching image ID: ' + err.message);
    } finally {
      setSearchLoading(false);
    }
  }
}

/**
 * Handle image file selection
 */
function handleImageFileSelect(input) {
  if (input.files && input.files[0]) {
    const file = input.files[0];
    state.activeUploadedFile = file;
    state.activeQueryImageId = null;

    // Deselect thumbs
    document.querySelectorAll('.picker-thumb').forEach(img => img.classList.remove('selected'));

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('upload-preview-img').src = e.target.result;
      document.getElementById('upload-preview-container').style.display = 'block';
      document.getElementById('upload-prompt').style.display = 'none';
    };
    reader.readAsDataURL(file);

    reRunImageSearch();
  }
}

function clearImageUpload(e) {
  if (e) e.stopPropagation();
  state.activeUploadedFile = null;
  document.getElementById('image-file-input').value = '';
  document.getElementById('upload-preview-container').style.display = 'none';
  document.getElementById('upload-prompt').style.display = 'block';
}

function setupDragAndDrop() {
  const dropzone = document.getElementById('dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-active');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-active');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      const input = document.getElementById('image-file-input');
      input.files = files;
      handleImageFileSelect(input);
    }
  });
}

/**
 * Render telemetry stats bar
 */
function renderTelemetry(latencyMs) {
  const bar = document.getElementById('telemetry-bar');
  bar.style.display = 'flex';
  document.getElementById('telemetry-latency').textContent = `${latencyMs} ms`;
}

/**
 * Render Search Results
 */
function renderResults(results, queryLabel) {
  const grid = document.getElementById('results-grid');
  const countEl = document.getElementById('results-count');
  const headerEl = document.getElementById('results-header');

  grid.innerHTML = '';

  if (!results || results.length === 0) {
    countEl.textContent = '0 results found';
    grid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted); background: var(--bg-secondary); border-radius: var(--radius-md); border: 1px dashed var(--border-light);">
        No matching vectors found for current query & filters in ChromaDB.
      </div>
    `;
    return;
  }

  countEl.textContent = `${results.length} ranked neighbors (${queryLabel})`;

  results.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = 'result-card';

    const tagsHtml = item.tags.map(t => `<span class="tag-pill">#${escapeHtml(t)}</span>`).join('');

    card.innerHTML = `
      <div class="card-img-wrapper">
        <img src="${item.image_url}" alt="${escapeHtml(item.title)}" class="card-img" loading="lazy">
        <span class="card-rank-badge">#${index + 1}</span>
        <span class="card-sim-badge">${item.similarity_percent}% Match</span>
      </div>
      <div class="card-body">
        <div class="card-meta-top">
          <span class="card-category">${escapeHtml(item.category)}</span>
          <span class="card-distance" title="Cosine distance d = 1 - cos(θ)">d = ${item.distance.toFixed(4)}</span>
        </div>
        <h3 class="card-title">${escapeHtml(item.title)}</h3>
        <p class="card-desc">${escapeHtml(item.description)}</p>

        <div class="similarity-bar-wrap">
          <div class="similarity-bar-bg">
            <div class="similarity-bar-fill" style="width: ${item.similarity_percent}%"></div>
          </div>
        </div>

        <div class="card-tags">
          ${tagsHtml}
        </div>

        <div class="card-actions">
          <button class="card-action-btn" onclick="findSimilarByCard('${item.id}')" title="Use as Image Query">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            Find Similar
          </button>
          <button class="card-action-btn" onclick="openItemVectorModal('${item.id}', '${escapeHtml(item.title)}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            </svg>
            Vector
          </button>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

/**
 * Trigger Image-to-Image search directly from a card's "Find Similar"
 */
function findSimilarByCard(imageId) {
  switchTab('image');
  selectPickerAsset(imageId);
}

/**
 * Load Database Explorer Table & Picker Thumbnails
 */
async function loadCatalogExplorer() {
  const tbody = document.getElementById('explorer-table-body');
  const pickerThumbs = document.getElementById('picker-thumbnails');

  try {
    const res = await fetch('/api/catalog');
    const data = await res.json();
    state.catalogItems = data.items;

    // 1. Populate Explorer Table
    if (data.items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-muted">No records in collection. Click 'Reseed' to initialize.</td></tr>`;
    } else {
      tbody.innerHTML = data.items.map(item => {
        const tags = item.tags.map(t => `<span class="tag-pill">#${escapeHtml(t)}</span>`).join(' ');
        const vectorPreview = item.embedding_preview.length > 0 
          ? `[${item.embedding_preview.join(', ')}...]` 
          : '128-D vector';

        return `
          <tr>
            <td>
              <img src="${item.image_url}" alt="${escapeHtml(item.title)}" class="table-thumb">
            </td>
            <td>
              <strong>${escapeHtml(item.id)}</strong>
              <div style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml(item.title)}</div>
            </td>
            <td>
              <span class="pill pill-blue">${escapeHtml(item.category)}</span>
            </td>
            <td>${tags}</td>
            <td>
              <code class="code-badge" style="cursor:pointer" onclick="openItemVectorModal('${item.id}', '${escapeHtml(item.title)}')">${vectorPreview}</code>
            </td>
            <td>
              <button class="btn-text" style="color:#ef4444;" onclick="deleteCatalogItem('${item.id}')">Delete</button>
            </td>
          </tr>
        `;
      }).join('');
    }

    // 2. Populate Picker Thumbnails in Image Tab
    pickerThumbs.innerHTML = data.items.map(item => `
      <img 
        src="${item.image_url}" 
        alt="${escapeHtml(item.title)}" 
        class="picker-thumb ${state.activeQueryImageId === item.id ? 'selected' : ''}" 
        data-id="${item.id}"
        title="${escapeHtml(item.title)}" 
        onclick="selectPickerAsset('${item.id}')"
      >
    `).join('');

    await loadDatabaseStats();
  } catch (err) {
    console.error('Error fetching catalog:', err);
    tbody.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-muted">Failed to load catalog records.</td></tr>`;
  }
}

/**
 * Reseed Default Database Collection
 */
async function reseedDatabase() {
  if (!confirm("Reset and reseed ChromaDB collection with the 12 sample assets?")) return;
  try {
    const res = await fetch('/api/reseed', { method: 'POST' });
    const data = await res.json();
    alert(`Successfully reseeded database! Total vectors: ${data.total_vectors}`);
    await loadCatalogExplorer();
    await loadDatabaseStats();
  } catch (err) {
    alert('Failed to reseed database: ' + err.message);
  }
}

/**
 * Delete single record from ChromaDB
 */
async function deleteCatalogItem(itemId) {
  if (!confirm(`Delete record '${itemId}' from ChromaDB?`)) return;
  try {
    const res = await fetch(`/api/items/${itemId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    await loadCatalogExplorer();
    await loadDatabaseStats();
  } catch (err) {
    alert('Failed to delete item: ' + err.message);
  }
}

/**
 * Index a new user-uploaded asset
 */
async function submitNewAsset(e) {
  e.preventDefault();
  const title = document.getElementById('index-title').value.trim();
  const category = document.getElementById('index-category').value;
  const tags = document.getElementById('index-tags').value.trim();
  const description = document.getElementById('index-description').value.trim();
  const fileInput = document.getElementById('index-file');
  const feedback = document.getElementById('index-feedback');
  const btn = document.getElementById('btn-submit-index');

  if (!fileInput.files || fileInput.files.length === 0) {
    alert('Please select an image file to upload.');
    return;
  }

  const formData = new FormData();
  formData.append('title', title);
  formData.append('category', category);
  formData.append('tags', tags);
  formData.append('description', description);
  formData.append('file', fileInput.files[0]);

  btn.disabled = true;
  feedback.style.color = 'var(--text-muted)';
  feedback.textContent = 'Extracting 128D visual features & indexing into ChromaDB...';

  try {
    const res = await fetch('/api/index', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    feedback.style.color = '#059669';
    feedback.textContent = `Indexed '${data.indexed_id}' successfully! Total in DB: ${data.total_count}`;

    document.getElementById('index-form').reset();
    await loadDatabaseStats();
    await loadCatalogExplorer();

    // After 2 seconds, redirect to explorer tab
    setTimeout(() => {
      switchTab('explorer');
      feedback.textContent = '';
      btn.disabled = false;
    }, 1500);

  } catch (err) {
    feedback.style.color = '#ef4444';
    feedback.textContent = 'Failed: ' + err.message;
    btn.disabled = false;
  }
}

/**
 * Modal: Vector Inspector
 */
function openItemVectorModal(itemId, title) {
  const modal = document.getElementById('vector-modal');
  document.getElementById('vector-modal-title').textContent = `Vector Inspector: ${title} (${itemId})`;

  const item = state.catalogItems.find(i => i.id === itemId);
  let content = '';

  if (item && item.embedding_preview) {
    content = `Asset ID: ${item.id}\nCategory: ${item.category}\nTags: ${item.tags.join(', ')}\n\nFirst Dimensions Sample:\n${JSON.stringify(item.embedding_preview, null, 2)}`;
  } else {
    content = `Loading vector representation for ${itemId}...`;
  }

  document.getElementById('vector-modal-content').textContent = content;
  modal.classList.add('active');
}

function toggleQueryVectorModal() {
  const modal = document.getElementById('vector-modal');
  document.getElementById('vector-modal-title').textContent = 'Active Query Vector in R¹²⁸';
  
  if (state.lastQueryVector) {
    document.getElementById('vector-modal-content').textContent = 
      `Query Vector Coordinates (First 8 of 128 dimensions):\n${JSON.stringify(state.lastQueryVector, null, 2)}\n\nDistance Metric: Cosine (hnsw:space = cosine)\nSpace: Projected continuous unit hypersphere ||v||₂ = 1.0`;
  } else {
    document.getElementById('vector-modal-content').textContent = 'No active query vector available yet. Execute a search first.';
  }
  modal.classList.add('active');
}

function closeVectorModal(e) {
  if (e && e.target !== e.currentTarget && !e.target.classList.contains('modal-close-btn')) return;
  document.getElementById('vector-modal').classList.remove('active');
}

function setSearchLoading(isLoading) {
  state.isSearching = isLoading;
  const countEl = document.getElementById('results-count');
  if (isLoading) {
    countEl.textContent = 'Traversing HNSW graph layers...';
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>"']/g, function(m) {
    return {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[m];
  });
}
