/**
 * Intelligent Document Extraction Platform - Frontend Application Script
 * 
 * Handles:
 * - Backend health status checking (GET /api/v1/health)
 * - Document type selection (Invoice, Balance Sheet, Profit & Loss, Cash Flow)
 * - File upload via drag & drop or file browser
 * - Client-side validation for file types (PDF, JPG, JPEG, PNG) and size
 * - Form submission to POST /api/v1/documents/process
 * - Dynamic rendering of:
 *   1. Processing overview and status badge
 *   2. Financial validation results (PASS, FAILED, NOT_APPLICABLE) with formulas and variances
 *   3. Extracted key-value fields and financial line items table
 *   4. Formatted raw JSON response with clipboard copy
 */

(function () {
  'use strict';

  // =========================================================================
  // Configuration & Constants
  // =========================================================================
  const API_BASE_URL = window.API_BASE_URL || (
    window.location.origin.startsWith('http') ? window.location.origin : 'http://127.0.0.1:8000'
  );

  const ALLOWED_EXTENSIONS = ['pdf', 'jpg', 'jpeg', 'png'];
  const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50MB limit

  // =========================================================================
  // State
  // =========================================================================
  let currentFile = null;
  let currentDocType = 'invoice';
  let isProcessing = false;
  let stageInterval = null;

  // =========================================================================
  // DOM Elements
  // =========================================================================
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const btnRefreshHealth = document.getElementById('btnRefreshHealth');
  const alertContainer = document.getElementById('alertContainer');

  const docTypeGrid = document.getElementById('docTypeGrid');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const fileInfoBox = document.getElementById('fileInfoBox');
  const fileBadge = document.getElementById('fileBadge');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const btnRemoveFile = document.getElementById('btnRemoveFile');

  const btnProcess = document.getElementById('btnProcess');
  const btnProcessText = document.getElementById('btnProcessText');
  const loadingSection = document.getElementById('loadingSection');
  const loadingStageText = document.getElementById('loadingStageText');

  const resultsWrapper = document.getElementById('resultsWrapper');
  const statusBadgeContainer = document.getElementById('statusBadgeContainer');
  const resDocName = document.getElementById('resDocName');
  const resDocType = document.getElementById('resDocType');
  const resFileType = document.getElementById('resFileType');
  const resPageCount = document.getElementById('resPageCount');
  const resDocId = document.getElementById('resDocId');
  const resTimestamp = document.getElementById('resTimestamp');

  const validationCounts = document.getElementById('validationCounts');
  const badgePassCount = document.getElementById('badgePassCount');
  const badgeFailCount = document.getElementById('badgeFailCount');
  const badgeNaCount = document.getElementById('badgeNaCount');
  const validationList = document.getElementById('validationList');

  const fieldsGrid = document.getElementById('fieldsGrid');
  const lineItemsContainer = document.getElementById('lineItemsContainer');
  const lineItemCountPill = document.getElementById('lineItemCountPill');
  const lineItemsTableHead = document.getElementById('lineItemsTableHead');
  const lineItemsTableBody = document.getElementById('lineItemsTableBody');
  const additionalFieldsContainer = document.getElementById('additionalFieldsContainer');
  const additionalFieldsGrid = document.getElementById('additionalFieldsGrid');

  const rawJsonCode = document.getElementById('rawJsonCode');
  const btnCopyJson = document.getElementById('btnCopyJson');
  const jsonDetails = document.getElementById('jsonDetails');

  // =========================================================================
  // Utility Functions
  // =========================================================================
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
  }

  function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  function formatNumber(num) {
    if (num === null || num === undefined) return '—';
    if (typeof num === 'number') {
      return num.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 4 });
    }
    return String(num);
  }

  function formatTitle(str) {
    if (!str) return '—';
    return String(str)
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  }

  function showAlert(message, type = 'danger') {
    if (!alertContainer) return;
    alertContainer.className = `alert-container alert-${type}`;
    alertContainer.innerHTML = `
      <div class="alert-content">
        <span class="alert-icon">${type === 'success' ? '✓' : type === 'warning' ? '⚠' : 'ℹ'}</span>
        <div class="alert-text">${escapeHtml(message)}</div>
      </div>
      <button type="button" class="alert-dismiss" aria-label="Close alert">&times;</button>
    `;
    alertContainer.classList.remove('hidden');

    const dismissBtn = alertContainer.querySelector('.alert-dismiss');
    if (dismissBtn) {
      dismissBtn.onclick = () => hideAlert();
    }
  }

  function hideAlert() {
    if (!alertContainer) return;
    alertContainer.classList.add('hidden');
    alertContainer.innerHTML = '';
  }

  // =========================================================================
  // Health Check Service
  // =========================================================================
  async function checkBackendHealth() {
    statusDot.className = 'status-dot status-unknown';
    statusText.textContent = 'Checking...';

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });

      if (response.ok) {
        const data = await response.json();
        if (data.status === 'healthy') {
          statusDot.className = 'status-dot status-healthy';
          statusText.textContent = 'Backend Online';
          return;
        }
      }
      statusDot.className = 'status-dot status-unhealthy';
      statusText.textContent = 'Backend Issue';
    } catch (err) {
      statusDot.className = 'status-dot status-unhealthy';
      statusText.textContent = 'Backend Offline';
    }
  }

  // =========================================================================
  // Document Type Selector
  // =========================================================================
  function initDocTypeSelector() {
    if (!docTypeGrid) return;
    const cards = docTypeGrid.querySelectorAll('.doc-type-card');

    cards.forEach(card => {
      card.addEventListener('click', () => {
        cards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');

        const radio = card.querySelector('input[type="radio"]');
        if (radio) {
          radio.checked = true;
          currentDocType = radio.value;
        }
      });
    });
  }

  // =========================================================================
  // File Upload & Drag-and-Drop
  // =========================================================================
  function getFileExtension(filename) {
    if (!filename || !filename.includes('.')) return '';
    return filename.split('.').pop().toLowerCase();
  }

  function validateFile(file) {
    if (!file) return { valid: false, error: 'No file selected.' };

    const ext = getFileExtension(file.name);
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return {
        valid: false,
        error: `Unsupported file format ".${ext}". Please upload a PDF, JPG, JPEG, or PNG file.`,
      };
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      return {
        valid: false,
        error: `File size (${formatBytes(file.size)}) exceeds maximum limit of 50MB.`,
      };
    }

    if (file.size === 0) {
      return {
        valid: false,
        error: 'The selected file is empty (0 bytes).',
      };
    }

    return { valid: true };
  }

  function setFile(file) {
    const validation = validateFile(file);
    if (!validation.valid) {
      showAlert(validation.error, 'danger');
      clearFile();
      return;
    }

    hideAlert();
    currentFile = file;
    const ext = getFileExtension(file.name).toUpperCase();

    fileBadge.textContent = ext;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);

    fileInfoBox.classList.remove('hidden');
    dropZone.classList.add('has-file');
    btnProcess.disabled = false;
  }

  function clearFile() {
    currentFile = null;
    fileInput.value = '';
    fileInfoBox.classList.add('hidden');
    dropZone.classList.remove('has-file');
    btnProcess.disabled = true;
  }

  function initFileUpload() {
    if (!dropZone || !fileInput) return;

    // Trigger file chooser on dropzone click
    dropZone.addEventListener('click', (e) => {
      // Don't trigger if clicking inside file-info-box or remove button
      if (e.target.closest('#fileInfoBox')) return;
      fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        setFile(e.target.files[0]);
      }
    });

    // Remove file button
    btnRemoveFile.addEventListener('click', (e) => {
      e.stopPropagation();
      clearFile();
    });

    // Drag-and-drop events
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        setFile(dt.files[0]);
      }
    });
  }

  // =========================================================================
  // Document Processing Submission
  // =========================================================================
  const PROCESSING_STAGES = [
    'Validating file structure and binary header...',
    'Performing text extraction / Multimodal vision analysis...',
    'Executing structured Gemini financial field extraction...',
    'Applying deterministic accounting and mathematical rules...',
    'Persisting document and validation audit record in PostgreSQL...',
  ];

  function startLoadingAnimation() {
    isProcessing = true;
    btnProcess.disabled = true;
    btnProcessText.textContent = 'Processing...';
    loadingSection.classList.remove('hidden');
    resultsWrapper.classList.add('hidden');
    hideAlert();

    let stageIdx = 0;
    loadingStageText.textContent = PROCESSING_STAGES[0];
    stageInterval = setInterval(() => {
      stageIdx = (stageIdx + 1) % PROCESSING_STAGES.length;
      loadingStageText.textContent = PROCESSING_STAGES[stageIdx];
    }, 2800);
  }

  function stopLoadingAnimation() {
    isProcessing = false;
    btnProcess.disabled = !currentFile;
    btnProcessText.textContent = 'Process Document';
    loadingSection.classList.add('hidden');
    if (stageInterval) {
      clearInterval(stageInterval);
      stageInterval = null;
    }
  }

  async function processDocument() {
    if (!currentFile || isProcessing) return;

    startLoadingAnimation();

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('document_type', currentDocType);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/documents/process`, {
        method: 'POST',
        body: formData,
      });

      let data = null;
      try {
        data = await response.json();
      } catch (jsonErr) {
        throw new Error(`Server returned HTTP ${response.status} with non-JSON response.`);
      }

      stopLoadingAnimation();

      if (response.status === 200 || response.status === 400) {
        // Successful response (either COMPLETED, VALIDATION_FAILED, or pre-validation failure)
        displayResults(data, response.status);
      } else {
        showAlert(`Server error (${response.status}): ${data.detail || 'Internal processing error'}`, 'danger');
      }
    } catch (err) {
      stopLoadingAnimation();
      showAlert(`Network/Processing Error: ${err.message}. Ensure backend is reachable at ${API_BASE_URL}.`, 'danger');
    }
  }

  // =========================================================================
  // Rendering Results
  // =========================================================================
  function displayResults(data, httpStatus) {
    if (!data) return;

    resultsWrapper.classList.remove('hidden');
    resultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });

    renderOverview(data, httpStatus);
    renderValidations(data.validations);
    renderExtractedData(data.extracted_data, data.document_type || currentDocType);
    renderRawJson(data);

    // Provide contextual user feedback based on processing_status
    if (data.processing_status === 'COMPLETED') {
      showAlert('Document processed and all deterministic validation checks passed successfully!', 'success');
    } else if (data.processing_status === 'VALIDATION_FAILED') {
      showAlert('Document extraction succeeded, but deterministic financial validation detected mathematical variance.', 'warning');
    } else if (data.processing_status === 'EXTRACTION_FAILED') {
      showAlert('Extraction could not extract required financial figures from the document.', 'danger');
    }
  }

  function renderOverview(data, httpStatus) {
    resDocName.textContent = data.document_name || (currentFile ? currentFile.name : '—');
    resDocType.textContent = formatTitle(data.document_type || currentDocType);

    const fileVal = data.file_validation || {};
    const meta = data.metadata || {};

    resFileType.textContent = (fileVal.file_type || (currentFile ? getFileExtension(currentFile.name) : '—')).toUpperCase();
    resPageCount.textContent = fileVal.page_count || meta.page_count || '1';
    resDocId.textContent = data.id !== null && data.id !== undefined ? `#${data.id}` : 'Not Persisted';

    let formattedDate = '—';
    if (data.created_at) {
      try {
        formattedDate = new Date(data.created_at).toLocaleString();
      } catch (e) {
        formattedDate = String(data.created_at);
      }
    } else {
      formattedDate = new Date().toLocaleString();
    }
    resTimestamp.textContent = formattedDate;

    // Status Badge
    const status = data.processing_status || (httpStatus === 200 ? 'COMPLETED' : 'FAILED');
    let badgeClass = 'badge-na';
    let icon = '•';

    if (status === 'COMPLETED') {
      badgeClass = 'badge-pass';
      icon = '✓';
    } else if (status === 'VALIDATION_FAILED') {
      badgeClass = 'badge-fail';
      icon = '✗';
    } else if (status === 'EXTRACTION_FAILED' || status === 'FAILED' || status === 'PROCESSING_FAILED') {
      badgeClass = 'badge-fail';
      icon = '⚠';
    }

    statusBadgeContainer.innerHTML = `
      <span class="badge ${badgeClass}" style="font-size: 0.85rem; padding: 0.4rem 0.85rem;">
        ${icon} ${escapeHtml(status)}
      </span>
    `;
  }

  function renderValidations(validations) {
    if (!validations || !validations.checks || validations.checks.length === 0) {
      badgePassCount.textContent = '0 Passed';
      badgeFailCount.textContent = '0 Failed';
      badgeNaCount.textContent = '0 N/A';
      validationList.innerHTML = `
        <div class="empty-state">
          <p>No deterministic mathematical validations were executed for this document.</p>
        </div>
      `;
      return;
    }

    const summary = validations.summary || {};
    const passed = summary.passed_checks ?? validations.checks.filter(c => c.status === 'PASS').length;
    const failed = summary.failed_checks ?? validations.checks.filter(c => c.status === 'FAILED').length;
    const na = summary.not_applicable_checks ?? validations.checks.filter(c => c.status === 'NOT_APPLICABLE').length;

    badgePassCount.textContent = `${passed} Passed`;
    badgeFailCount.textContent = `${failed} Failed`;
    badgeNaCount.textContent = `${na} N/A`;

    let html = '';
    validations.checks.forEach((chk, idx) => {
      const status = chk.status || 'NOT_APPLICABLE';
      let cardClass = 'status-not_applicable';
      let badgeTag = '<span class="badge badge-na">NOT APPLICABLE</span>';

      if (status === 'PASS') {
        cardClass = 'status-pass';
        badgeTag = '<span class="badge badge-pass">✓ PASS</span>';
      } else if (status === 'FAILED') {
        cardClass = 'status-failed';
        badgeTag = '<span class="badge badge-fail">✗ FAILED</span>';
      }

      // Check details / metrics
      let metricsHtml = '';
      if (chk.calculated_value !== null && chk.calculated_value !== undefined) {
        metricsHtml += `<div class="val-metric"><span>Calculated:</span> <strong>${formatNumber(chk.calculated_value)}</strong></div>`;
      }
      if (chk.reported_value !== null && chk.reported_value !== undefined) {
        metricsHtml += `<div class="val-metric"><span>Reported:</span> <strong>${formatNumber(chk.reported_value)}</strong></div>`;
      }
      if (chk.variance !== null && chk.variance !== undefined) {
        metricsHtml += `<div class="val-metric"><span>Variance:</span> <strong>${formatNumber(chk.variance)}</strong></div>`;
      }
      if (chk.tolerance !== null && chk.tolerance !== undefined) {
        metricsHtml += `<div class="val-metric"><span>Tolerance:</span> <strong>${chk.tolerance}</strong></div>`;
      }

      // Missing fields pill tags
      let missingTagsHtml = '';
      if (chk.missing_fields && chk.missing_fields.length > 0) {
        missingTagsHtml = `
          <div class="val-metric">
            <span>Missing Fields:</span>
            ${chk.missing_fields.map(f => `<span class="val-missing-tag">${escapeHtml(f)}</span>`).join(' ')}
          </div>
        `;
      }

      html += `
        <div class="validation-card-item ${cardClass}">
          <div class="val-header">
            <span class="val-rule-name">${escapeHtml(chk.rule_name || `Check #${idx + 1}`)}</span>
            ${badgeTag}
          </div>
          ${chk.formula ? `<div class="val-formula">${escapeHtml(chk.formula)}</div>` : ''}
          ${metricsHtml || missingTagsHtml ? `<div class="val-details-row">${metricsHtml} ${missingTagsHtml}</div>` : ''}
          ${chk.message ? `<div class="val-message">${escapeHtml(chk.message)}</div>` : ''}
        </div>
      `;
    });

    validationList.innerHTML = html;
  }

  function renderExtractedData(extractedData, docType) {
    if (!extractedData || Object.keys(extractedData).length === 0) {
      fieldsGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <p>No structured data extracted.</p>
        </div>
      `;
      lineItemsContainer.classList.add('hidden');
      additionalFieldsContainer.classList.add('hidden');
      return;
    }

    // Separate flat fields from line_items or complex nested objects
    const flatFields = {};
    const complexFields = {};
    let lineItems = null;

    for (const [key, value] of Object.entries(extractedData)) {
      if (key === 'line_items' && Array.isArray(value)) {
        lineItems = value;
      } else if (Array.isArray(value) || (typeof value === 'object' && value !== null)) {
        complexFields[key] = value;
      } else {
        flatFields[key] = value;
      }
    }

    // Render Primary Flat Fields
    let gridHtml = '';
    for (const [key, val] of Object.entries(flatFields)) {
      const isNull = val === null || val === undefined || val === '';
      const displayVal = isNull ? 'null' : formatNumber(val);
      gridHtml += `
        <div class="field-card">
          <span class="field-key">${escapeHtml(formatTitle(key))}</span>
          <span class="field-val ${isNull ? 'null-val' : ''}" style="white-space: pre-line;">${escapeHtml(displayVal)}</span>
        </div>
      `;
    }
    fieldsGrid.innerHTML = gridHtml || '<p class="text-muted">No flat entity fields present.</p>';

    // Render Line Items Table
    if (lineItems && lineItems.length > 0) {
      lineItemsContainer.classList.remove('hidden');
      lineItemCountPill.textContent = `${lineItems.length} item${lineItems.length === 1 ? '' : 's'}`;

      // Detect headers based on keys across line items
      const sampleItem = lineItems[0] || {};
      const keys = Object.keys(sampleItem);

      // Render Table Headers
      lineItemsTableHead.innerHTML = `
        <th>#</th>
        ${keys.map(k => `<th>${escapeHtml(formatTitle(k))}</th>`).join('')}
      `;

      // Render Table Rows
      let rowsHtml = '';
      lineItems.forEach((item, idx) => {
        rowsHtml += `
          <tr>
            <td><strong>${idx + 1}</strong></td>
            ${keys.map(k => {
              const val = item[k];
              const isNull = val === null || val === undefined || val === '';
              return `<td>${isNull ? '<span class="text-muted">—</span>' : escapeHtml(formatNumber(val))}</td>`;
            }).join('')}
          </tr>
        `;
      });
      lineItemsTableBody.innerHTML = rowsHtml;
    } else {
      lineItemsContainer.classList.add('hidden');
    }

    // Render Additional / Disclosures Complex Fields
    const complexEntries = Object.entries(complexFields);
    if (complexEntries.length > 0) {
      additionalFieldsContainer.classList.remove('hidden');
      let addHtml = '';
      for (const [k, v] of complexEntries) {
        addHtml += `
          <div class="field-card" style="grid-column: span 2;">
            <span class="field-key">${escapeHtml(formatTitle(k))}</span>
            <pre style="font-size: 0.8rem; background: #fff; padding: 6px; border-radius: 4px; overflow-x: auto;"><code>${escapeHtml(JSON.stringify(v, null, 2))}</code></pre>
          </div>
        `;
      }
      additionalFieldsGrid.innerHTML = addHtml;
    } else {
      additionalFieldsContainer.classList.add('hidden');
    }
  }

  function renderRawJson(data) {
    if (!rawJsonCode) return;
    const jsonStr = JSON.stringify(data, null, 2);
    rawJsonCode.innerHTML = `<code>${escapeHtml(jsonStr)}</code>`;

    if (btnCopyJson) {
      btnCopyJson.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        try {
          await navigator.clipboard.writeText(jsonStr);
          const orig = btnCopyJson.textContent;
          btnCopyJson.textContent = 'Copied!';
          setTimeout(() => { btnCopyJson.textContent = orig; }, 2000);
        } catch (err) {
          // Fallback selection
          const range = document.createRange();
          range.selectNodeContents(rawJsonCode);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          document.execCommand('copy');
          btnCopyJson.textContent = 'Copied!';
          setTimeout(() => { btnCopyJson.textContent = 'Copy JSON'; }, 2000);
        }
      };
    }
  }

  // =========================================================================
  // Initialization
  // =========================================================================
  function init() {
    initDocTypeSelector();
    initFileUpload();

    if (btnProcess) {
      btnProcess.addEventListener('click', processDocument);
    }

    if (btnRefreshHealth) {
      btnRefreshHealth.addEventListener('click', (e) => {
        e.stopPropagation();
        checkBackendHealth();
      });
    }

    // Initial Health Check
    checkBackendHealth();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
