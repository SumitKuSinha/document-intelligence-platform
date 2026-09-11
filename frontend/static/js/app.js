/**
 * Intelligent Document Extraction Platform — Frontend Application Script
 * 
 * Capabilities:
 * - Real-time backend health monitoring (GET /api/v1/health) with animated status pulse
 * - Accessible document type selection (Invoice, Balance Sheet, Profit & Loss, Cash Flow)
 * - Drag-and-drop file upload with format & size validation
 * - 5-Stage visual processing pipeline indicator
 * - Dynamic rendering of:
 *   1. Execution overview, database persistence ID, and status badge
 *   2. Deterministic financial validation results with interactive category filtering (All/Pass/Fail/N/A)
 *   3. Extracted entity information & core financial figures (preserving raw strings vs formatted numbers)
 *   4. Responsive line items schedule table with sticky headers & right-aligned amounts
 *   5. Extraction evidence & source disclosures
 *   6. Collapsible raw JSON API payload with 1-click clipboard copy
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
  const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50MB

  const ENTITY_METADATA_KEYS = new Set([
    'entity_name', 'company_name', 'vendor_name', 'vendor_address', 'vendor_tax_id',
    'customer_name', 'customer_address', 'customer_tax_id',
    'invoice_number', 'invoice_date', 'due_date', 'payment_terms',
    'period_end_date', 'period_start_date', 'fiscal_year', 'reporting_period',
    'currency', 'reporting_scale', 'document_type', 'statement_type'
  ]);

  // =========================================================================
  // State
  // =========================================================================
  let currentFile = null;
  let currentDocType = 'invoice';
  let isProcessing = false;
  let pipelineInterval = null;
  let activeValidationFilter = 'all';

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
  const emptyDashboardState = document.getElementById('emptyDashboardState');

  const resultsWrapper = document.getElementById('resultsWrapper');
  const statusBadgeContainer = document.getElementById('statusBadgeContainer');
  const resDocName = document.getElementById('resDocName');
  const resDocType = document.getElementById('resDocType');
  const resFilePageSpec = document.getElementById('resFilePageSpec');
  const resDocId = document.getElementById('resDocId');
  const resTimestamp = document.getElementById('resTimestamp');
  const resPersistenceStatus = document.getElementById('resPersistenceStatus');
  const extractionWarningsBox = document.getElementById('extractionWarningsBox');
  const extractionWarningsText = document.getElementById('extractionWarningsText');

  const statTotalCount = document.getElementById('statTotalCount');
  const statPassCount = document.getElementById('statPassCount');
  const statFailCount = document.getElementById('statFailCount');
  const statNaCount = document.getElementById('statNaCount');
  const validationList = document.getElementById('validationList');

  const entityFieldsGrid = document.getElementById('entityFieldsGrid');
  const financialFieldsGrid = document.getElementById('financialFieldsGrid');
  const lineItemsContainer = document.getElementById('lineItemsContainer');
  const lineItemCountPill = document.getElementById('lineItemCountPill');
  const lineItemsTableHead = document.getElementById('lineItemsTableHead');
  const lineItemsTableBody = document.getElementById('lineItemsTableBody');
  const additionalFieldsContainer = document.getElementById('additionalFieldsContainer');
  const additionalFieldsGrid = document.getElementById('additionalFieldsGrid');

  const evidenceSection = document.getElementById('evidenceSection');
  const evidenceContent = document.getElementById('evidenceContent');
  const rawJsonCode = document.getElementById('rawJsonCode');
  const btnCopyJson = document.getElementById('btnCopyJson');
  const btnCopyJsonText = document.getElementById('btnCopyJsonText');

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

  /**
   * Preserves raw values:
   * 1. Actual numbers: formatted with locale commas & up to 4 decimals
   * 2. Strings: returned unmodified (never call parseFloat on IDs, dates, addresses)
   * 3. Null/undefined: returned as null
   */
  function formatFinancialValue(val) {
    if (val === null || val === undefined) return null;
    if (typeof val === 'number') {
      return val.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 4 });
    }
    return String(val);
  }

  function formatKeyTitle(key) {
    if (!key) return '—';
    return String(key)
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  }

  function showAlert(message, type = 'danger') {
    if (!alertContainer) return;
    alertContainer.className = `alert-container alert alert-${type}`;
    const icon = type === 'success' ? '✓' : type === 'warning' ? '⚠' : 'ℹ';
    alertContainer.innerHTML = `
      <div class="alert-content">
        <span class="alert-icon">${icon}</span>
        <div class="alert-text">${escapeHtml(message)}</div>
      </div>
      <button type="button" class="alert-dismiss" aria-label="Close notification">&times;</button>
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
  // Backend Health Check Service
  // =========================================================================
  async function checkBackendHealth() {
    if (statusDot) statusDot.className = 'status-dot status-unknown';
    if (statusText) statusText.textContent = 'Verifying API...';

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });

      if (response.ok) {
        const data = await response.json();
        if (data.status === 'healthy') {
          if (statusDot) statusDot.className = 'status-dot status-healthy';
          if (statusText) statusText.textContent = 'API Connected · Healthy';
          return;
        }
      }
      if (statusDot) statusDot.className = 'status-dot status-unhealthy';
      if (statusText) statusText.textContent = 'API Degraded';
    } catch (err) {
      if (statusDot) statusDot.className = 'status-dot status-unhealthy';
      if (statusText) statusText.textContent = 'API Offline';
    }
  }

  // =========================================================================
  // Document Type Selector (Accessible Keyboard & Click)
  // =========================================================================
  function initDocTypeSelector() {
    if (!docTypeGrid) return;
    const cards = Array.from(docTypeGrid.querySelectorAll('.doc-type-card'));

    function selectCard(card) {
      cards.forEach(c => {
        c.classList.remove('active');
        c.setAttribute('aria-checked', 'false');
      });
      card.classList.add('active');
      card.setAttribute('aria-checked', 'true');

      const radio = card.querySelector('input[type="radio"]');
      if (radio) {
        radio.checked = true;
        currentDocType = radio.value;
      }
    }

    cards.forEach((card, index) => {
      card.addEventListener('click', () => selectCard(card));

      card.addEventListener('keydown', (e) => {
        if (e.key === ' ' || e.key === 'Enter') {
          e.preventDefault();
          selectCard(card);
        } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
          e.preventDefault();
          const nextIndex = (index + 1) % cards.length;
          cards[nextIndex].focus();
          selectCard(cards[nextIndex]);
        } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
          e.preventDefault();
          const prevIndex = (index - 1 + cards.length) % cards.length;
          cards[prevIndex].focus();
          selectCard(cards[prevIndex]);
        }
      });
    });
  }

  // =========================================================================
  // File Upload & Drag-and-Drop Handling
  // =========================================================================
  function getFileExtension(name) {
    if (!name || !name.includes('.')) return '';
    return name.split('.').pop().toLowerCase();
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
    const valResult = validateFile(file);
    if (!valResult.valid) {
      showAlert(valResult.error, 'danger');
      clearFile();
      return;
    }

    hideAlert();
    currentFile = file;
    const ext = getFileExtension(file.name).toUpperCase();

    if (fileBadge) fileBadge.textContent = ext;
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatBytes(file.size);

    if (fileInfoBox) fileInfoBox.classList.remove('hidden');
    if (dropZone) dropZone.classList.add('has-file');
    if (btnProcess) btnProcess.disabled = false;
  }

  function clearFile() {
    currentFile = null;
    if (fileInput) fileInput.value = '';
    if (fileInfoBox) fileInfoBox.classList.add('hidden');
    if (dropZone) dropZone.classList.remove('has-file');
    if (btnProcess) btnProcess.disabled = true;
  }

  function initFileUpload() {
    if (!dropZone || !fileInput) return;

    // Dropzone click triggers input
    dropZone.addEventListener('click', (e) => {
      if (e.target.closest('#fileInfoBox')) return;
      fileInput.click();
    });

    // Dropzone keyboard activation
    dropZone.addEventListener('keydown', (e) => {
      if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        fileInput.click();
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        setFile(e.target.files[0]);
      }
    });

    if (btnRemoveFile) {
      btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
      });
    }

    // Drag-and-drop animations
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
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
  // 5-Stage Processing Pipeline UX
  // =========================================================================
  const PIPELINE_STAGES = [
    { num: 1, text: 'Stage 1/5: Validating file headers, MIME-type, and PDF structure...' },
    { num: 2, text: 'Stage 2/5: Executing text extraction / PyMuPDF page rasterization...' },
    { num: 3, text: 'Stage 3/5: Calling Gemini structured multimodal extraction...' },
    { num: 4, text: 'Stage 4/5: Running zero-tolerance deterministic mathematical validation...' },
    { num: 5, text: 'Stage 5/5: Persisting document audit record in PostgreSQL...' },
  ];

  function startPipelineAnimation() {
    isProcessing = true;
    if (btnProcess) {
      btnProcess.disabled = true;
      if (btnProcessText) btnProcessText.textContent = 'Processing Pipeline...';
    }
    if (emptyDashboardState) emptyDashboardState.classList.add('hidden');
    if (loadingSection) loadingSection.classList.remove('hidden');
    if (resultsWrapper) resultsWrapper.classList.add('hidden');
    hideAlert();

    let stepIdx = 0;
    function updateStage() {
      const stage = PIPELINE_STAGES[stepIdx];
      if (loadingStageText) loadingStageText.textContent = stage.text;

      // Update active pipeline node visual highlight
      for (let i = 1; i <= 5; i++) {
        const stepEl = document.getElementById(`pipeStep${i}`);
        if (stepEl) {
          if (i === stage.num) {
            stepEl.classList.add('step-active');
          } else {
            stepEl.classList.remove('step-active');
          }
        }
      }
      stepIdx = (stepIdx + 1) % PIPELINE_STAGES.length;
    }

    updateStage();
    pipelineInterval = setInterval(updateStage, 2600);
  }

  function stopPipelineAnimation() {
    isProcessing = false;
    if (btnProcess) {
      btnProcess.disabled = !currentFile;
      if (btnProcessText) btnProcessText.textContent = 'Process Document';
    }
    if (loadingSection) loadingSection.classList.add('hidden');
    if (pipelineInterval) {
      clearInterval(pipelineInterval);
      pipelineInterval = null;
    }
  }

  // =========================================================================
  // Document Processing Submission
  // =========================================================================
  async function processDocument() {
    if (!currentFile || isProcessing) return;

    startPipelineAnimation();

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
        throw new Error(`Server returned HTTP ${response.status} with unparseable body.`);
      }

      stopPipelineAnimation();

      if (response.status === 200 || response.status === 400) {
        displayResults(data, response.status);
      } else {
        showAlert(`Server error (${response.status}): ${data.detail || 'Internal processing error'}`, 'danger');
      }
    } catch (err) {
      stopPipelineAnimation();
      showAlert(`Network/Connection Error: ${err.message}. Ensure backend is running on ${API_BASE_URL}.`, 'danger');
    }
  }

  // =========================================================================
  // Dynamic Results Rendering
  // =========================================================================
  function displayResults(data, httpStatus) {
    if (!data) return;

    if (resultsWrapper) {
      resultsWrapper.classList.remove('hidden');
      resultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    renderOverview(data, httpStatus);
    renderValidations(data.validations);
    renderExtractedData(data.extracted_data);
    renderEvidence(data);
    renderRawJson(data);

    // Contextual alert banner
    if (data.processing_status === 'COMPLETED') {
      showAlert('All deterministic mathematical validations passed successfully. Record persisted to PostgreSQL.', 'success');
    } else if (data.processing_status === 'VALIDATION_FAILED') {
      showAlert('Extraction succeeded, but deterministic validation flagged mathematical variances.', 'warning');
    } else if (data.processing_status === 'EXTRACTION_FAILED') {
      showAlert('Failed to extract required financial fields from the document.', 'danger');
    }
  }

  function renderOverview(data, httpStatus) {
    if (resDocName) resDocName.textContent = data.document_name || (currentFile ? currentFile.name : '—');
    if (resDocType) resDocType.textContent = formatKeyTitle(data.document_type || currentDocType);

    const fileVal = data.file_validation || {};
    const meta = data.metadata || {};
    const ext = fileVal.file_type || (currentFile ? getFileExtension(currentFile.name) : 'FILE');
    const pages = fileVal.page_count || meta.pages_processed || meta.page_count || 1;

    if (resFilePageSpec) {
      resFilePageSpec.innerHTML = `
        <span class="file-badge">${escapeHtml(ext.toUpperCase())}</span>
        <span>${pages} page${pages === 1 ? '' : 's'}</span>
      `;
    }

    if (resDocId) {
      resDocId.textContent = (data.id !== null && data.id !== undefined) ? `#${data.id}` : 'Not Persisted';
    }

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
    if (resTimestamp) resTimestamp.textContent = formattedDate;

    if (resPersistenceStatus) {
      const persisted = data.id !== null && data.id !== undefined;
      resPersistenceStatus.innerHTML = persisted
        ? `<span class="badge badge-pass" style="font-size: 0.75rem;">PostgreSQL Verified</span>`
        : `<span class="badge badge-na" style="font-size: 0.75rem;">Not Persisted</span>`;
    }

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

    if (statusBadgeContainer) {
      statusBadgeContainer.innerHTML = `
        <span class="badge ${badgeClass}" style="font-size: 0.85rem; padding: 0.45rem 0.95rem;">
          ${icon} ${escapeHtml(status)}
        </span>
      `;
    }

    // Extraction Warnings Box
    const warnings = meta.extraction_warnings;
    if (warnings && extractionWarningsBox && extractionWarningsText) {
      extractionWarningsText.textContent = Array.isArray(warnings) ? warnings.join('; ') : String(warnings);
      extractionWarningsBox.classList.remove('hidden');
    } else if (extractionWarningsBox) {
      extractionWarningsBox.classList.add('hidden');
    }
  }

  function renderValidations(validations) {
    if (!validations || !validations.checks || validations.checks.length === 0) {
      if (statTotalCount) statTotalCount.textContent = '0';
      if (statPassCount) statPassCount.textContent = '0';
      if (statFailCount) statFailCount.textContent = '0';
      if (statNaCount) statNaCount.textContent = '0';
      if (validationList) {
        validationList.innerHTML = `
          <div class="empty-state" style="padding: 1.5rem; text-align: center; color: var(--text-muted);">
            No deterministic mathematical validations were executed for this document.
          </div>
        `;
      }
      return;
    }

    const checks = validations.checks;
    const summary = validations.summary || {};
    const passed = summary.passed_checks ?? checks.filter(c => c.status === 'PASS').length;
    const failed = summary.failed_checks ?? checks.filter(c => c.status === 'FAILED').length;
    const na = summary.not_applicable_checks ?? checks.filter(c => c.status === 'NOT_APPLICABLE').length;

    if (statTotalCount) statTotalCount.textContent = checks.length;
    if (statPassCount) statPassCount.textContent = passed;
    if (statFailCount) statFailCount.textContent = failed;
    if (statNaCount) statNaCount.textContent = na;

    let html = '';
    checks.forEach((chk, idx) => {
      const status = chk.status || 'NOT_APPLICABLE';
      let cardClass = 'status-not_applicable';
      let badgeTag = '<span class="badge badge-na">N/A</span>';
      let filterTag = 'na';

      if (status === 'PASS') {
        cardClass = 'status-pass';
        badgeTag = '<span class="badge badge-pass">✓ PASS</span>';
        filterTag = 'pass';
      } else if (status === 'FAILED') {
        cardClass = 'status-failed';
        badgeTag = '<span class="badge badge-fail">✗ FAILED</span>';
        filterTag = 'failed';
      }

      // Metric items
      let metricsHtml = '';
      if (chk.calculated_value !== null && chk.calculated_value !== undefined) {
        metricsHtml += `
          <div class="val-metric">
            <span class="val-metric-label">Calculated Value</span>
            <span class="val-metric-val">${formatFinancialValue(chk.calculated_value)}</span>
          </div>
        `;
      }
      if (chk.reported_value !== null && chk.reported_value !== undefined) {
        metricsHtml += `
          <div class="val-metric">
            <span class="val-metric-label">Reported Value</span>
            <span class="val-metric-val">${formatFinancialValue(chk.reported_value)}</span>
          </div>
        `;
      }
      if (chk.variance !== null && chk.variance !== undefined) {
        const isZero = Math.abs(chk.variance) < 0.0001;
        metricsHtml += `
          <div class="val-metric">
            <span class="val-metric-label">Variance</span>
            <span class="val-metric-val ${isZero ? 'var-zero' : 'var-diff'}">
              ${formatFinancialValue(chk.variance)}
            </span>
          </div>
        `;
      }
      if (chk.tolerance !== null && chk.tolerance !== undefined) {
        metricsHtml += `
          <div class="val-metric">
            <span class="val-metric-label">Tolerance</span>
            <span class="val-metric-val">&plusmn;${chk.tolerance}</span>
          </div>
        `;
      }

      // Missing fields
      let missingTagsHtml = '';
      if (chk.missing_fields && chk.missing_fields.length > 0) {
        missingTagsHtml = `
          <div class="val-metric" style="grid-column: 1 / -1;">
            <span class="val-metric-label">Missing Required Inputs</span>
            <div>
              ${chk.missing_fields.map(f => `<span class="val-missing-tag">${escapeHtml(f)}</span>`).join('')}
            </div>
          </div>
        `;
      }

      // Message styling
      let msgClass = 'msg-na';
      if (status === 'PASS') msgClass = 'msg-pass';
      else if (status === 'FAILED') msgClass = 'msg-fail';

      html += `
        <article class="validation-card-item ${cardClass}" data-filter-tag="${filterTag}">
          <div class="val-header">
            <div class="val-title-group">
              <span class="val-rule-code">${escapeHtml(chk.rule_name || `Check #${idx + 1}`)}</span>
            </div>
            ${badgeTag}
          </div>

          ${chk.formula ? `
            <div class="val-formula-row">
              <span class="val-formula-badge">Formula:</span>
              <code class="val-formula">${escapeHtml(chk.formula)}</code>
            </div>
          ` : ''}

          ${(metricsHtml || missingTagsHtml) ? `
            <div class="val-details-row">
              ${metricsHtml}
              ${missingTagsHtml}
            </div>
          ` : ''}

          ${chk.message ? `
            <div class="val-message-box ${msgClass}">
              ${escapeHtml(chk.message)}
            </div>
          ` : ''}
        </article>
      `;
    });

    if (validationList) {
      validationList.innerHTML = html;
      applyValidationFilter(activeValidationFilter);
    }
  }

  function applyValidationFilter(filter) {
    activeValidationFilter = filter;
    if (!validationList) return;

    const cards = validationList.querySelectorAll('.validation-card-item');
    cards.forEach(card => {
      const tag = card.getAttribute('data-filter-tag');
      if (filter === 'all' || tag === filter) {
        card.style.display = 'flex';
      } else {
        card.style.display = 'none';
      }
    });

    // Update filter tabs active state
    const tabs = document.querySelectorAll('.filter-tab');
    tabs.forEach(tab => {
      const tabFilter = tab.getAttribute('data-filter');
      if (tabFilter === filter) {
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
      } else {
        tab.classList.remove('active');
        tab.setAttribute('aria-selected', 'false');
      }
    });
  }

  function initValidationFilterTabs() {
    const tabs = document.querySelectorAll('.filter-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const filter = tab.getAttribute('data-filter');
        applyValidationFilter(filter);
      });
    });
  }

  // =========================================================================
  // Extracted Financial Data (Separate Metadata vs Financials)
  // =========================================================================
  function renderExtractedData(extractedData) {
    if (!extractedData || Object.keys(extractedData).length === 0) {
      if (entityFieldsGrid) {
        entityFieldsGrid.innerHTML = `<p class="text-muted" style="grid-column: 1 / -1;">No structured data extracted.</p>`;
      }
      if (financialFieldsGrid) {
        financialFieldsGrid.innerHTML = `<p class="text-muted" style="grid-column: 1 / -1;">No financial fields extracted.</p>`;
      }
      if (lineItemsContainer) lineItemsContainer.classList.add('hidden');
      if (additionalFieldsContainer) additionalFieldsContainer.classList.add('hidden');
      return;
    }

    const entityFields = {};
    const financialFields = {};
    const complexFields = {};
    let lineItems = null;

    for (const [key, value] of Object.entries(extractedData)) {
      if (key === 'line_items' && Array.isArray(value)) {
        lineItems = value;
      } else if (Array.isArray(value) || (typeof value === 'object' && value !== null)) {
        complexFields[key] = value;
      } else if (ENTITY_METADATA_KEYS.has(key)) {
        entityFields[key] = value;
      } else {
        financialFields[key] = value;
      }
    }

    // Render Entity Metadata Fields
    if (entityFieldsGrid) {
      let entityHtml = '';
      for (const [k, v] of Object.entries(entityFields)) {
        const isNull = v === null || v === undefined || v === '';
        // CRITICAL: Strings (dates, IDs, tax IDs, addresses) must NEVER be formatted as currency numbers
        const display = isNull ? '<span class="null-badge">null</span>' : escapeHtml(String(v));
        entityHtml += `
          <div class="field-card">
            <span class="field-key">${escapeHtml(formatKeyTitle(k))}</span>
            <span class="field-val" style="white-space: pre-wrap;">${display}</span>
          </div>
        `;
      }
      entityFieldsGrid.innerHTML = entityHtml || '<p class="text-muted" style="grid-column: 1 / -1;">No statement metadata present.</p>';
    }

    // Render Financial Metric Fields
    if (financialFieldsGrid) {
      let finHtml = '';
      for (const [k, v] of Object.entries(financialFields)) {
        const isNull = v === null || v === undefined || v === '';
        let display = '';
        if (isNull) {
          display = '<span class="null-badge">null</span>';
        } else if (typeof v === 'number') {
          display = `<span class="financial-num">${escapeHtml(formatFinancialValue(v))}</span>`;
        } else {
          display = escapeHtml(String(v));
        }
        finHtml += `
          <div class="field-card">
            <span class="field-key">${escapeHtml(formatKeyTitle(k))}</span>
            <span class="field-val">${display}</span>
          </div>
        `;
      }
      financialFieldsGrid.innerHTML = finHtml || '<p class="text-muted" style="grid-column: 1 / -1;">No core totals extracted.</p>';
    }

    // Render Line Items Table
    if (lineItems && lineItems.length > 0 && lineItemsContainer && lineItemsTableHead && lineItemsTableBody) {
      lineItemsContainer.classList.remove('hidden');
      if (lineItemCountPill) lineItemCountPill.textContent = `${lineItems.length} item${lineItems.length === 1 ? '' : 's'}`;

      const sample = lineItems[0] || {};
      const keys = Object.keys(sample);

      lineItemsTableHead.innerHTML = `
        <th style="width: 48px;">#</th>
        ${keys.map(k => {
          const isNum = typeof sample[k] === 'number';
          return `<th class="${isNum ? 'num-cell' : ''}">${escapeHtml(formatKeyTitle(k))}</th>`;
        }).join('')}
      `;

      let rowsHtml = '';
      lineItems.forEach((item, idx) => {
        rowsHtml += `
          <tr>
            <td><strong>${idx + 1}</strong></td>
            ${keys.map(k => {
              const val = item[k];
              const isNull = val === null || val === undefined || val === '';
              if (isNull) return `<td class="text-muted">—</td>`;
              if (typeof val === 'number') {
                return `<td class="num-cell">${escapeHtml(formatFinancialValue(val))}</td>`;
              }
              return `<td>${escapeHtml(String(val))}</td>`;
            }).join('')}
          </tr>
        `;
      });
      lineItemsTableBody.innerHTML = rowsHtml;
    } else if (lineItemsContainer) {
      lineItemsContainer.classList.add('hidden');
    }

    // Render Additional Fields / Disclosures
    const complexEntries = Object.entries(complexFields);
    if (complexEntries.length > 0 && additionalFieldsContainer && additionalFieldsGrid) {
      additionalFieldsContainer.classList.remove('hidden');
      let addHtml = '';
      for (const [k, v] of complexEntries) {
        addHtml += `
          <div class="field-card" style="grid-column: span 2;">
            <span class="field-key">${escapeHtml(formatKeyTitle(k))}</span>
            <pre style="font-size: 0.8rem; background: #fafbfd; padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border-default); overflow-x: auto; font-family: var(--font-mono); color: var(--text-main);"><code>${escapeHtml(JSON.stringify(v, null, 2))}</code></pre>
          </div>
        `;
      }
      additionalFieldsGrid.innerHTML = addHtml;
    } else if (additionalFieldsContainer) {
      additionalFieldsContainer.classList.add('hidden');
    }
  }

  // =========================================================================
  // Extraction Evidence & Source Metadata
  // =========================================================================
  function renderEvidence(data) {
    if (!evidenceSection || !evidenceContent) return;
    const meta = data.metadata || {};
    const fileVal = data.file_validation || {};

    let html = `
      <div class="overview-grid" style="margin-bottom: 1rem;">
        <div class="overview-item">
          <span class="overview-label">Original MIME Type</span>
          <span class="overview-value mono-val">${escapeHtml(fileVal.mime_type || '—')}</span>
        </div>
        <div class="overview-item">
          <span class="overview-label">File SHA-256 Checksum</span>
          <span class="overview-value mono-val text-break" style="font-size: 0.75rem;">${escapeHtml(fileVal.file_hash || '—')}</span>
        </div>
        <div class="overview-item">
          <span class="overview-label">LLM Extractor Model</span>
          <span class="overview-value mono-val">${escapeHtml(meta.model_name || 'Gemini 3.5 / 3.6 Multimodal')}</span>
        </div>
        <div class="overview-item">
          <span class="overview-label">Rasterization Required</span>
          <span class="overview-value">${meta.rasterized ? 'Yes (PyMuPDF 300 DPI)' : 'No (Digital PDF/Image)'}</span>
        </div>
      </div>
    `;

    if (meta.source_snippets && Array.isArray(meta.source_snippets) && meta.source_snippets.length > 0) {
      html += `
        <h4 style="font-size: 0.85rem; font-weight: 700; margin: 1rem 0 0.5rem;">Verified OCR Source Snippets:</h4>
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
          ${meta.source_snippets.map(s => `
            <blockquote style="background: #ffffff; border-left: 3px solid var(--brand-600); padding: 0.5rem 0.85rem; font-size: 0.8rem; border-radius: 4px; border: 1px solid var(--border-default); border-left-width: 3px;">
              ${escapeHtml(s)}
            </blockquote>
          `).join('')}
        </div>
      `;
    }

    evidenceContent.innerHTML = html;
  }

  // =========================================================================
  // Raw JSON Response & Clipboard Copy
  // =========================================================================
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
          if (btnCopyJsonText) btnCopyJsonText.textContent = 'Copied to Clipboard!';
          btnCopyJson.classList.add('btn-primary');
          btnCopyJson.classList.remove('btn-outline');
          setTimeout(() => {
            if (btnCopyJsonText) btnCopyJsonText.textContent = 'Copy JSON';
            btnCopyJson.classList.remove('btn-primary');
            btnCopyJson.classList.add('btn-outline');
          }, 2200);
        } catch (err) {
          const range = document.createRange();
          range.selectNodeContents(rawJsonCode);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
          document.execCommand('copy');
          if (btnCopyJsonText) btnCopyJsonText.textContent = 'Copied!';
          setTimeout(() => {
            if (btnCopyJsonText) btnCopyJsonText.textContent = 'Copy JSON';
          }, 2200);
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
    initValidationFilterTabs();

    if (btnProcess) {
      btnProcess.addEventListener('click', processDocument);
    }

    if (btnRefreshHealth) {
      btnRefreshHealth.addEventListener('click', (e) => {
        e.stopPropagation();
        checkBackendHealth();
      });
    }

    // Initial Health Check and Periodic Polling
    checkBackendHealth();
    setInterval(checkBackendHealth, 45000);
  }

  // Expose displayResults on window for test harnesses
  window.displayResults = displayResults;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
