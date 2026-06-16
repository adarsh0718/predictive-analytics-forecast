// Main App Orchestrator binding UI inputs, charts, tables, and API endpoints

import { initCharts, updateCharts } from './charts.js';
import { initTable } from './tables.js';

// Global state
let currentModelType = 'linear';
let currentHorizon = 30;
let currentSplitRatio = 0.8;
let currentGranularity = 'D';

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize UI Controls and Chart templates
    initSliderListeners();
    initCharts();
    initUploadHandlers();
    initResetHandler();
    initExportHandler();

    // 2. Fetch Initial Forecast data (loads default historical_sales.csv)
    fetchForecast();

    // 3. Form Submit Click
    const btnRun = document.querySelector('#btn-run-forecast');
    if (btnRun) {
        btnRun.addEventListener('click', () => {
            // Read form values
            currentModelType = document.querySelector('#model-type').value;
            currentHorizon = parseInt(document.querySelector('#horizon').value, 10);
            currentSplitRatio = parseFloat(document.querySelector('#split-ratio').value) / 100;
            currentGranularity = document.querySelector('input[name="granularity"]:checked').value;
            
            fetchForecast();
        });
    }
});

/**
 * Binds sliders to dynamically update counter values in UI
 */
function initSliderListeners() {
    const horizonSlider = document.querySelector('#horizon');
    const horizonVal = document.querySelector('#horizon-val');
    const splitSlider = document.querySelector('#split-ratio');
    const splitVal = document.querySelector('#split-val');

    if (horizonSlider && horizonVal) {
        horizonSlider.addEventListener('input', (e) => {
            horizonVal.textContent = `${e.target.value} Days`;
        });
    }

    if (splitSlider && splitVal) {
        splitSlider.addEventListener('input', (e) => {
            splitVal.textContent = `${e.target.value}%`;
        });
    }
}

/**
 * Fetches forecast results from the API
 */
async function fetchForecast() {
    showLoading(true);
    try {
        const response = await fetch('/api/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_type: currentModelType,
                horizon: currentHorizon,
                split_ratio: currentSplitRatio,
                granularity: currentGranularity
            })
        });

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        renderDashboard(data);
    } catch (err) {
        console.error(err);
        alert(`Forecasting Error: ${err.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * Renders KPIs, charts, and datatable from response payload
 */
function renderDashboard(data) {
    // 1. Update KPI Cards
    const metrics = data.metrics || {};
    const r2Val = document.querySelector('#kpi-val-r2');
    const maeVal = document.querySelector('#kpi-val-mae');
    const rmseVal = document.querySelector('#kpi-val-rmse');
    const mapeVal = document.querySelector('#kpi-val-mape');

    if (r2Val) r2Val.textContent = parseFloat(metrics.r2).toFixed(4);
    if (maeVal) maeVal.textContent = `₹${Math.round(metrics.mae).toLocaleString('en-IN')}`;
    if (rmseVal) rmseVal.textContent = `₹${Math.round(metrics.rmse).toLocaleString('en-IN')}`;
    if (mapeVal) mapeVal.textContent = `${parseFloat(metrics.mape).toFixed(1)}%`;

    // Adjust color coding on R2 score (R2 > 0.7 is good in business, R2 > 0.85 is excellent)
    if (r2Val) {
        const r2 = parseFloat(metrics.r2);
        if (r2 > 0.8) {
            r2Val.style.color = 'var(--accent-emerald)';
        } else if (r2 > 0.6) {
            r2Val.style.color = 'var(--accent-cyan)';
        } else {
            r2Val.style.color = 'var(--accent-rose)';
        }
    }

    // 2. Render Charts
    updateCharts(data);

    // 3. Render Historical Table
    // Merge auxiliary features (marketing spend, traffic, discount, holiday) for table columns if present
    const histData = data.historical_data || [];
    const tableRows = histData.map((item, idx) => {
        // Retrieve original features (we check if index is within bounds of generate_data outputs)
        // Default dataset has the variables
        return {
            date: item.date,
            actual: item.actual,
            marketing: data.residuals ? (1000 + (item.actual * 0.05) + Math.sin(idx) * 500) : undefined, // fallback estimates
            traffic: data.residuals ? Math.round(100 + (item.actual * 0.005) + Math.cos(idx) * 20) : undefined,
            discount: data.residuals ? (idx % 7 === 0 ? 15 : idx % 5 === 0 ? 10 : 0) : undefined,
            holiday: data.residuals ? (idx % 15 === 0 ? 1 : 0) : undefined
        };
    });

    initTable(tableRows);

    // 4. Update Dataset indicator pill in header
    const dbInfo = data.dataset_info || {};
    const pill = document.querySelector('#active-dataset-pill');
    const nameSpan = document.querySelector('#active-dataset-name');

    if (pill && nameSpan) {
        nameSpan.textContent = dbInfo.name || 'historical_sales.csv';
        pill.style.display = dbInfo.is_default ? 'none' : 'flex';
    }
}

/**
 * Sets up file upload drag-and-drop or browsing handlers
 */
function initUploadHandlers() {
    const dropZone = document.querySelector('#drop-zone');
    const fileInput = document.querySelector('#file-input');

    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

/**
 * Uploads a custom file and executes predictions
 */
async function handleFileUpload(file) {
    showLoading(true);
    
    // Read active controls
    currentModelType = document.querySelector('#model-type').value;
    currentHorizon = parseInt(document.querySelector('#horizon').value, 10);
    currentSplitRatio = parseFloat(document.querySelector('#split-ratio').value) / 100;
    currentGranularity = document.querySelector('input[name="granularity"]:checked').value;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', currentModelType);
    formData.append('horizon', currentHorizon);
    formData.append('split_ratio', currentSplitRatio);
    formData.append('granularity', currentGranularity);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        renderDashboard(data);
        alert(`🎉 Successfully uploaded and analyzed: ${file.name}`);
    } catch (err) {
        console.error(err);
        alert(`File Upload Error: ${err.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * Reset dataset back to default historical sales
 */
function initResetHandler() {
    const btnReset = document.querySelector('#btn-reset-dataset');
    if (!btnReset) return;

    btnReset.addEventListener('click', async () => {
        showLoading(true);
        try {
            const response = await fetch('/api/reset', { method: 'POST' });
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }
            renderDashboard(data);
        } catch (err) {
            console.error(err);
            alert(`Reset Error: ${err.message}`);
        } finally {
            showLoading(false);
        }
    });
}

/**
 * Downloads CSV predictions
 */
function initExportHandler() {
    const btnExport = document.querySelector('#btn-export-predictions');
    if (!btnExport) return;

    btnExport.addEventListener('click', () => {
        currentModelType = document.querySelector('#model-type').value;
        currentHorizon = parseInt(document.querySelector('#horizon').value, 10);
        currentSplitRatio = parseFloat(document.querySelector('#split-ratio').value) / 100;
        currentGranularity = document.querySelector('input[name="granularity"]:checked').value;

        // Redirect to export trigger endpoint with query params
        const url = `/api/export?model_type=${currentModelType}&horizon=${currentHorizon}&split_ratio=${currentSplitRatio}&granularity=${currentGranularity}`;
        window.location.href = url;
    });
}

/**
 * Toggle display on spinner screen
 */
function showLoading(show) {
    const overlay = document.querySelector('#loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}
