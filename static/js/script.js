/**
 * ECG AI Clinical Screening Web Application - Client-Side Engine
 * Clinical-grade biomedical research interface logic
 */

let ecgChart = null;
let breakdownChart = null;
let currentWaveformData = null;

document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    initSampleButtons();
    initFilterTable();
    initMobileNav();
    if (window.lucide) {
        lucide.createIcons();
    }
});

// -------------------------------------------------------------
// Mobile Navigation Toggle
// -------------------------------------------------------------
function initMobileNav() {
    const toggleBtn = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');
    if (!toggleBtn || !navLinks) return;

    toggleBtn.addEventListener('click', () => {
        navLinks.classList.toggle('open');
        const isOpen = navLinks.classList.contains('open');
        toggleBtn.setAttribute('aria-expanded', isOpen);
    });
}

// -------------------------------------------------------------
// Drag & Drop File Upload Handling
// -------------------------------------------------------------
function initDropzone() {
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('ecgFileInput');
    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// -------------------------------------------------------------
// Sample Buttons Trigger
// -------------------------------------------------------------
function initSampleButtons() {
    const sampleButtons = document.querySelectorAll('.sample-btn');
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const sampleType = e.currentTarget.getAttribute('data-sample');
            if (sampleType) {
                fetchSampleAnalysis(sampleType);
            }
        });
    });
}

function fetchSampleAnalysis(sampleType) {
    showLoading(true, `Conditioning clinical sample (${sampleType.toUpperCase()}) and executing 1D CNN inference...`);
    hideError();

    fetch(`/api/sample/${sampleType}`)
        .then(res => res.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'Sample analysis failed.');
            }
        })
        .catch(err => {
            showLoading(false);
            showError('Network communication error: ' + err.message);
        });
}

// -------------------------------------------------------------
// File Upload AJAX
// -------------------------------------------------------------
function handleFileUpload(file) {
    if (!file) return;

    const allowed = ['.csv', '.txt', '.dat', '.npy'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
        showError(`Unsupported file format '${ext}'. Please upload a valid .csv, .txt, .dat, or .npy file.`);
        return;
    }

    const formData = new FormData();
    formData.append('ecg_file', file);

    showLoading(true, `Ingesting ${file.name}, applying zero-phase filter and running neural inference...`);
    hideError();

    fetch('/api/analyze', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        showLoading(false);
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'ECG analysis could not be completed.');
        }
    })
    .catch(err => {
        showLoading(false);
        showError('Network communication error: ' + err.message);
    });
}

// -------------------------------------------------------------
// Render Clinical Screening Results
// -------------------------------------------------------------
function displayResults(data) {
    const resultsPanel = document.getElementById('resultsPanel');
    if (!resultsPanel) return;

    resultsPanel.style.display = 'block';
    resultsPanel.scrollIntoView({ behavior: 'smooth' });

    // Primary Prediction & Status
    const predElem = document.getElementById('resPrediction');
    const badgeElem = document.getElementById('resBadge');
    const statusBox = document.getElementById('resStatusBox');
    const statusText = document.getElementById('resStatusText');

    predElem.textContent = data.prediction;
    statusText.textContent = data.status;

    if (data.is_abnormal) {
        badgeElem.className = 'status-badge status-abnormal';
        badgeElem.textContent = 'Abnormality Detected';
        statusBox.className = 'status-callout status-abnormal';
    } else {
        badgeElem.className = 'status-badge status-normal';
        badgeElem.textContent = 'Normal Pattern';
        statusBox.className = 'status-callout status-normal';
    }

    // Metric Cards (Tabular Numerals)
    document.getElementById('resConfidence').textContent = `${data.confidence.toFixed(1)}%`;
    document.getElementById('resHeartRate').textContent = data.heart_rate_bpm > 0 ? `${data.heart_rate_bpm} BPM` : 'N/A';
    document.getElementById('resTotalBeats').textContent = `${data.total_beats_analyzed} Beats`;
    document.getElementById('resDuration').textContent = `${data.recording_duration_sec}s (${data.sampling_rate_hz} Hz)`;

    // Render Waveform Canvas
    currentWaveformData = data.waveform;
    renderWaveformChart(data.waveform, true);

    // Render Breakdown Chart
    if (data.breakdown) {
        renderBreakdownChart(data.breakdown);
    }

    if (window.lucide) {
        lucide.createIcons();
    }
}

// -------------------------------------------------------------
// Chart.js ECG Waveform Visualization
// -------------------------------------------------------------
function renderWaveformChart(waveform, showClean = true) {
    const canvas = document.getElementById('ecgWaveformChart');
    if (!canvas) return;

    if (ecgChart) {
        ecgChart.destroy();
    }

    const labels = waveform.time;
    const signalData = showClean ? waveform.clean : waveform.raw;

    const peakPoints = [];
    if (waveform.peak_times && waveform.peak_voltages) {
        for (let i = 0; i < waveform.peak_times.length; i++) {
            peakPoints.push({
                x: waveform.peak_times[i],
                y: waveform.peak_voltages[i]
            });
        }
    }

    ecgChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: showClean ? 'Conditioned Lead MLII (0.5–45 Hz Bandpass)' : 'Raw Unfiltered Signal',
                    data: signalData,
                    borderColor: showClean ? '#0E7C86' : '#9CA3AF',
                    borderWidth: showClean ? 1.5 : 1.2,
                    pointRadius: 0,
                    tension: 0.05
                },
                {
                    label: 'Detected R-Peaks (QRS Centroids)',
                    data: peakPoints,
                    type: 'scatter',
                    backgroundColor: '#B4232C',
                    borderColor: '#FFFFFF',
                    borderWidth: 1.5,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#4B5563',
                        font: { size: 12, family: "'Inter', sans-serif", weight: '500' },
                        boxWidth: 14,
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: '#0F1B2D',
                    titleColor: '#F8FAFC',
                    bodyColor: '#F8FAFC',
                    titleFont: { family: "'Inter', sans-serif", size: 12 },
                    bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
                    padding: 10,
                    cornerRadius: 6,
                    callbacks: {
                        label: (ctx) => `Time: ${ctx.parsed.x.toFixed(3)}s | Amplitude: ${ctx.parsed.y.toFixed(3)}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Time (seconds)',
                        color: '#4B5563',
                        font: { family: "'Inter', sans-serif", size: 12, weight: '500' }
                    },
                    ticks: {
                        color: '#6B7280',
                        font: { family: "'JetBrains Mono', monospace", size: 11 }
                    },
                    grid: { color: '#E6E8EC' }
                },
                y: {
                    title: {
                        display: true,
                        text: showClean ? 'Normalized Voltage (Z-Score)' : 'Voltage (mV)',
                        color: '#4B5563',
                        font: { family: "'Inter', sans-serif", size: 12, weight: '500' }
                    },
                    ticks: {
                        color: '#6B7280',
                        font: { family: "'JetBrains Mono', monospace", size: 11 }
                    },
                    grid: { color: '#E6E8EC' }
                }
            }
        }
    });
}

function toggleWaveformType(type) {
    if (!currentWaveformData) return;
    renderWaveformChart(currentWaveformData, type === 'clean');
}

// -------------------------------------------------------------
// Chart.js Beat Breakdown Donut
// -------------------------------------------------------------
function renderBreakdownChart(breakdown) {
    const canvas = document.getElementById('breakdownChart');
    if (!canvas) return;

    if (breakdownChart) {
        breakdownChart.destroy();
    }

    const labels = Object.keys(breakdown);
    const counts = labels.map(k => breakdown[k].count);
    const colors = ['#0E7C86', '#B4232C', '#B45309', '#64748B'];

    breakdownChart = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#FFFFFF'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#4B5563',
                        font: { size: 11, family: "'Inter', sans-serif", weight: '500' },
                        boxWidth: 12,
                        padding: 12
                    }
                }
            }
        }
    });
}

// -------------------------------------------------------------
// History Table Filter & Delete
// -------------------------------------------------------------
function initFilterTable() {
    const filterInput = document.getElementById('historySearchInput');
    const table = document.getElementById('historyTable');
    if (!filterInput || !table) return;

    filterInput.addEventListener('keyup', () => {
        const query = filterInput.value.toLowerCase();
        const rows = table.getElementsByTagName('tr');

        for (let i = 1; i < rows.length; i++) {
            const rowText = rows[i].textContent.toLowerCase();
            rows[i].style.display = rowText.includes(query) ? '' : 'none';
        }
    });
}

function deleteHistoryRecord(recordId) {
    if (!confirm(`Are you sure you want to delete screening record #${recordId}?`)) return;

    fetch(`/api/history/delete/${recordId}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                const row = document.getElementById(`record-row-${recordId}`);
                if (row) row.remove();
            }
        });
}

// -------------------------------------------------------------
// UI Loading & Error Utilities
// -------------------------------------------------------------
function showLoading(show, message = 'Processing...') {
    const loadingElem = document.getElementById('loadingState');
    const loadingMsg = document.getElementById('loadingMessage');
    if (!loadingElem) return;

    if (show) {
        if (loadingMsg) loadingMsg.textContent = message;
        loadingElem.style.display = 'flex';
    } else {
        loadingElem.style.display = 'none';
    }
}

function showError(msg) {
    const errBox = document.getElementById('errorBanner');
    const errMsg = document.getElementById('errorMessage');
    if (errBox && errMsg) {
        errMsg.textContent = msg;
        errBox.style.display = 'block';
    }
}

function hideError() {
    const errBox = document.getElementById('errorBanner');
    if (errBox) errBox.style.display = 'none';
}
