/**
 * Enterprise Hybrid Forecasting Suite
 * Frontend Application & API Communication Layer
 */

// Determine backend API host dynamically
const API_BASE = window.location.port === '8000' 
  ? window.location.origin 
  : 'http://localhost:8000';

// Global application state
const APP_STATE = {
  currentDataset: null,
  forecastResult: null,
  diagnosticsResult: null,
  visibleLayers: {
    actuals: true,
    sarimax: true,
    xgboost: true,
    hybrid: true
  }
};

// DOM Element References
const dom = {
  backendBadge: document.getElementById('backendStatusBadge'),
  backendText: document.getElementById('backendStatusText'),
  csvFileInput: document.getElementById('csvFileInput'),
  datasetStatusIcon: document.getElementById('datasetStatusIcon'),
  btnRunPipeline: document.getElementById('btnRunPipeline'),
  runIcon: document.getElementById('runIcon'),
  runText: document.getElementById('runText'),
  btnExportCSV: document.getElementById('btnExportCSV'),
  
  // Info Banner
  currentFileName: document.getElementById('currentFileName'),
  currentFileStats: document.getElementById('currentFileStats'),
  currentDateRange: document.getElementById('currentDateRange'),
  
  // Selects & Sliders
  selectTargetCol: document.getElementById('selectTargetCol'),
  selectDateCol: document.getElementById('selectDateCol'),
  splitSlider: document.getElementById('splitSlider'),
  trainSplitLabel: document.getElementById('trainSplitLabel'),
  selectHorizon: document.getElementById('selectHorizon'),
  
  // KPI Metrics
  metricHybridRmse: document.getElementById('metricHybridRmse'),
  metricBaselineRmse: document.getElementById('metricBaselineRmse'),
  metricArimaRmse: document.getElementById('metricArimaRmse'),
  metricXgbRmse: document.getElementById('metricXgbRmse'),
  badgeRmseImp: document.getElementById('badgeRmseImp'),
  
  metricHybridMae: document.getElementById('metricHybridMae'),
  metricBaselineMae: document.getElementById('metricBaselineMae'),
  metricArimaMae: document.getElementById('metricArimaMae'),
  metricXgbMae: document.getElementById('metricXgbMae'),
  badgeMaeImp: document.getElementById('badgeMaeImp'),
  
  metricHybridMape: document.getElementById('metricHybridMape'),
  metricBaselineMape: document.getElementById('metricBaselineMape'),
  metricArimaMape: document.getElementById('metricArimaMape'),
  
  metricHybridR2: document.getElementById('metricHybridR2'),
  metricBaselineR2: document.getElementById('metricBaselineR2'),
  metricArimaR2: document.getElementById('metricArimaR2'),
  
  // ADF Diagnostics
  adfStatusBadge: document.getElementById('adfStatusBadge'),
  adfStatVal: document.getElementById('adfStatVal'),
  adfPVal: document.getElementById('adfPVal'),
  adfCrit1: document.getElementById('adfCrit1'),
  adfCrit5: document.getElementById('adfCrit5'),
  adfExplanation: document.getElementById('adfExplanation'),
  
  // Benchmark Table
  tblArimaRmse: document.getElementById('tblArimaRmse'),
  tblArimaMae: document.getElementById('tblArimaMae'),
  tblArimaMape: document.getElementById('tblArimaMape'),
  tblArimaR2: document.getElementById('tblArimaR2'),

  tblSarimaxRmse: document.getElementById('tblSarimaxRmse'),
  tblSarimaxMae: document.getElementById('tblSarimaxMae'),
  tblSarimaxMape: document.getElementById('tblSarimaxMape'),
  tblSarimaxR2: document.getElementById('tblSarimaxR2'),
  
  tblXgbRmse: document.getElementById('tblXgbRmse'),
  tblXgbMae: document.getElementById('tblXgbMae'),
  tblXgbMape: document.getElementById('tblXgbMape'),
  tblXgbR2: document.getElementById('tblXgbR2'),
  
  tblHybridRmse: document.getElementById('tblHybridRmse'),
  tblHybridMae: document.getElementById('tblHybridMae'),
  tblHybridMape: document.getElementById('tblHybridMape'),
  tblHybridR2: document.getElementById('tblHybridR2'),
  
  // Benchmark Table Rows
  benchmarkEmptyRow: document.getElementById('benchmarkEmptyRow'),
  rowArima: document.getElementById('rowArima'),
  rowSarimax: document.getElementById('rowSarimax'),
  rowXgb: document.getElementById('rowXgb'),
  rowHybrid: document.getElementById('rowHybrid'),
  
  // Dedicated Parallel Workspaces
  workspacesGrid: document.getElementById('workspacesGrid'),
  panelArima: document.getElementById('panelArima'),
  panelSarimax: document.getElementById('panelSarimax'),
  panelXgb: document.getElementById('panelXgb'),
  panelHybrid: document.getElementById('panelHybrid'),
  
  chartContainerArima: document.getElementById('workspace-0-chart') || document.getElementById('chartContainerArima'),
  chartContainerSarimax: document.getElementById('workspace-1-chart') || document.getElementById('chartContainerSarimax'),
  chartContainerXgb: document.getElementById('workspace-2-chart') || document.getElementById('chartContainerXgb'),
  chartContainerHybrid: document.getElementById('workspace-3-chart') || document.getElementById('chartContainerHybrid'),
  
  arimaPanelRmse: document.getElementById('arimaPanelRmse'),
  arimaPanelMae: document.getElementById('arimaPanelMae'),
  arimaPanelMape: document.getElementById('arimaPanelMape'),

  sarimaxPanelRmse: document.getElementById('sarimaxPanelRmse'),
  sarimaxPanelMae: document.getElementById('sarimaxPanelMae'),
  sarimaxPanelMape: document.getElementById('sarimaxPanelMape'),
  
  xgbPanelRmse: document.getElementById('xgbPanelRmse'),
  xgbPanelMae: document.getElementById('xgbPanelMae'),
  xgbPanelMape: document.getElementById('xgbPanelMape'),
  
  hybridPanelRmse: document.getElementById('hybridPanelRmse'),
  hybridPanelMae: document.getElementById('hybridPanelMae'),
  hybridPanelHorizon: document.getElementById('hybridPanelHorizon'),
  
  // View Switcher Buttons
  viewBtnAll: document.getElementById('viewBtnAll'),
  viewBtnArima: document.getElementById('viewBtnArima'),
  viewBtnSarimax: document.getElementById('viewBtnSarimax'),
  viewBtnXgb: document.getElementById('viewBtnXgb'),
  viewBtnHybrid: document.getElementById('viewBtnHybrid'),
  
  residualPlotContainer: document.getElementById('residualPlotContainer')
};

// ==========================================
// Initialization & Backend Connectivity
// ==========================================
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      dom.backendText.textContent = 'API Live (Connected)';
      dom.backendBadge.className = 'flex items-center space-x-2 px-3 py-1 rounded text-xs font-semibold uppercase tracking-wider bg-emerald-950/40 border border-emerald-500/40 text-emerald-300';
      return true;
    }
  } catch (err) {
    dom.backendText.textContent = 'Backend Offline';
    dom.backendBadge.className = 'flex items-center space-x-2 px-3 py-1 rounded text-xs font-semibold uppercase tracking-wider bg-rose-950/40 border border-rose-500/40 text-rose-300';
  }
  return false;
}

// ==========================================
// Dashboard State Reset Management
// ==========================================
function resetDashboardState() {
  APP_STATE.forecastResult = null;
  APP_STATE.diagnosticsResult = null;

  // Reset KPI cards
  if (dom.metricHybridRmse) dom.metricHybridRmse.textContent = '--';
  if (dom.metricBaselineRmse) dom.metricBaselineRmse.textContent = '--';
  if (dom.metricArimaRmse) dom.metricArimaRmse.textContent = '--';
  if (dom.metricXgbRmse) dom.metricXgbRmse.textContent = '--';
  if (dom.badgeRmseImp) {
    dom.badgeRmseImp.textContent = 'Awaiting Run';
    dom.badgeRmseImp.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-slate-800 text-slate-400 border border-slate-700';
  }

  if (dom.metricHybridMae) dom.metricHybridMae.textContent = '--';
  if (dom.metricBaselineMae) dom.metricBaselineMae.textContent = '--';
  if (dom.metricArimaMae) dom.metricArimaMae.textContent = '--';
  if (dom.metricXgbMae) dom.metricXgbMae.textContent = '--';
  if (dom.badgeMaeImp) {
    dom.badgeMaeImp.textContent = 'Awaiting Run';
    dom.badgeMaeImp.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-slate-800 text-slate-400 border border-slate-700';
  }

  if (dom.metricHybridMape) dom.metricHybridMape.textContent = '--';
  if (dom.metricBaselineMape) dom.metricBaselineMape.textContent = '--';
  if (dom.metricArimaMape) dom.metricArimaMape.textContent = '--';

  if (dom.metricHybridR2) dom.metricHybridR2.textContent = '--';
  if (dom.metricBaselineR2) dom.metricBaselineR2.textContent = '--';
  if (dom.metricArimaR2) dom.metricArimaR2.textContent = '--';

  // Reset ADF Stationarity
  if (dom.adfStatusBadge) {
    dom.adfStatusBadge.textContent = 'Awaiting Execution';
    dom.adfStatusBadge.className = 'px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider bg-slate-800 text-slate-400 border border-slate-700';
  }
  if (dom.adfStatVal) dom.adfStatVal.textContent = '--';
  if (dom.adfPVal) dom.adfPVal.textContent = '--';
  if (dom.adfCrit1) dom.adfCrit1.textContent = '--';
  if (dom.adfCrit5) dom.adfCrit5.textContent = '--';
  if (dom.adfExplanation) dom.adfExplanation.textContent = 'Upload a CSV dataset and click "Run Hybrid Engine" to compute stationarity diagnostics.';

  // Reset Workspace Badges
  if (dom.arimaPanelRmse) dom.arimaPanelRmse.textContent = '--';
  if (dom.arimaPanelMae) dom.arimaPanelMae.textContent = '--';
  if (dom.arimaPanelMape) dom.arimaPanelMape.textContent = '--';

  if (dom.sarimaxPanelRmse) dom.sarimaxPanelRmse.textContent = '--';
  if (dom.sarimaxPanelMae) dom.sarimaxPanelMae.textContent = '--';
  if (dom.sarimaxPanelMape) dom.sarimaxPanelMape.textContent = '--';

  if (dom.xgbPanelRmse) dom.xgbPanelRmse.textContent = '--';
  if (dom.xgbPanelMae) dom.xgbPanelMae.textContent = '--';
  if (dom.xgbPanelMape) dom.xgbPanelMape.textContent = '--';

  if (dom.hybridPanelRmse) dom.hybridPanelRmse.textContent = '--';
  if (dom.hybridPanelMae) dom.hybridPanelMae.textContent = '--';
  if (dom.hybridPanelHorizon) dom.hybridPanelHorizon.textContent = `+${dom.selectHorizon ? dom.selectHorizon.value : 14} Steps`;

  // Empty placeholder helper with pure typography
  const emptyPrompt = (label) => `
    <div class="w-full h-full flex items-center justify-center">
      <p class="text-xs font-medium text-slate-500 tracking-wider uppercase">
        Awaiting execution — run hybrid engine to plot ${label}
      </p>
    </div>`;

  if (dom.chartContainerArima) {
    try { Plotly.purge(dom.chartContainerArima); } catch (e) {}
    dom.chartContainerArima.innerHTML = emptyPrompt('Pure ARIMA Baseline');
  }
  if (dom.chartContainerSarimax) {
    try { Plotly.purge(dom.chartContainerSarimax); } catch (e) {}
    dom.chartContainerSarimax.innerHTML = emptyPrompt('SARIMAX Baseline');
  }
  if (dom.chartContainerXgb) {
    try { Plotly.purge(dom.chartContainerXgb); } catch (e) {}
    dom.chartContainerXgb.innerHTML = emptyPrompt('Standalone XGBoost');
  }
  if (dom.chartContainerHybrid) {
    try { Plotly.purge(dom.chartContainerHybrid); } catch (e) {}
    dom.chartContainerHybrid.innerHTML = emptyPrompt('Hybrid Framework & Horizon');
  }
  if (dom.residualPlotContainer) {
    try { Plotly.purge(dom.residualPlotContainer); } catch (e) {}
    dom.residualPlotContainer.innerHTML = emptyPrompt('Residual Diagnostics');
  }

  // Reset Benchmark Matrix Table
  if (dom.benchmarkEmptyRow) dom.benchmarkEmptyRow.classList.remove('hidden');
  if (dom.rowArima) dom.rowArima.classList.add('hidden');
  if (dom.rowSarimax) dom.rowSarimax.classList.add('hidden');
  if (dom.rowXgb) dom.rowXgb.classList.add('hidden');
  if (dom.rowHybrid) dom.rowHybrid.classList.add('hidden');

  // Disable Export CSV
  if (dom.btnExportCSV) {
    dom.btnExportCSV.classList.add('opacity-50', 'cursor-not-allowed');
  }
}

// ==========================================
// User Data Ingestion
// ==========================================
async function handleFileUpload(file) {
  if (!file) return;
  
  // Wipe all existing state clean immediately upon new upload
  resetDashboardState();
  setLoadingState(true, 'Uploading & parsing CSV...');
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData
    });
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    
    const data = await res.json();
    APP_STATE.currentDataset = data;
    
    updateDatasetUI(data);
  } catch (err) {
    alert(`File upload error: ${err.message}`);
  } finally {
    setLoadingState(false);
  }
}

function updateDatasetUI(data) {
  dom.currentFileName.textContent = data.filename || 'dataset.csv';
  dom.currentFileStats.textContent = `(${data.total_rows} observations, ${data.columns.length} columns)`;
  dom.currentDateRange.textContent = data.date_range || 'Time-indexed';
  
  if (dom.datasetStatusIcon) {
    dom.datasetStatusIcon.className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
    dom.datasetStatusIcon.textContent = 'Active';
  }
  
  // Populate Target Column Select
  dom.selectTargetCol.innerHTML = '';
  const numCols = data.numeric_columns || data.columns;
  numCols.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = col;
    if (col === data.detected_target_column) opt.selected = true;
    dom.selectTargetCol.appendChild(opt);
  });
  
  // Populate Date Column Select
  dom.selectDateCol.innerHTML = '';
  data.columns.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = col;
    if (col === data.detected_date_column) opt.selected = true;
    dom.selectDateCol.appendChild(opt);
  });
}

// ==========================================
// Diagnostics (ADF Stationarity & Resid)
// ==========================================
async function runDiagnostics() {
  try {
    const payload = {
      date_column: dom.selectDateCol.value,
      target_column: dom.selectTargetCol.value,
      seasonal_period: 7
    };
    
    const res = await fetch(`${API_BASE}/api/diagnostics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) return;
    const data = await res.json();
    APP_STATE.diagnosticsResult = data;
    
    const adf = data.adf_test;
    if (adf && !adf.error) {
      dom.adfStatVal.textContent = adf.test_statistic;
      dom.adfPVal.textContent = adf.p_value;
      dom.adfCrit1.textContent = adf.critical_values['1%'] || '-';
      dom.adfCrit5.textContent = adf.critical_values['5%'] || '-';
      
      if (adf.is_stationary) {
        dom.adfStatusBadge.textContent = 'STATIONARY (p < 0.05)';
        dom.adfStatusBadge.className = 'px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        dom.adfExplanation.textContent = `The Augmented Dickey-Fuller test yielded a p-value of ${adf.p_value} (< 0.05). The null hypothesis of a unit root is rejected; the series demonstrates stationary stochastic equilibrium.`;
      } else {
        dom.adfStatusBadge.textContent = 'NON-STATIONARY (p >= 0.05)';
        dom.adfStatusBadge.className = 'px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/20';
        dom.adfExplanation.textContent = `ADF test p-value is ${adf.p_value} (>= 0.05). The series contains macro trend / unit root non-stationarity, handled systematically by SARIMAX first-order integration (d=1).`;
      }
    }
  } catch (err) {
    console.warn('Diagnostics notice:', err);
  }
}

// ==========================================
// Forecasting Execution (SARIMAX + XGBoost)
// ==========================================
async function executePipeline() {
  if (!APP_STATE.currentDataset) {
    alert('Please upload a CSV dataset file first to run the forecasting engine.');
    return;
  }
  if (!dom.selectDateCol.value || !dom.selectTargetCol.value) {
    alert('Please ensure both target metric and datetime columns are selected.');
    return;
  }
  setLoadingState(true, 'Training SARIMAX & XGBoost residuals...');
  
  try {
    const payload = {
      date_column: dom.selectDateCol.value,
      target_column: dom.selectTargetCol.value,
      train_ratio: parseFloat(dom.splitSlider.value) / 100,
      forecast_horizon: parseInt(dom.selectHorizon.value),
      residual_lags: 7
    };
    
    const res = await fetch(`${API_BASE}/api/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Execution failed');
    }
    
    const result = await res.json();
    APP_STATE.forecastResult = result;
    
    // Ensure all parallel workspaces are active and visible
    setActiveWorkspaceView('all');
    
    updateMetricsUI(result);
    renderAllForecastWorkspaces(result);
    renderResidualPlot(result);
  } catch (err) {
    alert(`Pipeline Error: ${err.message}`);
  } finally {
    setLoadingState(false);
  }
}

// ==========================================
// UI Updates: KPI Cards & Benchmark Matrix
// ==========================================
function updateMetricsUI(result) {
  const m = result.metrics;
  const imp = m.improvements;
  
  // KPI 1: RMSE
  dom.metricHybridRmse.textContent = m.hybrid.rmse.toFixed(2);
  dom.metricBaselineRmse.textContent = m.sarimax.rmse.toFixed(2);
  if (dom.metricArimaRmse && m.arima) dom.metricArimaRmse.textContent = m.arima.rmse.toFixed(2);
  dom.metricXgbRmse.textContent = m.xgboost.rmse.toFixed(2);
  dom.badgeRmseImp.textContent = `+${Math.abs(imp.rmse_pct)}% Imp`;
  
  // KPI 2: MAE
  dom.metricHybridMae.textContent = m.hybrid.mae.toFixed(2);
  dom.metricBaselineMae.textContent = m.sarimax.mae.toFixed(2);
  if (dom.metricArimaMae && m.arima) dom.metricArimaMae.textContent = m.arima.mae.toFixed(2);
  dom.metricXgbMae.textContent = m.xgboost.mae.toFixed(2);
  dom.badgeMaeImp.textContent = `+${Math.abs(imp.mae_pct)}% Imp`;
  
  // KPI 3: MAPE
  dom.metricHybridMape.textContent = `${m.hybrid.mape.toFixed(2)}%`;
  dom.metricBaselineMape.textContent = `${m.sarimax.mape.toFixed(2)}%`;
  if (dom.metricArimaMape && m.arima) dom.metricArimaMape.textContent = `${m.arima.mape.toFixed(2)}%`;
  
  // KPI 4: R2 Score
  dom.metricHybridR2.textContent = m.hybrid.r2.toFixed(4);
  dom.metricBaselineR2.textContent = m.sarimax.r2.toFixed(4);
  if (dom.metricArimaR2 && m.arima) dom.metricArimaR2.textContent = m.arima.r2.toFixed(4);
  
  // Benchmark Matrix Table
  if (m.arima) {
    if (dom.tblArimaRmse) dom.tblArimaRmse.textContent = m.arima.rmse.toFixed(4);
    if (dom.tblArimaMae) dom.tblArimaMae.textContent = m.arima.mae.toFixed(4);
    if (dom.tblArimaMape) dom.tblArimaMape.textContent = `${m.arima.mape.toFixed(2)}%`;
    if (dom.tblArimaR2) dom.tblArimaR2.textContent = m.arima.r2.toFixed(4);
  }

  dom.tblSarimaxRmse.textContent = m.sarimax.rmse.toFixed(4);
  dom.tblSarimaxMae.textContent = m.sarimax.mae.toFixed(4);
  dom.tblSarimaxMape.textContent = `${m.sarimax.mape.toFixed(2)}%`;
  dom.tblSarimaxR2.textContent = m.sarimax.r2.toFixed(4);
  
  dom.tblXgbRmse.textContent = m.xgboost.rmse.toFixed(4);
  dom.tblXgbMae.textContent = m.xgboost.mae.toFixed(4);
  dom.tblXgbMape.textContent = `${m.xgboost.mape.toFixed(2)}%`;
  dom.tblXgbR2.textContent = m.xgboost.r2.toFixed(4);
  
  dom.tblHybridRmse.textContent = m.hybrid.rmse.toFixed(4);
  dom.tblHybridMae.textContent = m.hybrid.mae.toFixed(4);
  dom.tblHybridMape.textContent = `${m.hybrid.mape.toFixed(2)}%`;
  dom.tblHybridR2.textContent = m.hybrid.r2.toFixed(4);

  // Reveal benchmark table rows
  if (dom.benchmarkEmptyRow) dom.benchmarkEmptyRow.classList.add('hidden');
  if (dom.rowArima) dom.rowArima.classList.remove('hidden');
  if (dom.rowSarimax) dom.rowSarimax.classList.remove('hidden');
  if (dom.rowXgb) dom.rowXgb.classList.remove('hidden');
  if (dom.rowHybrid) dom.rowHybrid.classList.remove('hidden');

  // Enable Export CSV button
  if (dom.btnExportCSV) {
    dom.btnExportCSV.classList.remove('opacity-50', 'cursor-not-allowed');
  }
}

// ==========================================
// Dedicated Parallel Forecast Workspaces
// ==========================================
function renderArimaPlot(data) {
  const container = document.getElementById('workspace-0-chart') || dom.chartContainerArima;
  if (!container || !data || !data.timeseries) return;
  const ts = data.timeseries;
  
  const traceTrain = {
    x: ts.dates_train,
    y: ts.y_train,
    mode: 'lines',
    name: 'Historical Train Data',
    line: { color: '#475569', width: 1.2 }
  };
  
  const traceTestActual = {
    x: ts.dates_test,
    y: ts.y_test,
    mode: 'lines+markers',
    name: 'Actual Ground Truth (Test)',
    marker: { size: 3.5, color: '#F8FAFC' },
    line: { color: '#F8FAFC', width: 2 }
  };
  
  const traceArima = {
    x: ts.dates_test,
    y: ts.arima_test,
    mode: 'lines+markers',
    name: 'Pure ARIMA Baseline Fit',
    marker: { size: 4, color: '#818CF8' },
    line: { color: '#818CF8', width: 2.2, dash: 'dash' }
  };

  const layout = {
    autosize: true,
    width: null,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15, 23, 42, 0.4)',
    height: 440,
    margin: { l: 45, r: 15, t: 25, b: 35 },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#0F172A',
      bordercolor: '#818CF8',
      font: { color: '#F8FAFC', family: 'Inter, sans-serif', size: 11 }
    },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      autorange: true,
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis: {
      title: { text: data.metadata.target_column, font: { color: '#94A3B8', size: 11 } },
      gridcolor: 'rgba(255, 255, 255, 0.07)',
      zeroline: false,
      autorange: true,
      rangemode: 'normal',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.15,
      font: { color: '#CBD5E1', size: 9.5 }
    }
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false, useResizeHandler: true };
  try { Plotly.purge(container); } catch (e) {}
  container.innerHTML = '';
  container.style.width = '100%';
  Plotly.newPlot(container, [traceTrain, traceTestActual, traceArima], layout, config).then(() => {
    Plotly.Plots.resize(container);
  });

  // Update badges
  if (dom.arimaPanelRmse && data.metrics?.arima) dom.arimaPanelRmse.textContent = data.metrics.arima.rmse.toFixed(2);
  if (dom.arimaPanelMae && data.metrics?.arima) dom.arimaPanelMae.textContent = data.metrics.arima.mae.toFixed(2);
  if (dom.arimaPanelMape && data.metrics?.arima) dom.arimaPanelMape.textContent = `${data.metrics.arima.mape.toFixed(2)}%`;
}

function renderSarimaxPlot(data) {
  const container = document.getElementById('workspace-1-chart') || dom.chartContainerSarimax;
  if (!container || !data || !data.timeseries) return;
  const ts = data.timeseries;
  
  const traceTrain = {
    x: ts.dates_train,
    y: ts.y_train,
    mode: 'lines',
    name: 'Historical Train Data',
    line: { color: '#475569', width: 1.2 }
  };
  
  const traceTestActual = {
    x: ts.dates_test,
    y: ts.y_test,
    mode: 'lines+markers',
    name: 'Actual Ground Truth (Test)',
    marker: { size: 3.5, color: '#F8FAFC' },
    line: { color: '#F8FAFC', width: 2 }
  };
  
  const traceSarimax = {
    x: ts.dates_test,
    y: ts.sarimax_test,
    mode: 'lines+markers',
    name: 'SARIMAX Baseline Fit',
    marker: { size: 4, color: '#06B6D4' },
    line: { color: '#06B6D4', width: 2.4 }
  };

  const layout = {
    autosize: true,
    width: null,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15, 23, 42, 0.4)',
    height: 440,
    margin: { l: 45, r: 15, t: 25, b: 35 },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#0F172A',
      bordercolor: '#06B6D4',
      font: { color: '#F8FAFC', family: 'Inter, sans-serif', size: 11 }
    },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      autorange: true,
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis: {
      title: { text: data.metadata.target_column, font: { color: '#94A3B8', size: 11 } },
      gridcolor: 'rgba(255, 255, 255, 0.07)',
      zeroline: false,
      autorange: true,
      rangemode: 'normal',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.15,
      font: { color: '#CBD5E1', size: 9.5 }
    }
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false, useResizeHandler: true };
  try { Plotly.purge(container); } catch (e) {}
  container.innerHTML = '';
  container.style.width = '100%';
  Plotly.newPlot(container, [traceTrain, traceTestActual, traceSarimax], layout, config).then(() => {
    Plotly.Plots.resize(container);
  });

  // Update badges
  if (dom.sarimaxPanelRmse && data.metrics?.sarimax) dom.sarimaxPanelRmse.textContent = data.metrics.sarimax.rmse.toFixed(2);
  if (dom.sarimaxPanelMae && data.metrics?.sarimax) dom.sarimaxPanelMae.textContent = data.metrics.sarimax.mae.toFixed(2);
  if (dom.sarimaxPanelMape && data.metrics?.sarimax) dom.sarimaxPanelMape.textContent = `${data.metrics.sarimax.mape.toFixed(2)}%`;
}

function renderXgbPlot(data) {
  const container = document.getElementById('workspace-2-chart') || dom.chartContainerXgb;
  if (!container || !data || !data.timeseries) return;
  const ts = data.timeseries;
  
  const traceTrain = {
    x: ts.dates_train,
    y: ts.y_train,
    mode: 'lines',
    name: 'Historical Train Data',
    line: { color: '#475569', width: 1.2 }
  };
  
  const traceTestActual = {
    x: ts.dates_test,
    y: ts.y_test,
    mode: 'lines+markers',
    name: 'Actual Ground Truth (Test)',
    marker: { size: 3.5, color: '#F8FAFC' },
    line: { color: '#F8FAFC', width: 2 }
  };
  
  const traceXGB = {
    x: ts.dates_test,
    y: ts.xgboost_test,
    mode: 'lines+markers',
    name: 'Standalone XGBoost Lags Fit',
    marker: { size: 4, color: '#F59E0B' },
    line: { color: '#F59E0B', width: 2.2, dash: 'dot' }
  };

  const layout = {
    autosize: true,
    width: null,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15, 23, 42, 0.4)',
    height: 440,
    margin: { l: 45, r: 15, t: 25, b: 35 },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#0F172A',
      bordercolor: '#F59E0B',
      font: { color: '#F8FAFC', family: 'Inter, sans-serif', size: 11 }
    },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      autorange: true,
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis: {
      title: { text: data.metadata.target_column, font: { color: '#94A3B8', size: 11 } },
      gridcolor: 'rgba(255, 255, 255, 0.07)',
      zeroline: false,
      autorange: true,
      rangemode: 'normal',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.15,
      font: { color: '#CBD5E1', size: 9.5 }
    }
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false, useResizeHandler: true };
  try { Plotly.purge(container); } catch (e) {}
  container.innerHTML = '';
  container.style.width = '100%';
  Plotly.newPlot(container, [traceTrain, traceTestActual, traceXGB], layout, config).then(() => {
    Plotly.Plots.resize(container);
  });

  // Update badges
  if (dom.xgbPanelRmse && data.metrics?.xgboost) dom.xgbPanelRmse.textContent = data.metrics.xgboost.rmse.toFixed(2);
  if (dom.xgbPanelMae && data.metrics?.xgboost) dom.xgbPanelMae.textContent = data.metrics.xgboost.mae.toFixed(2);
  if (dom.xgbPanelMape && data.metrics?.xgboost) dom.xgbPanelMape.textContent = `${data.metrics.xgboost.mape.toFixed(2)}%`;
}

function renderHybridPlot(data) {
  const container = document.getElementById('workspace-3-chart') || dom.chartContainerHybrid;
  if (!container || !data || !data.timeseries) return;
  const ts = data.timeseries;
  
  const traceTrain = {
    x: ts.dates_train,
    y: ts.y_train,
    mode: 'lines',
    name: 'Historical Train Data',
    line: { color: '#475569', width: 1.2 }
  };
  
  const traceTestActual = {
    x: ts.dates_test,
    y: ts.y_test,
    mode: 'lines+markers',
    name: 'Actual Ground Truth (Test)',
    marker: { size: 3.5, color: '#F8FAFC' },
    line: { color: '#F8FAFC', width: 2 }
  };
  
  const traceHybrid = {
    x: ts.dates_test,
    y: ts.hybrid_test,
    mode: 'lines+markers',
    name: 'Sequential Hybrid Forecast (Proposed)',
    marker: { size: 5, color: '#10B981' },
    line: { color: '#10B981', width: 2.8 }
  };
  
  const traceFuture = {
    x: ts.dates_future,
    y: ts.hybrid_future,
    mode: 'lines+markers',
    name: `Future Horizon (+${data.metadata.forecast_horizon} Steps)`,
    marker: { size: 5.5, color: '#A855F7' },
    line: { color: '#A855F7', width: 2.4, dash: 'solid' }
  };

  const layout = {
    autosize: true,
    width: null,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15, 23, 42, 0.4)',
    height: 440,
    margin: { l: 45, r: 15, t: 25, b: 35 },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#0F172A',
      bordercolor: '#10B981',
      font: { color: '#F8FAFC', family: 'Inter, sans-serif', size: 11 }
    },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      autorange: true,
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis: {
      title: { text: data.metadata.target_column, font: { color: '#94A3B8', size: 11 } },
      gridcolor: 'rgba(255, 255, 255, 0.07)',
      zeroline: false,
      autorange: true,
      rangemode: 'normal',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.15,
      font: { color: '#CBD5E1', size: 9.5 }
    }
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false, useResizeHandler: true };
  try { Plotly.purge(container); } catch (e) {}
  container.innerHTML = '';
  container.style.width = '100%';
  Plotly.newPlot(container, [traceTrain, traceTestActual, traceHybrid, traceFuture], layout, config).then(() => {
    Plotly.Plots.resize(container);
  });

  // Update badges
  if (dom.hybridPanelRmse && data.metrics?.improvements) {
    const pct = data.metrics.improvements.rmse_pct;
    dom.hybridPanelRmse.textContent = `${data.metrics.hybrid.rmse.toFixed(2)} (+${Math.abs(pct)}% Imp)`;
  }
  if (dom.hybridPanelMae && data.metrics?.hybrid) dom.hybridPanelMae.textContent = data.metrics.hybrid.mae.toFixed(2);
  if (dom.hybridPanelHorizon) dom.hybridPanelHorizon.textContent = `+${data.metadata.forecast_horizon} Steps`;
}

function renderAllForecastWorkspaces(data) {
  if (!data || !data.timeseries) return;
  
  // 1. Ensure all four workspace panels are visible
  if (dom.panelArima) dom.panelArima.classList.remove('hidden');
  if (dom.panelSarimax) dom.panelSarimax.classList.remove('hidden');
  if (dom.panelXgb) dom.panelXgb.classList.remove('hidden');
  if (dom.panelHybrid) dom.panelHybrid.classList.remove('hidden');

  // 2. Render all four workspaces simultaneously using Plotly.newPlot
  renderArimaPlot(data);
  renderSarimaxPlot(data);
  renderXgbPlot(data);
  renderHybridPlot(data);

  // 3. Multi-pass resize to guarantee complete layout synchronization
  requestAnimationFrame(resizeAllCharts);
  setTimeout(resizeAllCharts, 80);
  setTimeout(resizeAllCharts, 250);
}

// ==========================================
// Residual Diagnostics Chart
// ==========================================
function renderResidualPlot(data) {
  if (!dom.residualPlotContainer || !data || !data.residuals) return;
  const dates = data.timeseries.dates_test;
  const actualRes = data.residuals.test_actual_residuals;
  const predRes = data.residuals.test_predicted_residuals;

  const traceActualResid = {
    x: dates,
    y: actualRes,
    type: 'bar',
    name: 'SARIMAX Residual (Actual Discrepancy)',
    marker: { color: 'rgba(239, 68, 68, 0.65)' }
  };

  const tracePredictedResid = {
    x: dates,
    y: predRes,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'XGBoost Residual Learner Prediction',
    marker: { size: 4, color: '#38BDF8' },
    line: { color: '#38BDF8', width: 2 }
  };

  const layout = {
    autosize: true,
    width: null,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15, 23, 42, 0.4)',
    height: 380,
    margin: { l: 50, r: 20, t: 30, b: 40 },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#0F172A',
      bordercolor: '#38BDF8',
      font: { color: '#F8FAFC', family: 'Inter, sans-serif', size: 11 }
    },
    xaxis: {
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      autorange: true,
      tickfont: { color: '#94A3B8', size: 10 }
    },
    yaxis: {
      title: { text: 'Residual Error (e)', font: { color: '#94A3B8', size: 11 } },
      gridcolor: 'rgba(255, 255, 255, 0.05)',
      zerolinecolor: 'rgba(255, 255, 255, 0.25)',
      zerolinewidth: 1.5,
      autorange: true,
      rangemode: 'normal',
      tickfont: { color: '#94A3B8', size: 10 }
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.15,
      font: { color: '#CBD5E1', size: 10 }
    }
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false, useResizeHandler: true };
  try { Plotly.purge(dom.residualPlotContainer); } catch (e) {}
  dom.residualPlotContainer.innerHTML = '';
  dom.residualPlotContainer.style.width = '100%';
  dom.residualPlotContainer.style.height = '380px';
  Plotly.newPlot(dom.residualPlotContainer, [traceActualResid, tracePredictedResid], layout, config).then(() => {
    Plotly.Plots.resize(dom.residualPlotContainer);
  });
}

// ==========================================
// Export Forecast CSV
// ==========================================
function exportForecastCSV() {
  if (!APP_STATE.forecastResult) {
    alert('Please execute the forecast engine first before exporting.');
    return;
  }
  
  const ts = APP_STATE.forecastResult.timeseries;
  const targetName = APP_STATE.forecastResult.metadata.target_column;
  
  let csvContent = 'data:text/csv;charset=utf-8,';
  csvContent += `date,type,actual_${targetName},arima_pred,sarimax_pred,xgboost_pred,hybrid_pred\n`;
  
  // Test records
  for (let i = 0; i < ts.dates_test.length; i++) {
    const arimaVal = ts.arima_test && ts.arima_test[i] !== undefined ? ts.arima_test[i] : '';
    csvContent += `${ts.dates_test[i]},test_evaluation,${ts.y_test[i]},${arimaVal},${ts.sarimax_test[i]},${ts.xgboost_test[i]},${ts.hybrid_test[i]}\n`;
  }
  
  // Future horizon records
  for (let j = 0; j < ts.dates_future.length; j++) {
    const arimaFutureVal = ts.arima_future && ts.arima_future[j] !== undefined ? ts.arima_future[j] : '';
    csvContent += `${ts.dates_future[j]},future_horizon,,${arimaFutureVal},${ts.sarimax_future[j]},${ts.xgboost_future[j]},${ts.hybrid_future[j]}\n`;
  }
  
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `enterprise_hybrid_forecast_${new Date().toISOString().slice(0,10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==========================================
// Helper States & Event Listeners
// ==========================================
function setLoadingState(isLoading, message = 'Processing...') {
  if (isLoading) {
    dom.runIcon.className = 'fa-solid fa-spinner fa-spin';
    dom.runText.textContent = message;
    dom.btnRunPipeline.disabled = true;
    dom.btnRunPipeline.classList.add('opacity-75', 'cursor-not-allowed');
  } else {
    dom.runIcon.className = 'fa-solid fa-play';
    dom.runText.textContent = 'Run Hybrid Engine';
    dom.btnRunPipeline.disabled = false;
    dom.btnRunPipeline.classList.remove('opacity-75', 'cursor-not-allowed');
  }
}

// Event Listeners
dom.splitSlider.addEventListener('input', (e) => {
  const trainVal = e.target.value;
  dom.trainSplitLabel.textContent = `${trainVal}% / ${100 - trainVal}%`;
});

dom.csvFileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileUpload(e.target.files[0]);
  }
});

dom.btnRunPipeline.addEventListener('click', async () => {
  await runDiagnostics();
  await executePipeline();
});

dom.btnExportCSV.addEventListener('click', exportForecastCSV);

// Workspace View Switcher Handlers
function setActiveWorkspaceView(mode) {
  const buttons = [
    { btn: dom.viewBtnAll, key: 'all' },
    { btn: dom.viewBtnArima, key: 'arima' },
    { btn: dom.viewBtnSarimax, key: 'sarimax' },
    { btn: dom.viewBtnXgb, key: 'xgb' },
    { btn: dom.viewBtnHybrid, key: 'hybrid' }
  ];

  buttons.forEach(({ btn, key }) => {
    if (!btn) return;
    if (key === mode) {
      btn.className = 'px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 font-bold uppercase tracking-wider text-[11px] border border-cyan-500/30 transition-all';
    } else {
      btn.className = 'px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 font-bold uppercase tracking-wider text-[11px] transition-all';
    }
  });

  if (mode === 'all') {
    if (dom.workspacesGrid) {
      dom.workspacesGrid.classList.remove('mode-single', 'grid-cols-1');
      dom.workspacesGrid.classList.add('mode-all', 'xl:grid-cols-4');
    }
    if (dom.panelArima) dom.panelArima.classList.remove('hidden');
    if (dom.panelSarimax) dom.panelSarimax.classList.remove('hidden');
    if (dom.panelXgb) dom.panelXgb.classList.remove('hidden');
    if (dom.panelHybrid) dom.panelHybrid.classList.remove('hidden');
  } else {
    if (dom.workspacesGrid) {
      dom.workspacesGrid.classList.remove('mode-all', 'xl:grid-cols-4');
      dom.workspacesGrid.classList.add('mode-single', 'grid-cols-1');
    }
    if (dom.panelArima) dom.panelArima.classList.toggle('hidden', mode !== 'arima');
    if (dom.panelSarimax) dom.panelSarimax.classList.toggle('hidden', mode !== 'sarimax');
    if (dom.panelXgb) dom.panelXgb.classList.toggle('hidden', mode !== 'xgb');
    if (dom.panelHybrid) dom.panelHybrid.classList.toggle('hidden', mode !== 'hybrid');
  }

  // If forecast results exist, re-render and re-size the active workspaces
  if (APP_STATE.forecastResult) {
    if (mode === 'all') {
      renderArimaPlot(APP_STATE.forecastResult);
      renderSarimaxPlot(APP_STATE.forecastResult);
      renderXgbPlot(APP_STATE.forecastResult);
      renderHybridPlot(APP_STATE.forecastResult);
    } else if (mode === 'arima') {
      renderArimaPlot(APP_STATE.forecastResult);
    } else if (mode === 'sarimax') {
      renderSarimaxPlot(APP_STATE.forecastResult);
    } else if (mode === 'xgb') {
      renderXgbPlot(APP_STATE.forecastResult);
    } else if (mode === 'hybrid') {
      renderHybridPlot(APP_STATE.forecastResult);
    }
  }

  // Trigger Plotly responsive resize
  requestAnimationFrame(resizeAllCharts);
  setTimeout(resizeAllCharts, 60);
  setTimeout(resizeAllCharts, 250);
}

if (dom.viewBtnAll) dom.viewBtnAll.addEventListener('click', () => setActiveWorkspaceView('all'));
if (dom.viewBtnArima) dom.viewBtnArima.addEventListener('click', () => setActiveWorkspaceView('arima'));
if (dom.viewBtnSarimax) dom.viewBtnSarimax.addEventListener('click', () => setActiveWorkspaceView('sarimax'));
if (dom.viewBtnXgb) dom.viewBtnXgb.addEventListener('click', () => setActiveWorkspaceView('xgb'));
if (dom.viewBtnHybrid) dom.viewBtnHybrid.addEventListener('click', () => setActiveWorkspaceView('hybrid'));

// Auto-resize Plotly charts on window resizing
function resizeAllCharts() {
  const arimaEl = document.getElementById('workspace-0-chart') || dom.chartContainerArima;
  const sarimaxEl = document.getElementById('workspace-1-chart') || dom.chartContainerSarimax;
  const xgbEl = document.getElementById('workspace-2-chart') || dom.chartContainerXgb;
  const hybridEl = document.getElementById('workspace-3-chart') || dom.chartContainerHybrid;
  const residEl = dom.residualPlotContainer;

  if (arimaEl && arimaEl.offsetHeight > 0) Plotly.Plots.resize(arimaEl);
  if (sarimaxEl && sarimaxEl.offsetHeight > 0) Plotly.Plots.resize(sarimaxEl);
  if (xgbEl && xgbEl.offsetHeight > 0) Plotly.Plots.resize(xgbEl);
  if (hybridEl && hybridEl.offsetHeight > 0) Plotly.Plots.resize(hybridEl);
  if (residEl && residEl.offsetHeight > 0) Plotly.Plots.resize(residEl);
}

window.addEventListener('resize', resizeAllCharts);

// Initial Launch on Page Load
window.addEventListener('DOMContentLoaded', async () => {
  resetDashboardState();
  await checkBackendHealth();
});

