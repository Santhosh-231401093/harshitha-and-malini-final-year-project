# Full-Stack Enterprise Hybrid Forecasting Suite

**Project Title:** A Data-Driven Hybrid Forecasting Framework Integrating SARIMAX and XGBoost with Sequential Residual Learning  
**Architecture:** Client-Server Full-Stack (`FastAPI` Python Backend + Custom Tailwind / JavaScript Dashboard)

---

## 🌟 Executive Overview
Traditional time-series models (such as pure ARIMA and SARIMAX) are mathematically rigorous at modeling linear trends and cyclical seasonality, but struggle with non-linear volatility, exogenous shocks, and abrupt regime changes. Conversely, pure machine learning tree models (like XGBoost) excel at non-linear relationships but fundamentally struggle with out-of-range linear trend extrapolation.

This enterprise full-stack suite implements a complete **Evolutionary Model Comparison Pipeline**:
1. **Workspace 1 (Non-Seasonal Linear Baseline - Pure ARIMA):** Benchmark autoregressive integrated moving average $(p,d,q)$ modeling without seasonal terms.
2. **Workspace 2 (Seasonal Linear Base - SARIMAX):** Captures linear trends and cyclical weekly seasonality $(s=7)$.
3. **Workspace 3 (Non-Linear Autoregressive ML - Standalone XGBoost):** Direct lag learning $(t-1 \dots t-7)$ on raw target values.
4. **Workspace 4 (Two-Stage Sequential Residual Hybrid Framework):** Fuses SARIMAX trend/seasonality with XGBoost residual error learning ($\hat{y}_{\text{Hybrid}} = \hat{y}_{\text{SARIMAX}} + \hat{e}_{\text{XGBoost}}$) to outperform all standalone competitors.

---

## 🏗️ Project Architecture & Directory Structure

```text
hybrid Time-Series Forecasting Suite project/
│
├── backend/
│   ├── main.py              # FastAPI REST API, CORS middleware, file uploads & endpoints
│   ├── engine.py            # Core ML pipeline (SARIMAX, XGBoost, ADF test, evaluation metrics)
│   ├── requirements.txt     # Python package dependencies
│   └── sample_data.csv      # Ready-to-use synthetic 420-day energy demand time series
│
└── frontend/
    ├── index.html           # High-end enterprise dashboard UI (Tailwind CSS, Inter font)
    ├── script.js            # API fetch handler, state manager & Plotly.js chart renderer
    └── style.css            # Dark-mode glassmorphic styling, glow accents, scrollbars
```

---

## 🚀 Quickstart Guide

### 1. Launch the Backend Server
The FastAPI backend runs on Uvicorn. To start the server:

```powershell
# Using Anaconda Python or your virtual environment
& "C:\ProgramData\anaconda3\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The server automatically serves both the API and the custom frontend dashboard:
* **Interactive Frontend Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive FastAPI Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **REST API Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 📡 REST API Specifications

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status and active dataset verification |
| `GET` | `/api/sample` | Preloads the built-in enterprise sample dataset |
| `POST` | `/api/upload` | Multipart CSV upload handler; cleans datetime & returns column metadata |
| `POST` | `/api/diagnostics` | Runs Augmented Dickey-Fuller (**ADF**) stationarity test & seasonal decomposition |
| `POST` | `/api/forecast` | Executes SARIMAX + XGBoost Sequential Residual Learning and returns metrics |

---

## 📊 Benchmark Performance Results (Sample Energy Dataset)

| Metric | Pure ARIMA Baseline | SARIMAX Baseline | Standalone XGBoost | Hybrid Framework (Proposed) | Improvement vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSE** ↓ | 11.4280 | 10.9639 | 20.7858 | **10.0067** | **+8.73%** |
| **MAE** ↓ | 8.9240 | 8.5991 | 16.1345 | **7.7834** | **+9.49%** |
| **MAPE** ↓ | 2.38% | 2.23% | 4.03% | **2.01%** | **+9.87%** |
| **$R^2$ Score** ↑ | 0.8521 | 0.8644 | 0.5125 | **0.8870** | **Superior Fit** |

---

## 💻 Frontend Dashboard Capabilities
* **Dynamic Ingestion:** Drag-and-drop CSV uploader or instant one-click sample loader.
* **Auto Column Inference:** Automatically detects timestamp and numeric target columns.
* **Hyperparameter Controls:** Interactive slider for Train/Test split (70% - 90%) and forecast horizon selector (7, 14, 21, 30 days).
* **Live KPI Cards:** Displays real-time test RMSE, MAE, MAPE, and $R^2$ with delta badges.
* **Multi-Layered Visualizer:** Plotly.js chart with interactive layer toggles (Historical actuals, SARIMAX baseline, Standalone XGBoost, Hybrid forecast, and future forecast horizon).
* **Statistical Diagnostics:** Augmented Dickey-Fuller test card with p-value, critical thresholds, and stationarity verdict.
* **Residual Analysis:** Dedicated chart visualizing SARIMAX error residuals vs XGBoost learned error compensation.
* **Export Hub:** One-click CSV export of ground truth, test predictions, and future forecasts.
