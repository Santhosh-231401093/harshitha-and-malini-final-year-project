import io
import os
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.engine import (
    clean_and_prepare_series,
    run_adf_test,
    run_decomposition,
    run_hybrid_pipeline
)

app = FastAPI(
    title="Full-Stack Enterprise Hybrid Forecasting Suite API",
    description="Backend engine integrating SARIMAX and XGBoost with Sequential Residual Learning",
    version="1.0.0"
)

# Enable CORS for seamless client-server frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for the active dataset session (starts clean waiting for upload)
SESSION_DATA: Dict[str, Any] = {
    "df": None,
    "filename": None,
    "date_column": None,
    "target_column": None
}


def auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Intelligently detects datetime and target numeric columns."""
    date_col = None
    target_col = None
    
    # 1. Search for datetime column
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ["date", "time", "timestamp", "dt", "year", "day"]):
            date_col = col
            break
            
    if not date_col:
        for col in df.columns:
            try:
                converted = pd.to_datetime(df[col].dropna().iloc[:30], errors='coerce')
                if converted.notnull().mean() > 0.8:
                    date_col = col
                    break
            except Exception:
                continue

    # 2. Search for numeric target column
    numeric_cols = [c for c in df.select_dtypes(include=['number']).columns if c != date_col]
    if numeric_cols:
        target_col = numeric_cols[0]
        
    return date_col, target_col


@app.get("/api/health")
def health_check():
    """Health status check endpoint."""
    return {
        "status": "online",
        "service": "Enterprise Hybrid Forecasting Engine",
        "has_active_dataset": SESSION_DATA["df"] is not None,
        "dataset_name": SESSION_DATA.get("filename")
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accepts CSV dataset uploads.
    Cleans dates, parses time-series columns, and returns dataset summary metadata.
    """
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        
        # Try multiple common encodings to handle any CSV file safely
        decoded = None
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                decoded = contents.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
                
        if decoded is None:
            raise HTTPException(status_code=400, detail="Could not decode file with standard encodings (utf-8, latin-1, cp1252, iso-8859-1).")
            
        df = pd.read_csv(io.StringIO(decoded))
        
        # Save temporarily as the active custom dataset
        try:
            df.to_csv("backend/active_data.csv", index=False)
        except Exception:
            pass

        if df.empty or len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Uploaded CSV must contain at least 2 columns and data rows.")

        SESSION_DATA["df"] = df
        SESSION_DATA["filename"] = file.filename
        
        date_col, target_col = auto_detect_columns(df)
        SESSION_DATA["date_column"] = date_col or df.columns[0]
        SESSION_DATA["target_column"] = target_col or df.columns[-1]

        # Calculate date range if date column detected
        date_range_str = "N/A"
        try:
            temp_dates = pd.to_datetime(df[SESSION_DATA["date_column"]], errors='coerce').dropna()
            if not temp_dates.empty:
                date_range_str = f"{temp_dates.min().strftime('%Y-%m-%d')} to {temp_dates.max().strftime('%Y-%m-%d')}"
        except Exception:
            pass

        preview = df.head(5).fillna("").to_dict(orient="records")

        return {
            "status": "success",
            "filename": file.filename,
            "total_rows": len(df),
            "columns": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "date_range": date_range_str,
            "detected_date_column": SESSION_DATA["date_column"],
            "detected_target_column": SESSION_DATA["target_column"],
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")


class DiagnosticsRequest(BaseModel):
    date_column: Optional[str] = None
    target_column: Optional[str] = None
    seasonal_period: Optional[int] = 7


@app.post("/api/diagnostics")
def compute_diagnostics(req: DiagnosticsRequest):
    """
    Executes Augmented Dickey-Fuller (ADF) stationarity test
    and seasonal decomposition on the active dataset.
    """
    if SESSION_DATA["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload a CSV or load the sample dataset first.")

    df = SESSION_DATA["df"]
    date_col = req.date_column or SESSION_DATA["date_column"]
    target_col = req.target_column or SESSION_DATA["target_column"]

    if date_col not in df.columns or target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columns '{date_col}' or '{target_col}' not found in active dataset.")

    try:
        cleaned_df = clean_and_prepare_series(df, date_col, target_col)
        adf_result = run_adf_test(cleaned_df[target_col])
        decomp_result = run_decomposition(cleaned_df[target_col], cleaned_df[date_col], period=req.seasonal_period or 7)

        return {
            "status": "success",
            "target_column": target_col,
            "date_column": date_col,
            "adf_test": adf_result,
            "decomposition": decomp_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {str(e)}")


class ForecastRequest(BaseModel):
    date_column: Optional[str] = None
    target_column: Optional[str] = None
    train_ratio: Optional[float] = 0.8
    forecast_horizon: Optional[int] = 14
    arima_p: Optional[int] = 1
    arima_d: Optional[int] = 1
    arima_q: Optional[int] = 1
    sarimax_p: Optional[int] = 1
    sarimax_d: Optional[int] = 1
    sarimax_q: Optional[int] = 1
    seasonal_p: Optional[int] = 1
    seasonal_d: Optional[int] = 0
    seasonal_q: Optional[int] = 1
    seasonal_s: Optional[int] = 7
    residual_lags: Optional[int] = 7


@app.post("/api/forecast")
def run_forecast(req: ForecastRequest):
    """
    Executes the Two-Stage Hybrid Forecasting Pipeline (SARIMAX + XGBoost Sequential Residuals),
    benchmarks against standalone pure ARIMA, SARIMAX, and XGBoost models, and generates future horizon predictions.
    """
    if SESSION_DATA["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload a CSV or load the sample dataset first.")

    df = SESSION_DATA["df"]
    date_col = req.date_column or SESSION_DATA["date_column"]
    target_col = req.target_column or SESSION_DATA["target_column"]

    if date_col not in df.columns or target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Columns '{date_col}' or '{target_col}' not found.")

    try:
        arima_order = (req.arima_p or 1, req.arima_d or 1, req.arima_q or 1)
        sarimax_order = (req.sarimax_p or 1, req.sarimax_d or 1, req.sarimax_q or 1)
        seasonal_order = (req.seasonal_p or 1, req.seasonal_d or 0, req.seasonal_q or 1, req.seasonal_s or 7)
        
        result = run_hybrid_pipeline(
            df=df,
            date_col=date_col,
            target_col=target_col,
            train_ratio=req.train_ratio or 0.8,
            forecast_horizon=req.forecast_horizon or 14,
            arima_order=arima_order,
            sarimax_order=sarimax_order,
            seasonal_order=seasonal_order,
            lag_count=req.residual_lags or 7
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting engine error: {str(e)}")


# Mount frontend static directory if exists
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(frontend_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found"}

    @app.get("/style.css")
    def serve_style():
        return FileResponse(os.path.join(frontend_path, "style.css"))

    @app.get("/script.js")
    def serve_script():
        return FileResponse(os.path.join(frontend_path, "script.js"))
