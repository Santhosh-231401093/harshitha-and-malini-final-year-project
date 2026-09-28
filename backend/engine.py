import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error safely avoiding division by zero."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates enterprise forecasting evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "r2": round(r2, 4)
    }


def clean_and_prepare_series(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Prepares time-series dataframe by parsing dates, sorting, handling duplicates and NaNs."""
    df_clean = df.copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[date_col])
    df_clean = df_clean.sort_values(by=date_col)
    
    # In case of duplicate timestamps, average the target
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.groupby(date_col)[target_col].mean().reset_index()
    
    # Interpolate remaining missing target values
    df_clean[target_col] = df_clean[target_col].interpolate(method='linear').bfill().ffill()
    return df_clean


def run_adf_test(series: pd.Series) -> Dict[str, Any]:
    """Executes the Augmented Dickey-Fuller stationarity test."""
    cleaned = series.dropna()
    if len(cleaned) < 15:
        return {
            "error": "Dataset too short for meaningful ADF test."
        }
    res = adfuller(cleaned, autolag='AIC')
    stat, p_val, used_lag, nobs, crit_vals, icbest = res
    is_stationary = bool(p_val < 0.05)
    
    return {
        "test_statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 5),
        "is_stationary": is_stationary,
        "lags_used": int(used_lag),
        "observations": int(nobs),
        "critical_values": {k: round(float(v), 4) for k, v in crit_vals.items()},
        "verdict": "Stationary Series (p < 0.05)" if is_stationary else "Non-Stationary Series (p >= 0.05 - Differencing recommended)"
    }


def run_decomposition(series: pd.Series, dates: pd.Series, period: int = 7) -> Dict[str, Any]:
    """Computes additive classical seasonal decomposition."""
    if len(series) < period * 2:
        period = max(2, len(series) // 4)
        
    decomp = seasonal_decompose(series.values, model='additive', period=period, extrapolate_trend='freq')
    
    date_strs = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates]
    return {
        "dates": date_strs,
        "observed": [round(float(x), 2) for x in decomp.observed],
        "trend": [round(float(x), 2) if not np.isnan(x) else None for x in decomp.trend],
        "seasonal": [round(float(x), 2) if not np.isnan(x) else None for x in decomp.seasonal],
        "residual": [round(float(x), 2) if not np.isnan(x) else None for x in decomp.resid]
    }


def create_lag_matrix(series_arr: np.ndarray, n_lags: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Creates autoregressive lag feature matrix (X) and target vector (y)."""
    X, y = [], []
    for i in range(n_lags, len(series_arr)):
        X.append(series_arr[i - n_lags:i])
        y.append(series_arr[i])
    return np.array(X), np.array(y)


def run_hybrid_pipeline(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    train_ratio: float = 0.8,
    forecast_horizon: int = 14,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    sarimax_order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
    lag_count: int = 7
) -> Dict[str, Any]:
    """
    Executes the Two-Stage Hybrid Forecasting Architecture:
    0. Fits Pure ARIMA to provide non-seasonal baseline competitor.
    1. Fits SARIMAX to model linear trends & cyclical seasonality.
    2. Calculates training residuals (y_train - y_sarimax_fit).
    3. Trains XGBoost Regressor on autoregressive residual lags (t-1 .. t-7).
    4. Computes fused Hybrid Predictions (SARIMAX baseline + sequential residual correction).
    5. Benchmarks against Standalone ARIMA, Standalone SARIMAX, and Standalone XGBoost.
    6. Produces multi-step future out-of-sample forecasts.
    """
    cleaned_df = clean_and_prepare_series(df, date_col, target_col)
    n = len(cleaned_df)
    
    if n < 30:
        raise ValueError("Dataset has fewer than 30 observations. Provide at least 30 observations for modeling.")

    dates = cleaned_df[date_col].values
    y = cleaned_df[target_col].values
    
    # Train / Test split
    split_idx = int(n * train_ratio)
    if split_idx <= lag_count + 10 or (n - split_idx) < 5:
        split_idx = max(lag_count + 10, int(n * 0.8))
        
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_len = len(y_test)

    # ==========================================
    # STAGE 0: Fit Pure ARIMA Baseline (Non-Seasonal)
    # ==========================================
    try:
        arima_mod = ARIMA(
            y_train,
            order=arima_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        arima_fit = arima_mod.fit()
        arima_test_pred = arima_fit.forecast(steps=test_len)
        arima_future_pred = arima_fit.forecast(steps=test_len + forecast_horizon)[test_len:]
    except Exception:
        # Fallback to ARIMA(1, 1, 0) if convergence issues occur
        arima_mod = ARIMA(y_train, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
        arima_fit = arima_mod.fit()
        arima_test_pred = arima_fit.forecast(steps=test_len)
        arima_future_pred = arima_fit.forecast(steps=test_len + forecast_horizon)[test_len:]

    # ==========================================
    # STAGE 1: Fit SARIMAX
    # ==========================================
    try:
        sarimax_mod = SARIMAX(
            y_train,
            order=sarimax_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarimax_fit = sarimax_mod.fit(disp=False, maxiter=80)
        sarimax_train_pred = sarimax_fit.fittedvalues
        sarimax_test_pred = sarimax_fit.forecast(steps=test_len)
        sarimax_future_pred = sarimax_fit.forecast(steps=test_len + forecast_horizon)[test_len:]
    except Exception:
        # Fallback to ARIMA(1,1,0) if convergence issues occur
        sarimax_mod = SARIMAX(y_train, order=(1, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
        sarimax_fit = sarimax_mod.fit(disp=False, maxiter=50)
        sarimax_train_pred = sarimax_fit.fittedvalues
        sarimax_test_pred = sarimax_fit.forecast(steps=test_len)
        sarimax_future_pred = sarimax_fit.forecast(steps=test_len + forecast_horizon)[test_len:]

    # ==========================================
    # STAGE 2: Sequential Residual Learning (XGBoost)
    # ==========================================
    # Discard initial initialization samples to prevent burn-in transient distortions
    burn_in = min(lag_count * 2, len(y_train) // 4)
    train_residuals = (y_train - sarimax_train_pred)[burn_in:]
    
    # Build lag features for residuals
    X_res_train, y_res_train = create_lag_matrix(train_residuals, n_lags=lag_count)
    
    xgb_resid_model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42
    )
    xgb_resid_model.fit(X_res_train, y_res_train)

    # Sequential walk-forward residual prediction on the test set
    all_res_history = list(train_residuals)
    predicted_test_residuals = []
    
    for i in range(test_len):
        input_lag = np.array(all_res_history[-lag_count:]).reshape(1, -1)
        res_pred = float(xgb_resid_model.predict(input_lag)[0])
        predicted_test_residuals.append(res_pred)
        # Advance residual stream with realized SARIMAX prediction discrepancy
        actual_step_res = float(y_test[i] - sarimax_test_pred[i])
        all_res_history.append(actual_step_res)
        
    predicted_test_residuals = np.array(predicted_test_residuals)

    # Future residual predictions (recursively rolling with dampening)
    future_residuals = []
    future_lag_history = list(all_res_history)
    for step_i in range(forecast_horizon):
        input_lag = np.array(future_lag_history[-lag_count:]).reshape(1, -1)
        res_pred = float(xgb_resid_model.predict(input_lag)[0])
        # Dampen future residual shocks toward zero over extended horizons
        damp_factor = max(0.2, 1.0 - (step_i / max(forecast_horizon, 1)) * 0.5)
        damped_pred = res_pred * damp_factor
        future_residuals.append(damped_pred)
        future_lag_history.append(damped_pred)
    future_residuals = np.array(future_residuals)

    # ==========================================
    # STAGE 3: Hybrid Fusion
    # ==========================================
    hybrid_test_pred = sarimax_test_pred + predicted_test_residuals
    hybrid_future_pred = sarimax_future_pred + future_residuals

    # ==========================================
    # BENCHMARK: Standalone XGBoost (Direct Target Lags)
    # ==========================================
    X_target_train, y_target_train = create_lag_matrix(y_train, n_lags=lag_count)
    xgb_direct_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.04,
        random_state=42
    )
    xgb_direct_model.fit(X_target_train, y_target_train)

    # Walk-forward on test set for fair comparison
    direct_history = list(y_train)
    direct_test_pred = []
    for i in range(test_len):
        input_lag = np.array(direct_history[-lag_count:]).reshape(1, -1)
        pred = float(xgb_direct_model.predict(input_lag)[0])
        direct_test_pred.append(pred)
        direct_history.append(float(y_test[i]))
    direct_test_pred = np.array(direct_test_pred)

    direct_future_pred = []
    for _ in range(forecast_horizon):
        input_lag = np.array(direct_history[-lag_count:]).reshape(1, -1)
        pred = float(xgb_direct_model.predict(input_lag)[0])
        direct_future_pred.append(pred)
        direct_history.append(pred)
    direct_future_pred = np.array(direct_future_pred)

    # ==========================================
    # METRICS EVALUATION
    # ==========================================
    metrics_arima = calculate_metrics(y_test, arima_test_pred)
    metrics_sarimax = calculate_metrics(y_test, sarimax_test_pred)
    metrics_xgb = calculate_metrics(y_test, direct_test_pred)
    metrics_hybrid = calculate_metrics(y_test, hybrid_test_pred)

    # Calculate percentage improvement of Hybrid over Baseline SARIMAX
    rmse_improvement = round(
        ((metrics_sarimax["rmse"] - metrics_hybrid["rmse"]) / max(1e-6, metrics_sarimax["rmse"])) * 100, 2
    )
    mae_improvement = round(
        ((metrics_sarimax["mae"] - metrics_hybrid["mae"]) / max(1e-6, metrics_sarimax["mae"])) * 100, 2
    )
    mape_improvement = round(
        ((metrics_sarimax["mape"] - metrics_hybrid["mape"]) / max(1e-6, metrics_sarimax["mape"])) * 100, 2
    )

    # Calculate percentage improvement of Hybrid over Baseline ARIMA
    rmse_imp_arima = round(
        ((metrics_arima["rmse"] - metrics_hybrid["rmse"]) / max(1e-6, metrics_arima["rmse"])) * 100, 2
    )
    mae_imp_arima = round(
        ((metrics_arima["mae"] - metrics_hybrid["mae"]) / max(1e-6, metrics_arima["mae"])) * 100, 2
    )
    mape_imp_arima = round(
        ((metrics_arima["mape"] - metrics_hybrid["mape"]) / max(1e-6, metrics_arima["mape"])) * 100, 2
    )

    # Build future dates
    last_date = pd.to_datetime(dates[-1])
    try:
        freq = pd.infer_freq(pd.to_datetime(dates[:min(50, len(dates))]))
        if not freq:
            freq = 'D'
    except Exception:
        freq = 'D'
        
    future_date_series = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq=freq)
    future_date_strs = [d.strftime('%Y-%m-%d') for d in future_date_series]

    date_strs_all = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates]
    date_strs_train = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in train_dates]
    date_strs_test = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in test_dates]

    return {
        "status": "success",
        "metadata": {
            "total_samples": n,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "forecast_horizon": forecast_horizon,
            "target_column": target_col,
            "date_column": date_col,
            "split_ratio": train_ratio
        },
        "metrics": {
            "arima": metrics_arima,
            "sarimax": metrics_sarimax,
            "xgboost": metrics_xgb,
            "hybrid": metrics_hybrid,
            "improvements": {
                "rmse_pct": rmse_improvement,
                "mae_pct": mae_improvement,
                "mape_pct": mape_improvement,
                "rmse_pct_arima": rmse_imp_arima,
                "mae_pct_arima": mae_imp_arima,
                "mape_pct_arima": mape_imp_arima
            }
        },
        "timeseries": {
            "dates_train": date_strs_train,
            "y_train": [round(float(x), 2) for x in y_train],
            "dates_test": date_strs_test,
            "y_test": [round(float(x), 2) for x in y_test],
            "arima_test": [round(float(x), 2) for x in arima_test_pred],
            "sarimax_test": [round(float(x), 2) for x in sarimax_test_pred],
            "xgboost_test": [round(float(x), 2) for x in direct_test_pred],
            "hybrid_test": [round(float(x), 2) for x in hybrid_test_pred],
            "dates_future": future_date_strs,
            "arima_future": [round(float(x), 2) for x in arima_future_pred],
            "sarimax_future": [round(float(x), 2) for x in sarimax_future_pred],
            "xgboost_future": [round(float(x), 2) for x in direct_future_pred],
            "hybrid_future": [round(float(x), 2) for x in hybrid_future_pred]
        },
        "residuals": {
            "test_actual_residuals": [round(float(a - p), 2) for a, p in zip(y_test, sarimax_test_pred)],
            "test_predicted_residuals": [round(float(x), 2) for x in predicted_test_residuals]
        }
    }
