@echo off
echo =======================================================
echo Starting Enterprise Hybrid Forecasting Suite (FastAPI)
echo Dashboard: http://127.0.0.1:8000
echo API Docs:  http://127.0.0.1:8000/docs
echo =======================================================
start http://127.0.0.1:8000
py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
