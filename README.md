<<<<<<< HEAD
# BTC Analyser (Django + React + MLflow)

This project forecasts BTC prices for:

- 1 day
- 1 week
- 1 month
- 3 months

using the past 3 years of daily data from `yfinance`.

Backend is in `backend/` (Django API).  
Frontend is in `frontend/` (React + Vite).  
MLflow tracks each forecast run with params and metrics.

## 1) Activate virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If `.venv` does not exist:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install backend dependencies

```powershell
pip install -r backend\requirements.txt
```

## 3) Run Django backend

```powershell
cd backend
python manage.py migrate
python manage.py runserver
```

API endpoints:

- `http://127.0.0.1:8000/api/health/`
- `http://127.0.0.1:8000/api/forecast/`
- `http://127.0.0.1:8000/api/forecast/?model=arima` (ARIMA output)
- `http://127.0.0.1:8000/api/forecast/?model=both` (linear + ARIMA in one response)

MLflow data is stored locally in `backend/mlruns/`.

To inspect runs:

```powershell
cd backend
mlflow ui
```

Then open `http://127.0.0.1:5000`.

## 4) Run React frontend

In a new terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend default URL:

- `http://127.0.0.1:5173`

If backend URL changes, set:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8001"
```
=======
# MLflow
>>>>>>> f51ed3aa50e6da752166b2e7efc184ac4bc3c82f
