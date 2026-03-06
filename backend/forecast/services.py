import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HORIZONS = {
    "1_day": 1,
    "1_week": 7,
    "1_month": 30,
    "3_months": 90,
}
TREND_WINDOW_DAYS = 365
RECENT_RETURNS_DAYS = 90
MAX_DAILY_LOG_MOVE = 0.0035


@dataclass
class ForecastResult:
    symbol: str
    latest_close: float
    forecasts: dict
    metrics: dict
    mlflow_run_id: str
    history: list


def _tracking_uri() -> str:
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return (Path(__file__).resolve().parent.parent / "mlruns").as_uri()


def _history_cache_path(symbol: str) -> Path:
    safe_symbol = symbol.replace("-", "_").replace("/", "_")
    return Path(__file__).resolve().parent.parent / ".cache" / f"{safe_symbol}_3y_1d.csv"


def _fetch_yfinance_data(symbol: str, retries: int = 3, wait_seconds: int = 2):
    for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(proxy_key, None)

    cache_dir = Path(__file__).resolve().parent.parent / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    no_proxy_session = requests.Session()
    no_proxy_session.trust_env = False

    last_exception = None
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers=symbol,
                period="3y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                proxy=None,
                session=no_proxy_session,
                threads=False,
            )
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.droplevel(-1, axis=1)
                if "Close" in data.columns:
                    return data
        except Exception as exc:
            last_exception = exc

        try:
            ticker = yf.Ticker(symbol, session=no_proxy_session)
            data = ticker.history(period="3y", interval="1d", auto_adjust=True)
            if not data.empty and "Close" in data.columns:
                return data
        except Exception as exc:
            last_exception = exc

        if attempt < retries - 1:
            time.sleep(wait_seconds * (attempt + 1))

    if last_exception:
        raise last_exception
    return pd.DataFrame()


def run_btc_forecast(symbol: str = "BTC-USD") -> ForecastResult:
    try:
        data = _fetch_yfinance_data(symbol=symbol)
    except Exception:
        data = pd.DataFrame()

    cache_file = _history_cache_path(symbol)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if data.empty or "Close" not in data:
        if cache_file.exists():
            data = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
        else:
            raise ValueError(
                "No BTC price data returned from yfinance (likely temporary rate-limit). "
                "Retry in 1-2 minutes, or run once later to seed local cache."
            )
    else:
        data.reset_index().to_csv(cache_file, index=False)

    closes = data["Close"].dropna().to_numpy()
    if closes.size < 120:
        raise ValueError("Not enough BTC history to build forecast.")

    y = closes.astype(float)
    y_log = np.log(y)
    x = np.arange(len(y_log)).reshape(-1, 1)

    split_idx = int(len(y_log) * 0.8)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train_log, y_test = y_log[:split_idx], y[split_idx:]

    train_model = LinearRegression()
    train_model.fit(x_train, y_train_log)

    test_pred = np.exp(train_model.predict(x_test))
    metrics = {
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "r2": float(r2_score(y_test, test_pred)),
    }

    latest_close = float(y[-1])
    latest_log_close = float(y_log[-1])

    trend_window = min(TREND_WINDOW_DAYS, len(y_log))
    x_trend = np.arange(trend_window).reshape(-1, 1)
    y_trend_log = y_log[-trend_window:]
    trend_model = LinearRegression()
    trend_model.fit(x_trend, y_trend_log)
    raw_daily_log_trend = float(trend_model.coef_[0])

    recent_window = min(RECENT_RETURNS_DAYS, len(y_log) - 1)
    if recent_window > 0:
        recent_log_returns = np.diff(y_log[-(recent_window + 1) :])
        median_recent_log_return = float(np.median(recent_log_returns))
    else:
        median_recent_log_return = raw_daily_log_trend

    blended_daily_log_trend = (0.4 * raw_daily_log_trend) + (
        0.6 * median_recent_log_return
    )
    conservative_daily_log_trend = float(
        np.clip(blended_daily_log_trend, -MAX_DAILY_LOG_MOVE, MAX_DAILY_LOG_MOVE)
    )

    forecasts: dict[str, dict] = {}
    for label, days_ahead in HORIZONS.items():
        predicted = float(
            np.exp(latest_log_close + (conservative_daily_log_trend * days_ahead))
        )
        change_pct = ((predicted - latest_close) / latest_close) * 100.0
        forecasts[label] = {
            "days_ahead": days_ahead,
            "predicted_close": predicted,
            "predicted_change_pct": change_pct,
        }

    history_df = data[["Close"]].tail(180).reset_index()
    history = [
        {"date": row["Date"].strftime("%Y-%m-%d"), "close": float(row["Close"])}
        for _, row in history_df.iterrows()
    ]

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment("btc_forecasting")
    with mlflow.start_run(run_name="btc-log-trend-conservative-3y") as run:
        mlflow.log_params(
            {
                "symbol": symbol,
                "model": "log_trend_conservative",
                "lookback_period": "3y",
                "interval": "1d",
                "train_test_split": "80_20",
                "trend_window_days": trend_window,
                "recent_returns_days": recent_window,
                "max_daily_log_move": MAX_DAILY_LOG_MOVE,
            }
        )
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_metric("latest_close", latest_close)
        mlflow.log_metric("raw_daily_log_trend", raw_daily_log_trend)
        mlflow.log_metric("median_recent_log_return", median_recent_log_return)
        mlflow.log_metric("blended_daily_log_trend", blended_daily_log_trend)
        mlflow.log_metric("conservative_daily_log_trend", conservative_daily_log_trend)
        for key, value in forecasts.items():
            mlflow.log_metric(f"forecast_{key}_close", value["predicted_close"])
            mlflow.log_metric(
                f"forecast_{key}_change_pct", value["predicted_change_pct"]
            )

        run_id = run.info.run_id

    return ForecastResult(
        symbol=symbol,
        latest_close=latest_close,
        forecasts=forecasts,
        metrics=metrics,
        mlflow_run_id=run_id,
        history=history,
    )
