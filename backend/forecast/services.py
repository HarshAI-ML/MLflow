import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

HORIZONS = {
    "1_day": 1,
    "1_week": 7,
    "1_month": 30,
    "3_months": 90,
}
TREND_WINDOW_DAYS = 365
RECENT_RETURNS_DAYS = 90
MAX_DAILY_LOG_MOVE = 0.0035
ARIMA_CANDIDATE_ORDERS = [
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
    (3, 1, 1),
    (5, 1, 0),
]
SUPPORTED_MODELS = {"linear", "arima", "both"}


@dataclass
class ForecastResult:
    symbol: str
    latest_close: float
    forecasts: dict
    metrics: dict
    mlflow_run_id: str
    history: list
    model: str = "linear"
    models: dict | None = None
    mlflow_run_ids: dict | None = None


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


def _load_price_data(symbol: str) -> pd.DataFrame:
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

    return data


def _build_history(data: pd.DataFrame) -> list[dict]:
    history_df = data[["Close"]].tail(180).reset_index()
    return [
        {"date": row["Date"].strftime("%Y-%m-%d"), "close": float(row["Close"])}
        for _, row in history_df.iterrows()
    ]


def _compute_linear_outputs(y: np.ndarray, latest_close: float) -> tuple[dict, dict, dict]:
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

    params = {
        "model": "log_trend_conservative",
        "train_test_split": "80_20",
        "trend_window_days": trend_window,
        "recent_returns_days": recent_window,
        "max_daily_log_move": MAX_DAILY_LOG_MOVE,
    }
    metrics.update(
        {
            "raw_daily_log_trend": raw_daily_log_trend,
            "median_recent_log_return": median_recent_log_return,
            "blended_daily_log_trend": blended_daily_log_trend,
            "conservative_daily_log_trend": conservative_daily_log_trend,
        }
    )
    return forecasts, metrics, params


def _fit_arima(series: np.ndarray, order: tuple[int, int, int]):
    model = ARIMA(
        series,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit()


def _select_arima_order(series: np.ndarray) -> tuple[tuple[int, int, int], float]:
    best_order: tuple[int, int, int] | None = None
    best_aic = np.inf

    for order in ARIMA_CANDIDATE_ORDERS:
        try:
            fit_result = _fit_arima(series, order)
            if np.isfinite(fit_result.aic) and fit_result.aic < best_aic:
                best_aic = float(fit_result.aic)
                best_order = order
        except Exception:
            continue

    if best_order is None:
        raise ValueError("ARIMA training failed for all candidate orders.")
    return best_order, best_aic


def _compute_arima_outputs(y: np.ndarray, latest_close: float) -> tuple[dict, dict, dict]:
    split_idx = int(len(y) * 0.8)
    split_idx = max(60, min(split_idx, len(y) - 5))
    y_train, y_test = y[:split_idx], y[split_idx:]

    selected_order, selected_order_aic = _select_arima_order(y_train)
    train_model = _fit_arima(y_train, selected_order)
    test_pred = train_model.forecast(steps=len(y_test))
    metrics = {
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "r2": float(r2_score(y_test, test_pred)),
        "train_aic": float(train_model.aic),
        "selected_order_aic": selected_order_aic,
    }

    final_model = _fit_arima(y, selected_order)
    max_days_ahead = max(HORIZONS.values())
    future_pred = final_model.forecast(steps=max_days_ahead)

    forecasts: dict[str, dict] = {}
    for label, days_ahead in HORIZONS.items():
        predicted = float(max(future_pred[days_ahead - 1], 1e-6))
        change_pct = ((predicted - latest_close) / latest_close) * 100.0
        forecasts[label] = {
            "days_ahead": days_ahead,
            "predicted_close": predicted,
            "predicted_change_pct": change_pct,
        }

    params = {
        "model": "arima",
        "train_test_split": "80_20",
        "arima_order": str(selected_order),
        "candidate_orders": str(ARIMA_CANDIDATE_ORDERS),
    }
    return forecasts, metrics, params


def _log_model_run(
    symbol: str,
    model_key: str,
    latest_close: float,
    forecasts: dict,
    metrics: dict,
    model_params: dict,
) -> str:
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment("btc_forecasting")
    with mlflow.start_run(run_name=f"btc-{model_key}-3y") as run:
        params = {
            "symbol": symbol,
            "lookback_period": "3y",
            "interval": "1d",
        }
        params.update(model_params)
        mlflow.log_params(params)
        mlflow.log_metric("latest_close", latest_close)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        for key, value in forecasts.items():
            mlflow.log_metric(f"forecast_{key}_close", value["predicted_close"])
            mlflow.log_metric(
                f"forecast_{key}_change_pct", value["predicted_change_pct"]
            )
        return run.info.run_id


def _normalize_model_label(model_value: str | None) -> str:
    if not model_value:
        return "unknown"
    model_value = model_value.strip().lower()
    if model_value == "arima":
        return "arima"
    if model_value in {"log_trend_conservative", "linear"}:
        return "linear"
    return model_value


def get_recent_mlflow_runs(limit: int = 10) -> list[dict]:
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient()
    experiment = client.get_experiment_by_name("btc_forecasting")
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=limit,
        order_by=["start_time DESC"],
    )

    recent_runs = []
    for run in runs:
        rmse = run.data.metrics.get("rmse")
        mse = (float(rmse) ** 2) if rmse is not None else None
        recent_runs.append(
            {
                "mlflow_run_id": run.info.run_id,
                "model": _normalize_model_label(run.data.params.get("model")),
                "mse": mse,
                "r2": run.data.metrics.get("r2"),
            }
        )
    return recent_runs


def run_btc_forecast(symbol: str = "BTC-USD", model: str = "linear") -> ForecastResult:
    model = model.lower().strip()
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model}'. Choose one of: {', '.join(sorted(SUPPORTED_MODELS))}."
        )

    data = _load_price_data(symbol)
    closes = data["Close"].dropna().to_numpy()
    if closes.size < 120:
        raise ValueError("Not enough BTC history to build forecast.")

    y = closes.astype(float)
    latest_close = float(y[-1])
    history = _build_history(data)

    forecasts_linear = {}
    metrics_linear = {}
    run_id_linear = ""
    if model in {"linear", "both"}:
        forecasts_linear, metrics_linear, params_linear = _compute_linear_outputs(
            y, latest_close
        )
        run_id_linear = _log_model_run(
            symbol=symbol,
            model_key="linear-log-trend-conservative",
            latest_close=latest_close,
            forecasts=forecasts_linear,
            metrics=metrics_linear,
            model_params=params_linear,
        )
        if model == "linear":
            return ForecastResult(
                symbol=symbol,
                latest_close=latest_close,
                forecasts=forecasts_linear,
                metrics=metrics_linear,
                mlflow_run_id=run_id_linear,
                history=history,
                model="linear",
            )

    forecasts_arima = {}
    metrics_arima = {}
    run_id_arima = ""
    if model in {"arima", "both"}:
        forecasts_arima, metrics_arima, params_arima = _compute_arima_outputs(
            y, latest_close
        )
        run_id_arima = _log_model_run(
            symbol=symbol,
            model_key="arima",
            latest_close=latest_close,
            forecasts=forecasts_arima,
            metrics=metrics_arima,
            model_params=params_arima,
        )
        if model == "arima":
            return ForecastResult(
                symbol=symbol,
                latest_close=latest_close,
                forecasts=forecasts_arima,
                metrics=metrics_arima,
                mlflow_run_id=run_id_arima,
                history=history,
                model="arima",
            )

    return ForecastResult(
        symbol=symbol,
        latest_close=latest_close,
        forecasts=forecasts_linear,
        metrics=metrics_linear,
        mlflow_run_id=run_id_linear,
        history=history,
        model="both",
        models={
            "linear": {"forecasts": forecasts_linear, "metrics": metrics_linear},
            "arima": {"forecasts": forecasts_arima, "metrics": metrics_arima},
        },
        mlflow_run_ids={"linear": run_id_linear, "arima": run_id_arima},
    )
