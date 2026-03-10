import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";
const FORECAST_ENDPOINT = `${API_BASE}/api/forecast/`;

function formatUsd(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2
  }).format(value);
}

function formatPct(value) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatMetric(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedModel, setSelectedModel] = useState("linear");

  const fetchForecast = async (model = selectedModel) => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${FORECAST_ENDPOINT}?model=${encodeURIComponent(model)}`);
      if (!res.ok) {
        let message = `API request failed with status ${res.status}`;
        try {
          const errJson = await res.json();
          if (errJson?.error) {
            message = errJson.error;
          }
        } catch (_ignored) {}
        throw new Error(message);
      }
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchForecast(selectedModel);
  }, [selectedModel]);

  const forecastCards = useMemo(() => {
    if (!data?.forecasts) return [];
    return [
      { label: "1 Day", key: "1_day" },
      { label: "1 Week", key: "1_week" },
      { label: "1 Month", key: "1_month" },
      { label: "3 Months", key: "3_months" }
    ]
      .map((item) => {
        const value = data.forecasts[item.key];
        if (!value) return null;
        return { ...item, ...value };
      })
      .filter(Boolean);
  }, [data]);

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>BTC Forecast Analyser</h1>
          <p>3-year daily BTC-USD data from yfinance with MLflow tracking.</p>
        </div>
        <div className="controls">
          <label htmlFor="model-select">Model</label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={loading}
          >
            <option value="linear">Linear</option>
            <option value="arima">ARIMA</option>
          </select>
        </div>
        <button onClick={() => fetchForecast(selectedModel)} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh Forecast"}
        </button>
      </header>

      {error && <p className="error">{error}</p>}

      {data && (
        <>
          <section className="summary">
            <article className="card">
              <h3>Symbol</h3>
              <p>{data.symbol}</p>
            </article>
            <article className="card">
              <h3>Latest Close</h3>
              <p>{formatUsd(data.latest_close)}</p>
            </article>
            <article className="card wide">
              <h3>MLflow Run ID</h3>
              <p className="mono">{data.mlflow_run_id}</p>
            </article>
          </section>

          <section className="forecast-grid">
            {forecastCards.map((item) => (
              <article className="card" key={item.key}>
                <h3>{item.label}</h3>
                <p>{formatUsd(item.predicted_close)}</p>
                <p
                  className={
                    item.predicted_change_pct >= 0 ? "positive" : "negative"
                  }
                >
                  {formatPct(item.predicted_change_pct)}
                </p>
              </article>
            ))}
          </section>

          <section className="runs">
            <h2>Recent MLflow Runs (last 10)</h2>
            <table>
              <thead>
                <tr>
                  <th>MLflow Run ID</th>
                  <th>Model Type</th>
                  <th>MSE</th>
                  <th>R2</th>
                </tr>
              </thead>
              <tbody>
                {(data.recent_runs || []).map((row) => (
                  <tr key={row.mlflow_run_id}>
                    <td className="mono">{row.mlflow_run_id}</td>
                    <td>{row.model}</td>
                    <td>{formatMetric(row.mse, 2)}</td>
                    <td>{formatMetric(row.r2, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  );
}
