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

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchForecast = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(FORECAST_ENDPOINT);
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
    fetchForecast();
  }, []);

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
        <button onClick={fetchForecast} disabled={loading}>
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

          <section className="metrics">
            <h2>Model Metrics</h2>
            <div className="metric-list">
              <span>MAE: {data.metrics.mae.toFixed(2)}</span>
              <span>RMSE: {data.metrics.rmse.toFixed(2)}</span>
              <span>R2: {data.metrics.r2.toFixed(4)}</span>
            </div>
          </section>

          <section className="history">
            <h2>Recent BTC Close History (last 10 records)</h2>
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Close</th>
                </tr>
              </thead>
              <tbody>
                {data.history.slice(-10).reverse().map((row) => (
                  <tr key={row.date}>
                    <td>{row.date}</td>
                    <td>{formatUsd(row.close)}</td>
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
