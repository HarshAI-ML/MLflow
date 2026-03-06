from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .services import get_recent_mlflow_runs, run_btc_forecast


@require_GET
def health_view(_request):
    return JsonResponse({"status": "ok"})


@require_GET
def btc_forecast_view(request):
    model = request.GET.get("model", "linear")
    try:
        result = run_btc_forecast(symbol="BTC-USD", model=model)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=503)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)

    payload = {
        "symbol": result.symbol,
        "model": result.model,
        "latest_close": result.latest_close,
        "forecasts": result.forecasts,
        "metrics": result.metrics,
        "mlflow_run_id": result.mlflow_run_id,
        "history": result.history,
        "recent_runs": get_recent_mlflow_runs(limit=10),
    }
    if result.models is not None:
        payload["models"] = result.models
    if result.mlflow_run_ids is not None:
        payload["mlflow_run_ids"] = result.mlflow_run_ids
    return JsonResponse(payload)
