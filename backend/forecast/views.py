from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .services import run_btc_forecast


@require_GET
def health_view(_request):
    return JsonResponse({"status": "ok"})


@require_GET
def btc_forecast_view(_request):
    try:
        result = run_btc_forecast(symbol="BTC-USD")
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=503)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse(
        {
            "symbol": result.symbol,
            "latest_close": result.latest_close,
            "forecasts": result.forecasts,
            "metrics": result.metrics,
            "mlflow_run_id": result.mlflow_run_id,
            "history": result.history,
        }
    )
