from django.urls import path

from .views import btc_forecast_view, health_view

urlpatterns = [
    path("health/", health_view, name="health"),
    path("forecast/", btc_forecast_view, name="btc-forecast"),
]
