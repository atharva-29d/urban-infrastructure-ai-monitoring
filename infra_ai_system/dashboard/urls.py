from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("scenarios/", views.scenario_explorer, name="scenario_explorer"),
    path("models/", views.model_results, name="model_results"),
    path("optimization/", views.optimization_analysis, name="optimization_analysis"),
    path("routing/", views.route_planner, name="route_planner"),
    path("routing/search/", views.route_search, name="route_search"),
    path("runs/<slug:slug>/", views.run_detail, name="run_detail"),
]
