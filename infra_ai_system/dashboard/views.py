from django.shortcuts import render
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def dashboard(request):

    cascade_file = os.path.join(BASE_DIR, "data/graphs/run_results_weather_2.5.json")
    repair_file = os.path.join(BASE_DIR, "data/graphs/repair_impact_scale_1.2.json")
    opt_file = os.path.join(BASE_DIR, "data/graphs/opt_vs_random_scale_1.2.json")
    roads_file = os.path.join(BASE_DIR, "data/gis/roads_final.geojson")

    cascade_data = None
    repair_data = None
    opt_data = None
    roads_geojson = None

    if os.path.exists(cascade_file):
        with open(cascade_file) as f:
            cascade_data = json.load(f)

    if os.path.exists(repair_file):
        with open(repair_file) as f:
            repair_data = json.load(f)

    if os.path.exists(opt_file):
        with open(opt_file) as f:
            opt_data = json.load(f)

    if os.path.exists(roads_file):
        with open(roads_file) as f:
            roads_geojson = json.load(f)

    context = {
        "cascade": cascade_data,
        "repair": repair_data,
        "optimization": opt_data,
        "roads": roads_geojson
    }

    return render(request, "dashboard/dashboard.html", context)