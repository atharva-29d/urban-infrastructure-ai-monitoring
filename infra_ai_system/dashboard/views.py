import json
import math
import pickle
import random
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import networkx as nx
import pandas as pd
from django.conf import settings
from django.db.models import Count
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render

from .models import ExperimentRun, OptimizationResult, ScenarioConfig


BENCHMARK_MODEL_METRICS = [
    {"model_name": "Random Forest", "roc_auc": 0.62, "family": "tabular"},
    {"model_name": "Temporal Neural Network", "roc_auc": 0.84, "family": "temporal"},
    {"model_name": "Graph Neural Network", "roc_auc": 0.51, "family": "graph"},
    {"model_name": "Spatio-Temporal GNN", "roc_auc": 0.63, "family": "graph-temporal"},
]


def repo_root():
    return Path(__file__).resolve().parents[2]


def dashboard_artifact_dir():
    return repo_root() / "data" / "dashboard_artifacts"


def mapbox_public_token():
    return getattr(settings, "MAPBOX_PUBLIC_TOKEN", "")


def mapbox_secret_token():
    return getattr(settings, "MAPBOX_SECRET_TOKEN", "")


def mapbox_enabled():
    return bool(mapbox_public_token() and mapbox_secret_token())


@lru_cache(maxsize=64)
def _load_json_cached(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return None


def load_json(path: Path):
    return _load_json_cached(str(path))


@lru_cache(maxsize=32)
def _load_pickle_cached(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


def load_pickle(path: Path):
    return _load_pickle_cached(str(path))


def available_severities():
    return [
        {
            "slug": "0.7",
            "label": "Mild 0.7",
            "graph_path": repo_root() / "data" / "graphs" / "pune_after_cascade_weather_0.7.gpickle",
            "results_path": repo_root() / "data" / "graphs" / "run_results_weather_0.7.json",
        },
        {
            "slug": "1.2",
            "label": "Severe 1.2",
            "graph_path": repo_root() / "data" / "graphs" / "pune_after_cascade_weather_1.2.gpickle",
            "results_path": None,
        },
        {
            "slug": "1.5",
            "label": "Very Severe 1.5",
            "graph_path": repo_root() / "data" / "graphs" / "pune_after_cascade_weather_1.5.gpickle",
            "results_path": repo_root() / "data" / "graphs" / "run_results_weather_1.5.json",
        },
        {
            "slug": "2.5",
            "label": "Extreme 2.5",
            "graph_path": repo_root() / "data" / "graphs" / "pune_after_cascade_weather.gpickle",
            "results_path": repo_root() / "data" / "graphs" / "run_results_weather_2.5.json",
        },
    ]


def parse_custom_scale(raw_value):
    if raw_value in (None, ""):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return round(value, 3)


def tuning_for_scale(scale):
    if scale <= 0.8:
        return 1.5, 1.2
    if scale <= 1.6:
        return 1.25, 1.5
    return 1.1, 1.8


def cascade_step_capacity_local(graph, overload_factor=0.8):
    new_failures = []
    failed_nodes = [node_id for node_id, data in graph.nodes(data=True) if data.get("failed")]

    if not failed_nodes:
        return []

    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "road" or data.get("failed"):
            continue

        base = data.get("base_traffic", data.get("traffic", 0))
        capacity = data.get("capacity", 1)

        failed_neighbors = 0
        for neighbor in graph.neighbors(node_id):
            neighbor_data = graph.nodes[neighbor]
            if neighbor_data.get("type") == "road" and neighbor_data.get("failed"):
                failed_neighbors += 1

        if failed_neighbors == 0:
            continue

        overload = base + overload_factor * base * failed_neighbors
        previous_stress = data.get("stress", 0)
        stress = previous_stress + overload / capacity
        data["stress"] = stress

        if stress > 1.0:
            probability = min(1.0, (stress - 1.0) * 1.5)
            if random.random() < probability:
                data["failed"] = True
                new_failures.append(node_id)

    return new_failures


def select_severity(severity_slug):
    severities = available_severities()
    severity_map = {item["slug"]: item for item in severities}
    return severity_map.get(severity_slug, severities[0]), severities


def load_roads_geojson():
    return load_json(repo_root() / "data" / "gis" / "roads_final.geojson")


def load_base_graph():
    return load_pickle(repo_root() / "data" / "graphs" / "pune_base_graph_weather.gpickle")


@lru_cache(maxsize=24)
def simulate_scale(scale, seed=42):
    base_graph = load_base_graph()
    graph = pickle.loads(pickle.dumps(base_graph)) if base_graph is not None else None
    if graph is None:
        return {
            "failed_ids": [],
            "summary": {
                "label": f"Custom {scale}",
                "slug": str(scale),
                "initial_failures": 0,
                "cascade_steps": 0,
                "peak_step_failures": 0,
                "capacity_cascade": [],
            },
        }

    capacity_factor, overload_factor = tuning_for_scale(scale)
    random.seed(seed)

    for _, data in graph.nodes(data=True):
        if data.get("type") != "road":
            continue
        traffic = safe_float(data.get("traffic"))
        data["base_traffic"] = traffic
        data["capacity"] = traffic * capacity_factor
        data["failed"] = False
        data["overloaded"] = False
        data["stress"] = 0.0

    initial_failed = 0
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "road":
            continue
        risk = max(0.0, min(1.0, safe_float(data.get("flood_risk"))))
        if random.random() < risk * scale:
            data["failed"] = True
            initial_failed += 1

    history = []
    for _ in range(20):
        new_failures = cascade_step_capacity_local(graph, overload_factor=overload_factor)
        history.append(len(new_failures))
        if not new_failures:
            break

    failed_ids = sorted(
        str(node_id)
        for node_id, data in graph.nodes(data=True)
        if data.get("type") == "road" and data.get("failed")
    )

    return {
        "failed_ids": failed_ids,
        "summary": {
            "label": f"Custom {scale}",
            "slug": str(scale),
            "initial_failures": initial_failed,
            "cascade_steps": len(history),
            "peak_step_failures": max(history) if history else 0,
            "capacity_cascade": history,
        },
    }


def load_failed_road_ids(graph_path: Path):
    graph = load_pickle(graph_path)
    if graph is None:
        return set()

    failed_ids = set()
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "road" and data.get("failed"):
            failed_ids.add(str(node_id))
    return failed_ids


def load_criticality_map():
    csv_path = repo_root() / "data" / "graphs" / "critical_roads.csv"
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    if "road_id" not in df.columns:
        return {}

    criticality_map = {}
    for _, row in df.iterrows():
        criticality_map[str(row["road_id"])] = {
            "criticality": safe_float(row.get("criticality")),
            "degree_centrality": safe_float(row.get("degree_centrality")),
            "traffic": safe_float(row.get("traffic")),
            "flood_risk_ranked": safe_float(row.get("flood_risk")),
        }
    return criticality_map


def safe_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default
    return result


def percentile_lookup(values_by_id):
    items = sorted(values_by_id.items(), key=lambda item: item[1])
    total = len(items)
    if total == 0:
        return {}

    percentiles = {}
    for index, (road_id, _) in enumerate(items):
        percentiles[road_id] = index / max(total - 1, 1)
    return percentiles


def feature_center(feature):
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    flat = []

    if geometry.get("type") == "LineString":
        flat = coords
    elif geometry.get("type") == "MultiLineString":
        for segment in coords:
            flat.extend(segment)

    if not flat:
        return None

    lon = sum(point[0] for point in flat) / len(flat)
    lat = sum(point[1] for point in flat) / len(flat)
    return lon, lat


def mapbox_request_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": "EDI-Research-Dashboard/1.0",
        },
    )
    try:
        with urlopen(request, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def mapbox_place_suggestions(query, limit=5):
    if not mapbox_enabled():
        return []

    encoded_query = quote(query.strip())
    params = urlencode(
        {
            "access_token": mapbox_secret_token(),
            "autocomplete": "true",
            "limit": limit,
            "country": "IN",
            "language": "en",
        }
    )
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_query}.json?{params}"
    payload = mapbox_request_json(url)
    if not payload:
        return []

    suggestions = []
    for feature in payload.get("features", []):
        center = feature.get("center") or []
        if len(center) != 2:
            continue
        suggestions.append(
            {
                "kind": "place",
                "label": feature.get("place_name") or feature.get("text") or "Mapbox place",
                "lng": center[0],
                "lat": center[1],
                "road_id": "",
            }
        )
    return suggestions


def mapbox_profile_name(travel_profile):
    return {
        "driving": "mapbox/driving-traffic",
        "walking": "mapbox/walking",
        "cycling": "mapbox/cycling",
    }.get(travel_profile, "mapbox/driving-traffic")


def mapbox_directions(origin, destination, travel_profile="driving"):
    if not mapbox_enabled():
        return None

    coordinates = f"{origin[0]},{origin[1]};{destination[0]},{destination[1]}"
    params = urlencode(
        {
            "access_token": mapbox_secret_token(),
            "alternatives": "true",
            "steps": "true",
            "overview": "full",
            "geometries": "geojson",
            "language": "en",
            "annotations": "distance,duration,speed",
        }
    )
    profile = mapbox_profile_name(travel_profile)
    url = f"https://api.mapbox.com/directions/v5/{profile}/{coordinates}?{params}"
    return mapbox_request_json(url)


@lru_cache(maxsize=1)
def roads_feature_lookup():
    roads_geojson = load_roads_geojson()
    lookup = {}
    if not roads_geojson:
        return lookup

    for feature in roads_geojson.get("features", []):
        road_id = resolve_feature_road_id(feature)
        if road_id:
            lookup[road_id] = feature
    return lookup


@lru_cache(maxsize=1)
def route_catalog():
    lookup = roads_feature_lookup()
    rows = []
    for road_id, feature in lookup.items():
        props = feature.get("properties", {})
        name = props.get("name") or props.get("road_name") or ""
        highway = props.get("highway") or ""
        center = feature_center(feature)
        if name:
            label = f"{name} ({road_id})"
        elif highway:
            label = f"{road_id} [{highway}]"
        else:
            label = road_id
        rows.append(
            {
                "id": road_id,
                "label": label,
                "name": name.lower(),
                "highway": str(highway).lower(),
                "center": center,
            }
        )
    return rows


def road_display_label(road_id):
    if not road_id:
        return ""
    for item in route_catalog():
        if item["id"] == road_id:
            return item["label"]
    return road_id


def nearest_road_id(lng, lat):
    best_id = ""
    best_distance = float("inf")
    for item in route_catalog():
        center = item.get("center")
        if not center:
            continue
        dx = center[0] - lng
        dy = center[1] - lat
        distance = (dx * dx) + (dy * dy)
        if distance < best_distance:
            best_distance = distance
            best_id = item["id"]
    return best_id


def build_graph_attribute_maps(graph):
    road_attrs = {}
    risk_values = {}
    traffic_values = {}

    if graph is None:
        return road_attrs, {}, {}

    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "road":
            continue

        road_id = str(node_id)
        flood_risk = safe_float(data.get("flood_risk"))
        traffic = safe_float(data.get("traffic"))

        road_attrs[road_id] = {
            "flood_risk": flood_risk,
            "traffic": traffic,
            "length": safe_float(data.get("length")),
            "rain_mm_mean": safe_float(data.get("rain_mm_mean")),
        }
        risk_values[road_id] = flood_risk
        traffic_values[road_id] = traffic

    return road_attrs, percentile_lookup(risk_values), percentile_lookup(traffic_values)


def critical_threshold(criticality_map, percentile=0.995):
    scores = sorted(
        attrs["criticality"]
        for attrs in criticality_map.values()
        if attrs.get("criticality") is not None
    )

    if not scores:
        return 1.0

    index = int(percentile * (len(scores) - 1))
    return scores[index]


def resolve_feature_road_id(feature):
    props = feature.get("properties", {})

    candidate_keys = [
        "road_id",
        "id",
        "fid",
        "osmid",
        "edge_id",
        "segment_id",
        "name",
    ]

    for key in candidate_keys:
        value = props.get(key)
        if value not in (None, ""):
            return str(value)

    if feature.get("id") not in (None, ""):
        return str(feature["id"])

    return None


def enrich_roads_geojson(roads_geojson, graph, failed_ids, criticality_map):
    if not roads_geojson:
        return None, 0

    graph_attrs, risk_percentiles, traffic_percentiles = build_graph_attribute_maps(graph)
    critical_cutoff = critical_threshold(criticality_map, percentile=0.995)

    enriched = {
        "type": roads_geojson.get("type", "FeatureCollection"),
        "features": [],
    }
    critical_count = 0

    for raw_feature in roads_geojson.get("features", []):
        feature = {
            "type": raw_feature.get("type", "Feature"),
            "geometry": raw_feature.get("geometry"),
            "properties": dict(raw_feature.get("properties", {})),
        }

        props = feature["properties"]
        road_id = resolve_feature_road_id(feature)
        graph_data = graph_attrs.get(road_id, {})
        criticality_data = criticality_map.get(road_id, {})

        criticality_score = safe_float(criticality_data.get("criticality"))
        is_critical = criticality_score >= critical_cutoff if criticality_map else False
        if is_critical:
            critical_count += 1

        props["resolved_road_id"] = road_id
        props["flood_risk"] = graph_data.get("flood_risk", 0.0)
        props["flood_risk_percentile"] = risk_percentiles.get(road_id, 0.0)
        props["traffic"] = graph_data.get("traffic", 0.0)
        props["traffic_percentile"] = traffic_percentiles.get(road_id, 0.0)
        props["rain_mm_mean"] = graph_data.get("rain_mm_mean", 0.0)
        props["criticality"] = criticality_score
        props["degree_centrality"] = safe_float(criticality_data.get("degree_centrality"))
        props["is_failed"] = road_id in failed_ids if road_id else False
        props["is_critical"] = is_critical

        enriched["features"].append(feature)

    return enriched, critical_count


@lru_cache(maxsize=1)
def critical_road_summary():
    roads_geojson = load_roads_geojson()
    criticality_map = load_criticality_map()
    if not roads_geojson or not criticality_map:
        return {}

    cutoff = critical_threshold(criticality_map, percentile=0.995)
    all_centers = []
    critical_centers = []

    for feature in roads_geojson.get("features", []):
        road_id = resolve_feature_road_id(feature)
        center = feature_center(feature)
        if center is None:
            continue
        all_centers.append(center)
        if safe_float(criticality_map.get(road_id, {}).get("criticality")) >= cutoff:
            critical_centers.append(center)

    if not all_centers or not critical_centers:
        return {}

    all_mid_lon = sum(point[0] for point in all_centers) / len(all_centers)
    critical_east = sum(1 for lon, _ in critical_centers if lon >= all_mid_lon)
    critical_west = len(critical_centers) - critical_east

    return {
        "critical_count": len(critical_centers),
        "city_mid_longitude": round(all_mid_lon, 6),
        "critical_east_share": round(critical_east / len(critical_centers), 3),
        "critical_west_share": round(critical_west / len(critical_centers), 3),
        "critical_centroid_lon": round(sum(point[0] for point in critical_centers) / len(critical_centers), 6),
        "critical_centroid_lat": round(sum(point[1] for point in critical_centers) / len(critical_centers), 6),
    }


def route_examples(limit=6):
    criticality_map = load_criticality_map()
    if not criticality_map:
        return []

    ranked = sorted(
        criticality_map.items(),
        key=lambda item: item[1].get("criticality", 0),
        reverse=True,
    )
    return [
        {
            "id": road_id,
            "label": road_display_label(road_id),
            "center": next((item.get("center") for item in route_catalog() if item["id"] == road_id), None),
        }
        for road_id, _ in ranked[:limit]
    ]


def search_route_options(query, limit=8):
    query = (query or "").strip().lower()
    if not query:
        return []

    starts = []
    contains = []
    for item in route_catalog():
        haystacks = [item["id"].lower(), item["label"].lower(), item["name"], item["highway"]]
        if any(h.startswith(query) for h in haystacks if h):
            starts.append(
                {
                    "kind": "road",
                    "id": item["id"],
                    "label": item["label"],
                    "lng": item["center"][0] if item.get("center") else None,
                    "lat": item["center"][1] if item.get("center") else None,
                    "road_id": item["id"],
                }
            )
        elif any(query in h for h in haystacks if h):
            contains.append(
                {
                    "kind": "road",
                    "id": item["id"],
                    "label": item["label"],
                    "lng": item["center"][0] if item.get("center") else None,
                    "lat": item["center"][1] if item.get("center") else None,
                    "road_id": item["id"],
                }
            )

        if len(starts) >= limit:
            break

    results = starts[:limit]
    if len(results) < limit:
        results.extend(contains[: limit - len(results)])
    remaining = max(limit - len(results), 0)
    if remaining:
        results.extend(mapbox_place_suggestions(query, limit=remaining))
    return results[:limit]


def routing_weight(graph, mode):
    def weight(u, v, edge_data):
        node = graph.nodes[v]
        length = safe_float(node.get("length"), 1.0)
        traffic = safe_float(node.get("traffic"))
        flood_risk = safe_float(node.get("flood_risk"))
        criticality = safe_float(node.get("criticality"))

        if mode == "fast":
            return length + 0.10 * traffic + 50.0 * flood_risk
        if mode == "balanced":
            return length + 0.20 * traffic + 120.0 * flood_risk + 80.0 * criticality
        if mode == "emergency":
            return length + 0.05 * traffic + 180.0 * flood_risk + 140.0 * criticality
        return length + 0.15 * traffic + 220.0 * flood_risk + 120.0 * criticality

    return weight


def build_route_geojson(route_ids):
    lookup = roads_feature_lookup()
    features = []
    for road_id in route_ids:
        feature = lookup.get(road_id)
        if feature:
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def route_metrics(graph, route_ids):
    total_length = 0.0
    total_traffic = 0.0
    total_risk = 0.0

    for road_id in route_ids:
        node = graph.nodes[road_id]
        total_length += safe_float(node.get("length"))
        total_traffic += safe_float(node.get("traffic"))
        total_risk += safe_float(node.get("flood_risk"))

    return {
        "segment_count": len(route_ids),
        "total_length": round(total_length, 2),
        "avg_traffic": round(total_traffic / max(len(route_ids), 1), 2),
        "avg_flood_risk": round(total_risk / max(len(route_ids), 1), 4),
    }


def route_safety_score(metrics, blocked_overlap=0):
    if not metrics:
        return 0

    risk_penalty = min(metrics.get("avg_flood_risk", 0.0) * 55.0, 55.0)
    traffic_penalty = min((metrics.get("avg_traffic", 0.0) / 220.0) * 20.0, 20.0)
    segment_penalty = min((metrics.get("segment_count", 0) / 28.0) * 10.0, 10.0)
    blocked_penalty = min(blocked_overlap * 7.0, 15.0)
    score = max(0.0, 100.0 - risk_penalty - traffic_penalty - segment_penalty - blocked_penalty)
    return round(score, 1)


def build_route_recommendation(mapbox_primary_route, route_result, route_comparison, blocked_overlap):
    if route_result and not route_result.get("error"):
        safe_score = route_safety_score(route_result.get("metrics", {}), blocked_overlap=0)
    else:
        safe_score = 0.0

    baseline_metrics = None
    if route_result and route_comparison is not None:
        baseline_metrics = {
            "segment_count": route_result["metrics"]["segment_count"] - route_comparison["segment_delta"],
            "avg_flood_risk": route_result["metrics"]["avg_flood_risk"] - route_comparison["risk_delta"],
            "avg_traffic": route_result["metrics"]["avg_traffic"],
        }
    baseline_score = route_safety_score(baseline_metrics or {}, blocked_overlap=blocked_overlap) if baseline_metrics else 0.0

    if blocked_overlap > 0:
        return {
            "label": "Recommended: Resilience Route",
            "reason": f"The normal local route overlaps {blocked_overlap} failed or unsafe segments, so the safer reroute is the better choice.",
            "live_score": baseline_score,
            "safe_score": safe_score,
        }

    if route_comparison and route_comparison["risk_delta"] < 0:
        return {
            "label": "Recommended: Resilience Route",
            "reason": f"The resilience route lowers average flood exposure by {abs(route_comparison['risk_delta'])} while keeping the trip navigable.",
            "live_score": baseline_score,
            "safe_score": safe_score,
        }

    if mapbox_primary_route:
        return {
            "label": "Recommended: Fastest Live Route",
            "reason": "Under the current scenario, the direct live route is still acceptable and has no strong safety disadvantage.",
            "live_score": baseline_score,
            "safe_score": safe_score,
        }

    return {
        "label": "Recommended: Resilience Route",
        "reason": "Only the resilience-aware route has enough infrastructure context to give a robust recommendation here.",
        "live_score": baseline_score,
        "safe_score": safe_score,
    }


def scenario_route_snapshots(source_id, target_id, mode):
    snapshots = []
    if not source_id or not target_id:
        return snapshots

    for severity in available_severities():
        failed_ids = load_failed_road_ids(severity["graph_path"])
        result = safest_route_result(source_id, target_id, failed_ids, mode=mode)
        if result and not result.get("error"):
            metrics = result["metrics"]
            snapshots.append(
                {
                    "severity_label": severity["label"],
                    "severity_slug": severity["slug"],
                    "route_available": True,
                    "segment_count": metrics["segment_count"],
                    "avg_flood_risk": metrics["avg_flood_risk"],
                    "safety_score": route_safety_score(metrics, blocked_overlap=0),
                }
            )
        else:
            snapshots.append(
                {
                    "severity_label": severity["label"],
                    "severity_slug": severity["slug"],
                    "route_available": False,
                    "segment_count": None,
                    "avg_flood_risk": None,
                    "safety_score": 0.0,
                    "error": result.get("error") if result else "Route unavailable",
                }
            )
    return snapshots


def summarize_mapbox_route(route, index):
    geometry = route.get("geometry") or {}
    legs = route.get("legs") or []
    steps = []
    if legs:
        for step in legs[0].get("steps", [])[:14]:
            instruction = (
                step.get("maneuver", {}).get("instruction")
                or step.get("name")
                or "Continue"
            )
            steps.append(
                {
                    "instruction": instruction,
                    "distance_m": round(safe_float(step.get("distance")), 1),
                    "duration_s": round(safe_float(step.get("duration")), 1),
                }
            )

    return {
        "rank": index,
        "label": "Fastest Live Route" if index == 0 else f"Alternative Route {index}",
        "distance_km": round(safe_float(route.get("distance")) / 1000, 2),
        "duration_min": round(safe_float(route.get("duration")) / 60, 1),
        "geometry": {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": geometry, "properties": {"label": f"route_{index}"}}]},
        "steps": steps,
    }


def safest_route_result(source_id, target_id, failed_ids, mode="safe"):
    base_graph = load_base_graph()
    if base_graph is None:
        return {"error": "Base graph is not available."}

    graph = base_graph.copy()
    criticality_map = load_criticality_map()

    for road_id, attrs in criticality_map.items():
        if road_id in graph.nodes:
            graph.nodes[road_id]["criticality"] = safe_float(attrs.get("criticality"))

    if source_id not in graph.nodes:
        return {"error": f"Source road '{source_id}' was not found."}
    if target_id not in graph.nodes:
        return {"error": f"Target road '{target_id}' was not found."}
    if source_id in failed_ids:
        return {"error": f"Source road '{source_id}' is failed under the selected severity."}
    if target_id in failed_ids:
        return {"error": f"Target road '{target_id}' is failed under the selected severity."}

    blocked = set(failed_ids) - {source_id, target_id}
    graph.remove_nodes_from(node_id for node_id in blocked if node_id in graph)

    try:
        route_ids = nx.shortest_path(
            graph,
            source=source_id,
            target=target_id,
            weight=routing_weight(graph, mode),
        )
    except nx.NetworkXNoPath:
        return {"error": "No safe reroute path was found between the selected roads."}

    return {
        "route_ids": route_ids,
        "route_geojson": build_route_geojson(route_ids),
        "metrics": route_metrics(graph, route_ids),
        "mode": mode,
    }


def summarize_selected_severity(severity, failed_count):
    results = load_json(severity["results_path"]) if severity["results_path"] else None
    capacity_cascade = results.get("capacity_cascade", []) if results else []

    if results:
        initial_failures = results.get("initial_failures", failed_count)
    else:
        initial_failures = failed_count

    return {
        "label": severity["label"],
        "slug": severity["slug"],
        "initial_failures": initial_failures,
        "cascade_steps": len(capacity_cascade),
        "peak_step_failures": max(capacity_cascade) if capacity_cascade else 0,
        "capacity_cascade": capacity_cascade,
    }


def build_severity_overview():
    overview = []
    for severity in available_severities():
        failed_count = len(load_failed_road_ids(severity["graph_path"]))
        summary = summarize_selected_severity(severity, failed_count)
        overview.append(
            {
                "slug": severity["slug"],
                "label": severity["label"],
                "failed_count": failed_count,
                "initial_failures": summary["initial_failures"],
                "cascade_steps": summary["cascade_steps"],
                "peak_step_failures": summary["peak_step_failures"],
            }
        )
    return overview


@lru_cache(maxsize=1)
def cached_severity_overview():
    return build_severity_overview()


def preset_artifact_path(severity_slug):
    return dashboard_artifact_dir() / f"roads_enriched_scale_{severity_slug}.json"


def severity_overview_artifact_path():
    return dashboard_artifact_dir() / "severity_overview.json"


@lru_cache(maxsize=8)
def serialized_roads_for_preset(severity_slug):
    artifact_payload = load_json(preset_artifact_path(severity_slug))
    if artifact_payload:
        return (
            json.dumps(artifact_payload.get("roads_geojson", {})),
            int(artifact_payload.get("failed_count", 0)),
            int(artifact_payload.get("critical_count", 0)),
        )

    severity, _ = select_severity(severity_slug)
    roads_geojson = load_roads_geojson()
    base_graph = load_base_graph()
    failed_ids = load_failed_road_ids(severity["graph_path"])
    criticality_map = load_criticality_map()
    enriched_roads, critical_count = enrich_roads_geojson(
        roads_geojson,
        base_graph,
        failed_ids,
        criticality_map,
    )
    return json.dumps(enriched_roads) if enriched_roads else "null", len(failed_ids), critical_count


@lru_cache(maxsize=24)
def serialized_roads_for_custom(scale):
    simulation = simulate_scale(scale, seed=42)
    roads_geojson = load_roads_geojson()
    base_graph = load_base_graph()
    criticality_map = load_criticality_map()
    enriched_roads, critical_count = enrich_roads_geojson(
        roads_geojson,
        base_graph,
        set(simulation["failed_ids"]),
        criticality_map,
    )
    return json.dumps(enriched_roads) if enriched_roads else "null", len(simulation["failed_ids"]), critical_count


def resolve_scale_context(request):
    custom_scale = parse_custom_scale(request.GET.get("custom_scale"))
    seed = 42

    if custom_scale is not None:
        simulation = simulate_scale(custom_scale, seed=seed)
        selected = {
            "slug": str(custom_scale),
            "label": f"Custom {custom_scale}",
            "graph_path": None,
            "results_path": None,
            "custom": True,
        }
        return {
            "selected_severity": selected,
            "severity_options": available_severities(),
            "failed_ids": set(simulation["failed_ids"]),
            "severity_summary": simulation["summary"],
            "custom_scale": custom_scale,
            "custom_seed": seed,
        }

    selected_severity, severity_options = select_severity(request.GET.get("severity"))
    failed_ids = load_failed_road_ids(selected_severity["graph_path"])
    severity_summary = summarize_selected_severity(selected_severity, len(failed_ids))
    selected_severity = {**selected_severity, "custom": False}
    return {
        "selected_severity": selected_severity,
        "severity_options": severity_options,
        "failed_ids": failed_ids,
        "severity_summary": severity_summary,
        "custom_scale": None,
        "custom_seed": seed,
    }


def get_dashboard_state(request):
    scenarios = ScenarioConfig.objects.annotate(run_count=Count("runs")).order_by("name")
    selected_scenario_slug = request.GET.get("scenario")
    scale_context = resolve_scale_context(request)
    selected_severity = scale_context["selected_severity"]
    severity_options = scale_context["severity_options"]

    if selected_scenario_slug:
        selected_scenario = get_object_or_404(ScenarioConfig, slug=selected_scenario_slug)
    else:
        selected_scenario = scenarios.first()

    runs = ExperimentRun.objects.select_related("scenario").prefetch_related(
        "artifacts",
        "model_results",
        "optimization_results",
    )

    if selected_scenario:
        runs = runs.filter(scenario=selected_scenario)

    latest_training_run = runs.filter(
        run_kind=ExperimentRun.RunKind.TRAINING,
        status=ExperimentRun.Status.COMPLETED,
    ).first()

    latest_optimization_run = runs.filter(
        run_kind=ExperimentRun.RunKind.OPTIMIZATION,
        status=ExperimentRun.Status.COMPLETED,
    ).first()

    optimization_rows = []
    if latest_optimization_run:
        optimization_rows = latest_optimization_run.optimization_results.order_by(
            "budget_k", "strategy_name"
        )

    severity_summary = scale_context["severity_summary"]
    severity_overview = load_json(severity_overview_artifact_path()) or cached_severity_overview()

    if scale_context["custom_scale"] is not None:
        roads_geojson_json, failed_count, critical_count = serialized_roads_for_custom(scale_context["custom_scale"])
    else:
        roads_geojson_json, failed_count, critical_count = serialized_roads_for_preset(selected_severity["slug"])

    return {
        "scenarios": scenarios,
        "selected_scenario": selected_scenario,
        "selected_severity": selected_severity,
        "severity_options": severity_options,
        "severity_summary": severity_summary,
        "severity_overview": severity_overview,
        "custom_scale": scale_context["custom_scale"],
        "custom_seed": scale_context["custom_seed"],
        "latest_training_run": latest_training_run,
        "latest_optimization_run": latest_optimization_run,
        "optimization_rows": optimization_rows,
        "all_runs": runs[:10],
        "roads_geojson": roads_geojson_json,
        "failed_count": failed_count,
        "critical_count": critical_count,
        "critical_summary": critical_road_summary(),
    }


def dashboard(request):
    context = get_dashboard_state(request)
    return render(request, "dashboard/dashboard.html", context)


def scenario_explorer(request):
    context = get_dashboard_state(request)
    context["page_title"] = "Scenario Explorer"
    return render(request, "dashboard/scenario_explorer.html", context)


def model_results(request):
    context = get_dashboard_state(request)
    model_runs = (
        ExperimentRun.objects.select_related("scenario")
        .prefetch_related("model_results", "artifacts")
        .filter(
            run_kind=ExperimentRun.RunKind.TRAINING,
            status=ExperimentRun.Status.COMPLETED,
        )
        .order_by("-started_at")
    )

    all_model_results = []
    for run in model_runs:
        for result in run.model_results.all():
            metrics = result.metrics if isinstance(result.metrics, dict) else {}
            roc_auc = metrics.get("roc_auc")
            if roc_auc is None:
                for benchmark in BENCHMARK_MODEL_METRICS:
                    if benchmark["model_name"].lower() == result.model_name.lower():
                        roc_auc = benchmark["roc_auc"]
                        break
            all_model_results.append(
                {
                    "run": run,
                    "result": result,
                    "roc_auc": roc_auc,
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1"),
                }
            )

    benchmark_by_name = {item["model_name"]: item["roc_auc"] for item in BENCHMARK_MODEL_METRICS}
    for item in all_model_results:
        if item["roc_auc"] is not None:
            benchmark_by_name[item["result"].model_name] = item["roc_auc"]

    chart_labels = list(benchmark_by_name.keys())
    chart_values = list(benchmark_by_name.values())

    feature_importance_rows = []
    feature_importance_path = repo_root() / "data" / "rf_feature_importance.csv"
    if feature_importance_path.exists():
        feature_importance_rows = pd.read_csv(feature_importance_path).head(12).to_dict(orient="records")

    model_comparison_rows = []
    comparison_path = repo_root() / "data" / "model_comparison.json"
    comparison_payload = load_json(comparison_path)
    if isinstance(comparison_payload, list):
        model_comparison_rows = comparison_payload

    context.update(
        {
            "page_title": "Model Results",
            "model_runs": model_runs,
            "all_model_results": all_model_results,
            "benchmark_model_metrics": BENCHMARK_MODEL_METRICS,
            "model_chart_labels": json.dumps(chart_labels),
            "model_chart_values": json.dumps(chart_values),
            "feature_importance_rows": feature_importance_rows,
            "model_comparison_rows": model_comparison_rows,
        }
    )
    return render(request, "dashboard/model_results.html", context)


def optimization_analysis(request):
    context = get_dashboard_state(request)
    optimization_runs = (
        ExperimentRun.objects.select_related("scenario")
        .prefetch_related("optimization_results")
        .filter(
            run_kind=ExperimentRun.RunKind.OPTIMIZATION,
            status=ExperimentRun.Status.COMPLETED,
        )
        .order_by("-started_at")
    )

    optimization_groups = []
    for run in optimization_runs:
        rows = list(run.optimization_results.order_by("budget_k", "strategy_name"))
        optimization_groups.append({"run": run, "rows": rows})

    chart_labels = []
    greedy_values = []
    random_values = []
    greedy_resilience_values = []
    random_resilience_values = []
    best_budget_summary = None

    if optimization_groups:
        first_group_rows = optimization_groups[0]["rows"]
        grouped_by_budget = {}
        for row in first_group_rows:
            grouped_by_budget.setdefault(row.budget_k, {})[row.strategy_name] = row.mean_failed
            grouped_by_budget[row.budget_k][f"{row.strategy_name}_resilience"] = row.resilience_score

        for budget in sorted(grouped_by_budget):
            chart_labels.append(str(budget))
            greedy_values.append(grouped_by_budget[budget].get("Greedy Repair", 0))
            random_values.append(grouped_by_budget[budget].get("Random Repair", 0))
            greedy_resilience_values.append(grouped_by_budget[budget].get("Greedy Repair_resilience", 0))
            random_resilience_values.append(grouped_by_budget[budget].get("Random Repair_resilience", 0))

        reductions = []
        for budget in sorted(grouped_by_budget):
            greedy = grouped_by_budget[budget].get("Greedy Repair", 0)
            random_value = grouped_by_budget[budget].get("Random Repair", 0)
            reductions.append(
                {
                    "budget_k": budget,
                    "failed_reduction": round(random_value - greedy, 2),
                    "resilience_gain": round(
                        (grouped_by_budget[budget].get("Greedy Repair_resilience") or 0)
                        - (grouped_by_budget[budget].get("Random Repair_resilience") or 0),
                        4,
                    ),
                }
            )
        if reductions:
            best_budget_summary = max(reductions, key=lambda item: item["failed_reduction"])

    context.update(
        {
            "page_title": "Optimization Analysis",
            "optimization_runs": optimization_runs,
            "optimization_groups": optimization_groups,
            "best_budget_summary": best_budget_summary,
            "optimization_chart_labels": json.dumps(chart_labels),
            "optimization_greedy_values": json.dumps(greedy_values),
            "optimization_random_values": json.dumps(random_values),
            "optimization_greedy_resilience_values": json.dumps(greedy_resilience_values),
            "optimization_random_resilience_values": json.dumps(random_resilience_values),
        }
    )
    return render(request, "dashboard/optimization_analysis.html", context)


def route_planner(request):
    context = get_dashboard_state(request)
    source_id = request.GET.get("source_road_id", "").strip()
    target_id = request.GET.get("target_road_id", "").strip()
    origin_name = request.GET.get("origin_name", "").strip()
    destination_name = request.GET.get("destination_name", "").strip()
    travel_profile = request.GET.get("travel_profile", "driving").strip() or "driving"
    mode = request.GET.get("mode", "safe").strip() or "safe"
    origin_lng = request.GET.get("origin_lng", "").strip()
    origin_lat = request.GET.get("origin_lat", "").strip()
    destination_lng = request.GET.get("destination_lng", "").strip()
    destination_lat = request.GET.get("destination_lat", "").strip()

    failed_ids = set()
    if context["custom_scale"] is not None:
        failed_ids = simulate_scale(context["custom_scale"], seed=context["custom_seed"])["failed_ids"]
        failed_ids = set(failed_ids)
    else:
        severity, _ = select_severity(context["selected_severity"]["slug"])
        failed_ids = load_failed_road_ids(severity["graph_path"])

    route_result = None
    baseline_route_result = None
    route_comparison = None
    route_segments = []
    baseline_route_segments = []
    safe_route_reasons = []
    blocked_baseline_overlap = 0
    baseline_safety_score = None
    resilient_safety_score = None
    route_recommendation = None
    scenario_snapshots = []
    mapbox_routes = []
    mapbox_primary_route = None
    mapbox_error = ""
    origin_point = None
    destination_point = None

    if origin_lng and origin_lat and destination_lng and destination_lat:
        origin_point = (safe_float(origin_lng), safe_float(origin_lat))
        destination_point = (safe_float(destination_lng), safe_float(destination_lat))
        mapbox_payload = mapbox_directions(origin_point, destination_point, travel_profile=travel_profile)
        if mapbox_payload and mapbox_payload.get("routes"):
            mapbox_routes = [
                summarize_mapbox_route(route, index)
                for index, route in enumerate(mapbox_payload.get("routes", []))
            ]
            mapbox_primary_route = mapbox_routes[0]
        elif mapbox_enabled():
            mapbox_error = "Live directions could not be retrieved from Mapbox for the selected places."

        if not source_id:
            source_id = nearest_road_id(origin_point[0], origin_point[1])
        if not target_id:
            target_id = nearest_road_id(destination_point[0], destination_point[1])

    if source_id and target_id:
        route_result = safest_route_result(source_id, target_id, failed_ids, mode=mode)
        baseline_route_result = safest_route_result(source_id, target_id, set(), mode=mode)
        if (
            route_result
            and baseline_route_result
            and not route_result.get("error")
            and not baseline_route_result.get("error")
        ):
            route_comparison = {
                "length_delta": round(
                    route_result["metrics"]["total_length"] - baseline_route_result["metrics"]["total_length"],
                    2,
                ),
                "risk_delta": round(
                    route_result["metrics"]["avg_flood_risk"] - baseline_route_result["metrics"]["avg_flood_risk"],
                    4,
                ),
                "segment_delta": route_result["metrics"]["segment_count"] - baseline_route_result["metrics"]["segment_count"],
            }
            route_segments = [
                {"id": road_id, "label": road_display_label(road_id)}
                for road_id in route_result["route_ids"]
            ]
            baseline_route_segments = [
                {"id": road_id, "label": road_display_label(road_id)}
                for road_id in baseline_route_result["route_ids"]
            ]
            blocked_baseline_overlap = sum(
                1 for road_id in baseline_route_result["route_ids"]
                if road_id in failed_ids and road_id not in {source_id, target_id}
            )
            baseline_safety_score = route_safety_score(
                baseline_route_result["metrics"],
                blocked_overlap=blocked_baseline_overlap,
            )
            resilient_safety_score = route_safety_score(route_result["metrics"], blocked_overlap=0)
            safe_route_reasons = [
                {
                    "title": "Failed roads excluded",
                    "detail": f"The safe route avoids {len(failed_ids)} failed road segments active under severity {context['selected_severity']['slug']}.",
                },
                {
                    "title": "Flood risk penalized",
                    "detail": f"The {mode} mode adds extra cost to segments with higher flood risk, so the reroute tries to move away from unstable corridors.",
                },
                {
                    "title": "Critical links considered",
                    "detail": "Critical roads receive extra penalty so the route is less likely to depend on high-impact bottlenecks during disruption.",
                },
            ]
            if route_comparison["risk_delta"] < 0:
                safe_route_reasons.append(
                    {
                        "title": "Lower risk than local baseline",
                        "detail": f"This reroute reduces average flood risk by {abs(route_comparison['risk_delta'])} compared with the graph baseline route.",
                    }
                )
            if route_comparison["length_delta"] > 0:
                safe_route_reasons.append(
                    {
                        "title": "Safety over shortest path",
                        "detail": f"The route is {route_comparison['length_delta']} units longer because it trades extra distance for lower infrastructure risk.",
                    }
                )
            if blocked_baseline_overlap > 0:
                safe_route_reasons.append(
                    {
                        "title": "Unsafe baseline overlap detected",
                        "detail": f"The local baseline route touches {blocked_baseline_overlap} segments that are failed under the selected severity, so the resilience route is safer for this trip.",
                    }
                )
            route_recommendation = build_route_recommendation(
                mapbox_primary_route,
                route_result,
                route_comparison,
                blocked_baseline_overlap,
            )

    if source_id and target_id:
        scenario_snapshots = scenario_route_snapshots(source_id, target_id, mode)

    context.update(
        {
            "page_title": "Safe Rerouting",
            "mapbox_enabled": mapbox_enabled(),
            "mapbox_public_token": mapbox_public_token(),
            "mapbox_error": mapbox_error,
            "mapbox_primary_route": mapbox_primary_route,
            "mapbox_routes": mapbox_routes,
            "route_result": route_result,
            "baseline_route_result": baseline_route_result,
            "route_source": source_id,
            "route_target": target_id,
            "route_source_label": origin_name or road_display_label(source_id),
            "route_target_label": destination_name or road_display_label(target_id),
            "route_source_road_label": road_display_label(source_id),
            "route_target_road_label": road_display_label(target_id),
            "origin_name": origin_name,
            "destination_name": destination_name,
            "origin_lng": origin_lng,
            "origin_lat": origin_lat,
            "destination_lng": destination_lng,
            "destination_lat": destination_lat,
            "travel_profile": travel_profile,
            "route_mode": mode,
            "route_mode_options": [
                ("safe", "Safest"),
                ("balanced", "Balanced"),
                ("fast", "Fastest Practical"),
                ("emergency", "Emergency Response"),
            ],
            "travel_profile_options": [
                ("driving", "Driving With Traffic"),
                ("walking", "Walking"),
                ("cycling", "Cycling"),
            ],
            "route_examples": route_examples(),
            "route_comparison": route_comparison,
            "route_segments": route_segments,
            "baseline_route_segments": baseline_route_segments,
            "safe_route_reasons": safe_route_reasons,
            "blocked_baseline_overlap": blocked_baseline_overlap,
            "baseline_safety_score": baseline_safety_score,
            "resilient_safety_score": resilient_safety_score,
            "route_recommendation": route_recommendation,
            "scenario_snapshots": scenario_snapshots,
            "route_geojson": json.dumps(route_result["route_geojson"]) if route_result and not route_result.get("error") else "null",
            "baseline_route_geojson": json.dumps(baseline_route_result["route_geojson"]) if baseline_route_result and not baseline_route_result.get("error") else "null",
            "mapbox_primary_geojson": json.dumps(mapbox_primary_route["geometry"]) if mapbox_primary_route else "null",
            "mapbox_alternative_geojsons": json.dumps([route["geometry"] for route in mapbox_routes[1:3]]) if mapbox_routes else "[]",
            "mapbox_origin_point": json.dumps([origin_point[1], origin_point[0]]) if origin_point else "null",
            "mapbox_destination_point": json.dumps([destination_point[1], destination_point[0]]) if destination_point else "null",
            "route_mode_descriptions": {
                "safe": "Prioritizes lower flood risk and lower criticality, even if the route becomes longer.",
                "balanced": "Trades off safety and travel burden for a more practical day-to-day reroute.",
                "fast": "Leans more toward short paths and lighter traffic while still penalizing risky roads.",
                "emergency": "Strongly avoids risky and critical segments for emergency-response style routing.",
            },
        }
    )
    return render(request, "dashboard/route_planner.html", context)


def route_search(request):
    query = request.GET.get("q", "")
    return JsonResponse({"results": search_route_options(query)})


def run_detail(request, slug):
    run = get_object_or_404(
        ExperimentRun.objects.select_related("scenario").prefetch_related(
            "artifacts",
            "model_results",
            "optimization_results",
        ),
        slug=slug,
    )

    optimization_rows = OptimizationResult.objects.filter(run=run).order_by("budget_k", "strategy_name")

    context = {
        "run": run,
        "optimization_rows": optimization_rows,
    }
    return render(request, "dashboard/run_detail.html", context)
