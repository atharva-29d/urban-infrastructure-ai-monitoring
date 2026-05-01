import json
from pathlib import Path

from django.core.management.base import BaseCommand

from dashboard import views


class Command(BaseCommand):
    help = "Export precomputed dashboard artifacts for faster research-page loading."

    def handle(self, *args, **options):
        artifact_dir = views.dashboard_artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)

        severity_overview = views.build_severity_overview()
        severity_overview_path = views.severity_overview_artifact_path()
        self.write_json(severity_overview_path, severity_overview)
        self.stdout.write(self.style.SUCCESS(f"Wrote {severity_overview_path}"))

        roads_geojson = views.load_roads_geojson()
        base_graph = views.load_base_graph()
        criticality_map = views.load_criticality_map()

        for severity in views.available_severities():
            failed_ids = views.load_failed_road_ids(severity["graph_path"])
            enriched_roads, critical_count = views.enrich_roads_geojson(
                roads_geojson,
                base_graph,
                failed_ids,
                criticality_map,
            )

            payload = {
                "severity": severity["slug"],
                "label": severity["label"],
                "failed_count": len(failed_ids),
                "critical_count": critical_count,
                "roads_geojson": enriched_roads,
            }

            target_path = views.preset_artifact_path(severity["slug"])
            self.write_json(target_path, payload)
            self.stdout.write(self.style.SUCCESS(f"Wrote {target_path}"))

    def write_json(self, path: Path, payload):
        payload = self.sanitize(payload)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, allow_nan=False)

    def sanitize(self, value):
        if isinstance(value, dict):
            return {key: self.sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.sanitize(item) for item in value]
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return 0.0
        return value
