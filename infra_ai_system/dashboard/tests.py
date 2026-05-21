from pathlib import Path
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from .models import ExperimentRun, ModelResult, OptimizationResult, RunArtifact, ScenarioConfig
from . import views


class ResearchModelsTest(TestCase):
    def setUp(self):
        self.scenario = ScenarioConfig.objects.create(
            name="Pune Extreme Flood 2020",
            description="Extreme rainfall scenario for dashboard integration.",
            rainfall_year=2020,
            severity_scale=2.5,
            capacity_factor=1.1,
            overload_factor=1.8,
            max_steps=20,
            random_seed=42,
            parameters={"city": "Pune", "graph": "weather-aware"},
        )
        self.run = ExperimentRun.objects.create(
            title="Extreme Flood Cascade Run",
            run_kind=ExperimentRun.RunKind.CASCADE,
            status=ExperimentRun.Status.COMPLETED,
            scenario=self.scenario,
            run_identifier="cascade-2020-extreme-seed42",
            metrics={"initial_failures": 34792, "cascade_steps": 5},
        )

    def test_slugs_are_generated(self):
        self.assertEqual(self.scenario.slug, "pune-extreme-flood-2020")
        self.assertEqual(
            self.run.slug,
            "extreme-flood-cascade-run-cascade-2020-extreme-seed42",
        )

    def test_artifact_resolves_repo_relative_path(self):
        artifact = RunArtifact.objects.create(
            run=self.run,
            label="Cascade Results JSON",
            artifact_kind=RunArtifact.ArtifactKind.METRICS,
            relative_path="data/graphs/run_results_weather_2.5.json",
            file_format="json",
        )

        self.assertEqual(
            artifact.absolute_path,
            Path("C:/PycharmProjects/EDI/data/graphs/run_results_weather_2.5.json"),
        )

    def test_related_results_attach_to_run(self):
        artifact = RunArtifact.objects.create(
            run=self.run,
            label="Random Forest Model",
            artifact_kind=RunArtifact.ArtifactKind.MODEL,
            relative_path="artifacts/models/rf_model.pkl",
            file_format="pkl",
        )
        model_result = ModelResult.objects.create(
            run=self.run,
            model_name="Random Forest",
            model_family="tabular",
            dataset_name="ml_dataset_v1",
            task_type=ModelResult.TaskType.NODE_FAILURE,
            train_rows=1000,
            test_rows=250,
            metrics={"roc_auc": 0.81},
            hyperparameters={"n_estimators": 300},
            model_artifact=artifact,
        )
        optimization_result = OptimizationResult.objects.create(
            run=self.run,
            strategy_name="Greedy Repair",
            budget_k=500,
            mean_failed=57181.0,
            std_failed=120.5,
            resilience_score=0.08,
            metadata={"replications": 5},
        )

        self.assertEqual(model_result.run, self.run)
        self.assertEqual(model_result.model_artifact, artifact)
        self.assertEqual(optimization_result.run, self.run)
        self.assertEqual(self.run.artifacts.count(), 1)
        self.assertEqual(self.run.model_results.count(), 1)
        self.assertEqual(self.run.optimization_results.count(), 1)


class DashboardViewsTest(TestCase):
    def setUp(self):
        self.scenario = ScenarioConfig.objects.create(
            name="Pune Baseline Scenario",
            rainfall_year=2020,
            severity_scale=0.7,
            capacity_factor=1.5,
            overload_factor=1.2,
            parameters={"city": "Pune"},
        )
        self.training_run = ExperimentRun.objects.create(
            title="Training Run",
            run_kind=ExperimentRun.RunKind.TRAINING,
            status=ExperimentRun.Status.COMPLETED,
            scenario=self.scenario,
            run_identifier="training-run-01",
            metrics={"ml_dataset_rows": 100, "temporal_dataset_rows": 50},
        )
        self.optimization_run = ExperimentRun.objects.create(
            title="Optimization Run",
            run_kind=ExperimentRun.RunKind.OPTIMIZATION,
            status=ExperimentRun.Status.COMPLETED,
            scenario=self.scenario,
            run_identifier="optimization-run-01",
        )
        self.detail_run = ExperimentRun.objects.create(
            title="Cascade Run",
            run_kind=ExperimentRun.RunKind.CASCADE,
            status=ExperimentRun.Status.COMPLETED,
            scenario=self.scenario,
            run_identifier="cascade-run-01",
            metrics={"initial_failures": 10},
        )
        artifact = RunArtifact.objects.create(
            run=self.training_run,
            label="Model Artifact",
            artifact_kind=RunArtifact.ArtifactKind.MODEL,
            relative_path="data/rf_model.pkl",
            file_format="pkl",
        )
        ModelResult.objects.create(
            run=self.training_run,
            model_name="Random Forest",
            model_family="tabular",
            dataset_name="ml_dataset.csv",
            task_type=ModelResult.TaskType.NODE_FAILURE,
            train_rows=100,
            test_rows=20,
            metrics={"roc_auc": 0.8},
            model_artifact=artifact,
        )
        OptimizationResult.objects.create(
            run=self.optimization_run,
            strategy_name="Greedy Repair",
            budget_k=500,
            mean_failed=42.0,
            std_failed=1.0,
        )

    def test_dashboard_page_renders(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Infrastructure Map")
        self.assertContains(response, "Severity Selector")
        self.assertContains(response, "Scale 0.7")
        self.assertContains(response, "Run custom scale")

    def test_dashboard_accepts_custom_scale(self):
        response = self.client.get(reverse("dashboard"), {"custom_scale": "0.9"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Custom 0.9")

    def test_serialized_roads_for_preset_prefers_exported_artifact(self):
        artifact_path = Path("C:/PycharmProjects/EDI/infra_ai_system/dashboard/test_roads_enriched_scale_0.7.json")
        artifact_path.write_text(
            '{"roads_geojson":{"type":"FeatureCollection","features":[]},"failed_count":12,"critical_count":7}',
            encoding="utf-8",
        )

        try:
            views.serialized_roads_for_preset.cache_clear()
            views._load_json_cached.cache_clear()
            with patch("dashboard.views.preset_artifact_path", return_value=artifact_path):
                roads_json, failed_count, critical_count = views.serialized_roads_for_preset("0.7")

            self.assertEqual(failed_count, 12)
            self.assertEqual(critical_count, 7)
            self.assertIn('"FeatureCollection"', roads_json)
        finally:
            if artifact_path.exists():
                artifact_path.unlink()

    def test_scenario_explorer_page_renders(self):
        response = self.client.get(reverse("scenario_explorer"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Scenario Explorer")
        self.assertContains(response, "Cascade Interpretation")
        self.assertContains(response, "Scale Comparison Chart")

    def test_model_results_page_renders(self):
        response = self.client.get(reverse("model_results"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Model Results")
        self.assertContains(response, "Random Forest")
        self.assertContains(response, "Model Performance Chart")

    def test_optimization_analysis_page_renders(self):
        response = self.client.get(reverse("optimization_analysis"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Optimization Analysis")
        self.assertContains(response, "Greedy Repair")
        self.assertContains(response, "Budget Sensitivity Chart")

    def test_route_planner_page_renders(self):
        response = self.client.get(reverse("route_planner"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Resilience-Aware Smart Rerouting")
        self.assertContains(response, "Generate Smart Routes")

    def test_route_search_endpoint_renders_json(self):
        response = self.client.get(reverse("route_search"), {"q": "road_5"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json())

    def test_run_detail_page_renders(self):
        response = self.client.get(reverse("run_detail", args=[self.detail_run.slug]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cascade Run")
