import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from dashboard.models import (
    ExperimentRun,
    ModelResult,
    OptimizationResult,
    RunArtifact,
    ScenarioConfig,
)

BENCHMARK_MODEL_RESULTS = [
    {
        "model_name": "Random Forest",
        "model_family": "tabular",
        "task_type": ModelResult.TaskType.NODE_FAILURE,
        "roc_auc": 0.62,
        "notes": "Tabular baseline from the project benchmark table.",
    },
    {
        "model_name": "Temporal Neural Network",
        "model_family": "temporal",
        "task_type": ModelResult.TaskType.TEMPORAL_FAILURE,
        "roc_auc": 0.84,
        "notes": "Temporal benchmark from the project benchmark table.",
    },
    {
        "model_name": "Graph Neural Network",
        "model_family": "graph",
        "task_type": ModelResult.TaskType.NODE_FAILURE,
        "roc_auc": 0.51,
        "notes": "Graph benchmark from the project benchmark table.",
    },
    {
        "model_name": "Spatio-Temporal GNN",
        "model_family": "graph-temporal",
        "task_type": ModelResult.TaskType.TEMPORAL_FAILURE,
        "roc_auc": 0.63,
        "notes": "Spatio-temporal benchmark from the project benchmark table.",
    },
]


class Command(BaseCommand):
    help = "Ingest existing research outputs into Django models."

    def handle(self, *args, **options):
        repo_root = Path(__file__).resolve().parents[4]
        data_dir = repo_root / "data"
        graphs_dir = data_dir / "graphs"
        rf_metrics_path = data_dir / "rf_model_metrics.json"

        scenario, _ = ScenarioConfig.objects.update_or_create(
            slug="pune-extreme-flood-2020",
            defaults={
                "name": "Pune Extreme Flood 2020",
                "description": "Extreme flood scenario built from Pune road graph and rainfall-informed flood risk.",
                "rainfall_year": 2020,
                "severity_scale": 2.5,
                "capacity_factor": 1.1,
                "overload_factor": 1.8,
                "max_steps": 20,
                "random_seed": 42,
                "parameters": {
                    "city": "Pune",
                    "graph_file": "data/graphs/pune_base_graph_weather.gpickle",
                },
                "is_baseline": False,
            },
        )

        cascade_run, _ = ExperimentRun.objects.update_or_create(
            run_identifier="cascade-pune-extreme-2020",
            defaults={
                "title": "Pune Extreme Flood Cascade",
                "run_kind": ExperimentRun.RunKind.CASCADE,
                "status": ExperimentRun.Status.COMPLETED,
                "scenario": scenario,
                "metrics": self.load_cascade_metrics(graphs_dir / "run_results_weather_2.5.json"),
                "input_summary": {
                    "graph": "data/graphs/pune_base_graph_weather.gpickle",
                    "result_file": "data/graphs/run_results_weather_2.5.json",
                },
                "notes": "Imported from precomputed research artifacts.",
            },
        )

        self.upsert_artifact(
            cascade_run,
            "Cascade Metrics",
            RunArtifact.ArtifactKind.METRICS,
            "data/graphs/run_results_weather_2.5.json",
            "json",
        )
        self.upsert_artifact(
            cascade_run,
            "Post Cascade Graph",
            RunArtifact.ArtifactKind.GRAPH,
            "data/graphs/pune_after_cascade_weather_2.5.gpickle",
            "gpickle",
        )

        optimization_run, _ = ExperimentRun.objects.update_or_create(
            run_identifier="optimization-pune-severe-2020",
            defaults={
                "title": "Pune Repair Optimization",
                "run_kind": ExperimentRun.RunKind.OPTIMIZATION,
                "status": ExperimentRun.Status.COMPLETED,
                "scenario": scenario,
                "metrics": {"source": "data/graphs/opt_vs_random_scale_1.2.json"},
                "input_summary": {
                    "critical_roads": "data/graphs/critical_roads.csv",
                    "optimization_file": "data/graphs/opt_vs_random_scale_1.2.json",
                    "repair_file": "data/graphs/repair_impact_scale_1.2.json",
                },
                "notes": "Greedy vs random repair comparison imported from saved outputs.",
            },
        )

        self.upsert_artifact(
            optimization_run,
            "Optimization Comparison",
            RunArtifact.ArtifactKind.METRICS,
            "data/graphs/opt_vs_random_scale_1.2.json",
            "json",
        )
        self.upsert_artifact(
            optimization_run,
            "Repair Impact",
            RunArtifact.ArtifactKind.METRICS,
            "data/graphs/repair_impact_scale_1.2.json",
            "json",
        )

        self.ingest_optimization_results(optimization_run, graphs_dir / "opt_vs_random_scale_1.2.json")

        training_run, _ = ExperimentRun.objects.update_or_create(
            run_identifier="training-tabular-baseline-v1",
            defaults={
                "title": "Tabular Failure Prediction Baseline",
                "run_kind": ExperimentRun.RunKind.TRAINING,
                "status": ExperimentRun.Status.COMPLETED,
                "scenario": scenario,
                "metrics": self.load_dataset_summary(data_dir),
                "input_summary": {
                    "dataset": "data/ml_dataset.csv",
                    "temporal_dataset": "data/temporal_dataset.csv",
                },
                "notes": "Imported dataset-backed training summary for dashboard integration.",
            },
        )

        model_artifact = self.upsert_artifact(
            training_run,
            "Random Forest Model",
            RunArtifact.ArtifactKind.MODEL,
            "data/rf_model.pkl",
            "pkl",
        )

        self.upsert_artifact(
            training_run,
            "ML Dataset",
            RunArtifact.ArtifactKind.DATASET,
            "data/ml_dataset.csv",
            "csv",
        )
        self.upsert_artifact(
            training_run,
            "Temporal Dataset",
            RunArtifact.ArtifactKind.DATASET,
            "data/temporal_dataset.csv",
            "csv",
        )

        dataset_summary = self.load_dataset_summary(data_dir)

        for benchmark in BENCHMARK_MODEL_RESULTS:
            dataset_name = (
                "temporal_dataset.csv"
                if benchmark["task_type"] == ModelResult.TaskType.TEMPORAL_FAILURE
                else "ml_dataset.csv"
            )
            train_rows = (
                dataset_summary["temporal_dataset_rows"]
                if dataset_name == "temporal_dataset.csv"
                else dataset_summary["ml_dataset_rows"]
            )
            ModelResult.objects.update_or_create(
                run=training_run,
                model_name=benchmark["model_name"],
                dataset_name=dataset_name,
                defaults={
                    "model_family": benchmark["model_family"],
                    "task_type": benchmark["task_type"],
                    "train_rows": train_rows,
                    "test_rows": 0,
                    "metrics": {
                        "roc_auc": benchmark["roc_auc"],
                        "status": "benchmark imported into structured results",
                    },
                    "hyperparameters": (
                        {
                            "n_estimators": 300,
                            "max_depth": 15,
                            "class_weight": "balanced",
                        }
                        if benchmark["model_name"] == "Random Forest"
                        else {}
                    ),
                    "model_artifact": model_artifact if benchmark["model_name"] == "Random Forest" else None,
                    "notes": benchmark["notes"],
                },
            )

        if rf_metrics_path.exists():
            with open(rf_metrics_path, "r", encoding="utf-8") as file:
                rf_metrics = json.load(file)

            ModelResult.objects.update_or_create(
                run=training_run,
                model_name="Random Forest",
                dataset_name="ml_dataset.csv",
                defaults={
                    "model_family": "tabular",
                    "task_type": ModelResult.TaskType.NODE_FAILURE,
                    "train_rows": rf_metrics.get("train_rows", dataset_summary["ml_dataset_rows"]),
                    "test_rows": rf_metrics.get("test_rows", 0),
                    "metrics": {
                        "roc_auc": rf_metrics.get("roc_auc"),
                        "pr_auc": rf_metrics.get("pr_auc"),
                        "f1": rf_metrics.get("f1"),
                        "balanced_accuracy": rf_metrics.get("balanced_accuracy"),
                        "threshold": rf_metrics.get("threshold"),
                        "best_validation_pr_auc": rf_metrics.get("best_validation_pr_auc"),
                    },
                    "hyperparameters": rf_metrics.get("best_params", {}),
                    "model_artifact": model_artifact,
                    "notes": "Loaded from tuned Random Forest training metrics.",
                },
            )

        self.stdout.write(self.style.SUCCESS("Research data ingestion completed."))

    def load_cascade_metrics(self, path: Path):
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        failures = data.get("capacity_cascade", [])
        return {
            "initial_failures": data.get("initial_failures", 0),
            "cascade_steps": len(failures),
            "total_cascade_failures": sum(failures),
            "peak_step_failures": max(failures) if failures else 0,
            "capacity_cascade": failures,
        }

    def load_dataset_summary(self, data_dir: Path):
        summary = {
            "ml_dataset_rows": 0,
            "temporal_dataset_rows": 0,
        }

        ml_path = data_dir / "ml_dataset.csv"
        temporal_path = data_dir / "temporal_dataset.csv"

        if ml_path.exists():
            summary["ml_dataset_rows"] = len(pd.read_csv(ml_path))

        if temporal_path.exists():
            summary["temporal_dataset_rows"] = len(pd.read_csv(temporal_path))

        return summary

    def ingest_optimization_results(self, run, path: Path):
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as file:
            results = json.load(file)

        for row in results:
            OptimizationResult.objects.update_or_create(
                run=run,
                strategy_name="Greedy Repair",
                budget_k=row["K"],
                defaults={
                    "mean_failed": row["greedy_failed"],
                    "std_failed": 0.0,
                    "metadata": {"baseline": "random", "comparison_value": row["random_failed"]},
                },
            )

            OptimizationResult.objects.update_or_create(
                run=run,
                strategy_name="Random Repair",
                budget_k=row["K"],
                defaults={
                    "mean_failed": row["random_failed"],
                    "std_failed": 0.0,
                    "metadata": {"baseline": "greedy", "comparison_value": row["greedy_failed"]},
                },
            )

    def upsert_artifact(self, run, label, artifact_kind, relative_path, file_format):
        artifact, _ = RunArtifact.objects.update_or_create(
            run=run,
            label=label,
            relative_path=relative_path,
            defaults={
                "artifact_kind": artifact_kind,
                "file_format": file_format,
            },
        )
        return artifact
