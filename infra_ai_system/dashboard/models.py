from pathlib import Path

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.text import slugify


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class ScenarioConfig(TimeStampedModel):
    name = models.CharField(max_length=120, unique=True)
    slug = models.SlugField(max_length=140, unique=True, blank=True)
    description = models.TextField(blank=True)
    rainfall_year = models.PositiveIntegerField(default=2020)
    severity_scale = models.FloatField(default=1.0)
    capacity_factor = models.FloatField(default=1.25)
    overload_factor = models.FloatField(default=1.5)
    max_steps = models.PositiveIntegerField(default=20)
    random_seed = models.PositiveIntegerField(null=True, blank=True)
    parameters = models.JSONField(default=dict, blank=True)
    is_baseline = models.BooleanField(default=False)

    class Meta:
        ordering = ["name"]
        verbose_name = "scenario configuration"
        verbose_name_plural = "scenario configurations"

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)


class ExperimentRun(TimeStampedModel):
    class RunKind(models.TextChoices):
        CASCADE = "cascade", "Cascade Simulation"
        DATASET = "dataset", "Dataset Generation"
        TRAINING = "training", "Model Training"
        OPTIMIZATION = "optimization", "Repair Optimization"
        EVALUATION = "evaluation", "Evaluation"
        DASHBOARD = "dashboard", "Dashboard Snapshot"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        ARCHIVED = "archived", "Archived"

    title = models.CharField(max_length=160)
    slug = models.SlugField(max_length=180, unique=True, blank=True)
    run_kind = models.CharField(max_length=20, choices=RunKind.choices)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.DRAFT,
    )
    scenario = models.ForeignKey(
        ScenarioConfig,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="runs",
    )
    run_identifier = models.CharField(max_length=80, unique=True)
    source_commit = models.CharField(max_length=64, blank=True)
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    input_summary = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at", "-created_at"]

    def __str__(self):
        return f"{self.title} [{self.run_identifier}]"

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(f"{self.title}-{self.run_identifier}")
        super().save(*args, **kwargs)


class RunArtifact(TimeStampedModel):
    class ArtifactKind(models.TextChoices):
        GRAPH = "graph", "Graph"
        DATASET = "dataset", "Dataset"
        MODEL = "model", "Model"
        METRICS = "metrics", "Metrics"
        GEOJSON = "geojson", "GeoJSON"
        PLOT = "plot", "Plot"
        REPORT = "report", "Report"
        OTHER = "other", "Other"

    run = models.ForeignKey(
        ExperimentRun,
        on_delete=models.CASCADE,
        related_name="artifacts",
    )
    label = models.CharField(max_length=120)
    artifact_kind = models.CharField(max_length=20, choices=ArtifactKind.choices)
    relative_path = models.CharField(
        max_length=255,
        help_text="Path relative to the repository root.",
    )
    file_format = models.CharField(max_length=32, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["label"]
        unique_together = ("run", "label", "relative_path")

    def __str__(self):
        return f"{self.run.run_identifier}: {self.label}"

    @property
    def absolute_path(self):
        project_root = Path(settings.BASE_DIR).parent
        return project_root / self.relative_path


class ModelResult(TimeStampedModel):
    class TaskType(models.TextChoices):
        NODE_FAILURE = "node_failure", "Node Failure Prediction"
        TEMPORAL_FAILURE = "temporal_failure", "Temporal Failure Prediction"
        RESILIENCE = "resilience", "Resilience Estimation"

    run = models.ForeignKey(
        ExperimentRun,
        on_delete=models.CASCADE,
        related_name="model_results",
    )
    model_name = models.CharField(max_length=120)
    model_family = models.CharField(max_length=80)
    dataset_name = models.CharField(max_length=120)
    task_type = models.CharField(max_length=32, choices=TaskType.choices)
    train_rows = models.PositiveIntegerField(default=0)
    test_rows = models.PositiveIntegerField(default=0)
    metrics = models.JSONField(default=dict, blank=True)
    hyperparameters = models.JSONField(default=dict, blank=True)
    model_artifact = models.ForeignKey(
        RunArtifact,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="trained_models",
    )
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["model_family", "model_name"]

    def __str__(self):
        return f"{self.model_name} ({self.get_task_type_display()})"


class OptimizationResult(TimeStampedModel):
    run = models.ForeignKey(
        ExperimentRun,
        on_delete=models.CASCADE,
        related_name="optimization_results",
    )
    strategy_name = models.CharField(max_length=120)
    budget_k = models.PositiveIntegerField()
    mean_failed = models.FloatField()
    std_failed = models.FloatField(default=0.0)
    resilience_score = models.FloatField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["strategy_name", "budget_k"]
        unique_together = ("run", "strategy_name", "budget_k")

    def __str__(self):
        return f"{self.strategy_name} @ K={self.budget_k}"
