from django.contrib import admin

from .models import (
    ExperimentRun,
    ModelResult,
    OptimizationResult,
    RunArtifact,
    ScenarioConfig,
)


@admin.register(ScenarioConfig)
class ScenarioConfigAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "rainfall_year",
        "severity_scale",
        "capacity_factor",
        "overload_factor",
        "is_baseline",
    )
    list_filter = ("rainfall_year", "is_baseline")
    search_fields = ("name", "description")
    prepopulated_fields = {"slug": ("name",)}


class RunArtifactInline(admin.TabularInline):
    model = RunArtifact
    extra = 0


class ModelResultInline(admin.TabularInline):
    model = ModelResult
    extra = 0


class OptimizationResultInline(admin.TabularInline):
    model = OptimizationResult
    extra = 0


@admin.register(ExperimentRun)
class ExperimentRunAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "run_identifier",
        "run_kind",
        "status",
        "scenario",
        "started_at",
        "completed_at",
    )
    list_filter = ("run_kind", "status", "scenario")
    search_fields = ("title", "run_identifier", "source_commit", "notes")
    prepopulated_fields = {"slug": ("title",)}
    inlines = (RunArtifactInline, ModelResultInline, OptimizationResultInline)


@admin.register(RunArtifact)
class RunArtifactAdmin(admin.ModelAdmin):
    list_display = ("label", "run", "artifact_kind", "file_format", "relative_path")
    list_filter = ("artifact_kind", "file_format")
    search_fields = ("label", "relative_path", "run__run_identifier")


@admin.register(ModelResult)
class ModelResultAdmin(admin.ModelAdmin):
    list_display = (
        "model_name",
        "model_family",
        "dataset_name",
        "task_type",
        "run",
    )
    list_filter = ("task_type", "model_family")
    search_fields = ("model_name", "dataset_name", "run__run_identifier")


@admin.register(OptimizationResult)
class OptimizationResultAdmin(admin.ModelAdmin):
    list_display = (
        "strategy_name",
        "budget_k",
        "mean_failed",
        "std_failed",
        "resilience_score",
        "run",
    )
    list_filter = ("strategy_name",)
    search_fields = ("strategy_name", "run__run_identifier")
