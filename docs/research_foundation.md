# Research Foundation Plan

This repository should evolve into a reproducible research system with Django as the main product surface.

## Canonical Repository Shape

Use the root repository as the single source of truth and treat `urban-infrastructure-ai-monitoring/` as a legacy duplicate to be retired.

Target layout:

```text
EDI/
  data/
    raw/
      weather/
      gis/
    processed/
      graphs/
      features/
      datasets/
  artifacts/
    experiments/<run_id>/
      config.json
      metrics.json
      plots/
      models/
  graph/
  risk/
  ml/
  optimization/
  visualization/
  infra_ai_system/
    dashboard/
    infra_ai_system/
  docs/
```

## Research System Boundaries

- `risk/`: rainfall ingestion and flood-risk scoring
- `graph/`: graph construction and cascade simulation
- `ml/`: dataset generation, training, evaluation, and ablations
- `optimization/`: repair policies and budget studies
- `infra_ai_system/dashboard/`: Django views for experiment browsing and interpretation

## Django Data Model Goals

The dashboard should present experiment-backed results rather than hardcoded files.

Core entities:

- `ScenarioConfig`: one reproducible scenario definition
- `ExperimentRun`: one execution of a scenario or research stage
- `RunArtifact`: saved outputs for a run
- `ModelResult`: evaluation metadata for trained models
- `OptimizationResult`: budget-vs-outcome records for repair strategies

## Immediate Next Steps

1. Wire Django views to these models instead of reading fixed JSON files.
2. Add management commands to register runs and ingest existing result files.
3. Create experiment detail pages for cascade, model, and optimization outputs.
4. Add tests around missing artifacts, invalid configs, and result rendering.

## Current Dashboard Workflow

The Django research app now supports:

- preset scale switching for `0.7`, `1.2`, `1.5`, and `2.5`
- custom on-the-fly scale simulation from the UI
- dedicated pages for scenarios, models, optimization, and run detail
- artifact-backed loading for enriched road layers

Recommended setup steps:

1. `python manage.py migrate`
2. `python manage.py ingest_research_data`
3. `python manage.py export_dashboard_artifacts`
4. `python manage.py runserver`

## Performance Notes

The slowest part of the app is enriching the full Pune road GeoJSON with graph-derived attributes.

To reduce request-time cost, `export_dashboard_artifacts` writes:

- `data/dashboard_artifacts/severity_overview.json`
- `data/dashboard_artifacts/roads_enriched_scale_0.7.json`
- `data/dashboard_artifacts/roads_enriched_scale_1.2.json`
- `data/dashboard_artifacts/roads_enriched_scale_1.5.json`
- `data/dashboard_artifacts/roads_enriched_scale_2.5.json`

When those files exist, Django uses them instead of recomputing the preset layers on every request.

## Remaining Research Caveats

- Flood-risk colors are data-based but still visually coarse because the rainfall-derived risk values are spatially compressed.
- Critical-road rankings are valid relative to the current graph and traffic assignment, but they still need methodological justification in the final report.
- Custom scales are simulated live in Django for exploration; final published figures should still be generated as saved experiment artifacts.
