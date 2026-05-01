# Urban Infrastructure AI Monitoring and Resilience Analysis

Research-oriented system for **flood-driven failure analysis of urban road infrastructure**, built around Pune road-network data, rainfall-derived flood-risk signals, cascading failure simulation, machine-learning prediction, and repair-strategy evaluation.

## Project Summary

This project studies how urban infrastructure networks behave under flood stress.

The workflow combines:

- GIS road-network data
- rainfall-informed flood-risk scoring
- graph-based cascade simulation
- ML-based failure prediction
- repair and resilience optimization
- a Django research dashboard for interactive analysis

The current implementation is designed as a **final-year / research-style prototype** with a working experimental pipeline and a multi-page Django interface for exploration.

## Research Pipeline

The end-to-end pipeline is:

1. Load road GIS data for Pune.
2. Attach rainfall-derived flood-risk values to road segments.
3. Build a graph representation of the infrastructure network.
4. Simulate flood-induced initial failures and cascading breakdown.
5. Generate ML datasets from repeated simulated scenarios.
6. Train predictive models for road failure.
7. Compare repair strategies under constrained budgets.
8. Surface the results through the Django dashboard.

## Repository Layout

```text
EDI/
  data/
    gis/
    graphs/
    weather/
    dashboard_artifacts/
  data_ingestion/
  docs/
  graph/
  infra_ai_system/
    dashboard/
    infra_ai_system/
  ml/
  notebooks/
  optimization/
  risk/
  visualization/
  run_scenario.py
```

## Main Components

- `risk/`
  Flood-risk estimation from rainfall datasets.

- `graph/`
  Cascading failure logic for infrastructure disruption.

- `ml/`
  Dataset generation and model-training scripts.

- `optimization/`
  Repair-budget comparison and resilience-oriented decision logic.

- `infra_ai_system/dashboard/`
  Django application for scenario exploration, model inspection, optimization analysis, and run detail views.

## Django Research App

The Django app now includes:

- dashboard overview page
- scenario explorer
- model results page
- optimization analysis page
- run detail page
- preset scale switching for `0.7`, `1.2`, `1.5`, and `2.5`
- custom user-entered scale simulation
- map layers for:
  - flood-risk percentile
  - failed roads
  - critical roads

## Running The Project

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Prepare Django and ingest results

Run these commands from:

```text
C:\PycharmProjects\EDI\infra_ai_system
```

```powershell
python manage.py migrate
python manage.py ingest_research_data
python manage.py export_dashboard_artifacts
```

### 3. Start the dashboard

```powershell
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Core Data Artifacts

Important generated and precomputed files include:

- `data/graphs/pune_base_graph_weather.gpickle`
- `data/graphs/pune_after_cascade_weather_0.7.gpickle`
- `data/graphs/pune_after_cascade_weather_1.2.gpickle`
- `data/graphs/pune_after_cascade_weather_1.5.gpickle`
- `data/graphs/run_results_weather_0.7.json`
- `data/graphs/run_results_weather_1.5.json`
- `data/graphs/run_results_weather_2.5.json`
- `data/graphs/critical_roads.csv`
- `data/ml_dataset.csv`
- `data/temporal_dataset.csv`
- `data/dashboard_artifacts/*.json`

## Current Strengths

- The project has a real graph-based simulation pipeline.
- The dashboard is now structured as a multi-page research interface.
- Preset dashboard map layers are precomputed for better runtime performance.
- Model and optimization results are stored through the Django data model.

## Current Caveats

- Flood-risk visualization is data-driven, but still spatially coarse because the source risk values are coarse.
- Critical-road concentration is currently driven by the present traffic and centrality assumptions in the graph.
- Custom scales are useful for exploration, but preset exported scales are the cleaner option for final presentation and reporting.

## Recommended Final Submission Workflow

1. Run `ingest_research_data`.
2. Run `export_dashboard_artifacts`.
3. Review the dashboard pages for the target severity.
4. Capture final screenshots/figures.
5. Use the docs in `docs/` to prepare the methodology and conclusion sections.

## Supporting Docs

- `docs/research_foundation.md`
- `docs/final_submission_checklist.md`

## Author

Atharva  
AIML Undergraduate
