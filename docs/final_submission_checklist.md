# Final Submission Checklist

Use this checklist before final demo, viva, report handoff, or repository submission.

## Research Validation

- Confirm the chosen presentation scale (`0.7`, `1.2`, `1.5`, or `2.5`) matches the screenshots and reported metrics.
- Verify that cascade summary values match the corresponding saved result JSON.
- Review whether the critical-road explanation is acceptable for the final report.
- State clearly that flood-risk colors are percentile-based and data-driven, but still spatially coarse.

## Dashboard Preparation

- Run `python manage.py ingest_research_data`
- Run `python manage.py export_dashboard_artifacts`
- Run `python manage.py check`
- Run `python manage.py test dashboard`
- Start the server and review:
  - `/`
  - `/scenarios/`
  - `/models/`
  - `/optimization/`

## Figures And Screenshots

- Take a homepage screenshot with the map visible.
- Capture one scenario-explorer screenshot with the scale comparison chart.
- Capture one model-results screenshot with the benchmark chart.
- Capture one optimization screenshot with the budget-sensitivity chart.
- Capture one run-detail page screenshot for evidence of experiment tracking.

## Report Content

- Problem statement
- Why flood resilience in infrastructure matters
- Dataset and GIS source summary
- Flood-risk generation method
- Graph and cascade methodology
- ML models used
- Optimization strategy used
- Key findings
- Limitations and future work

## Limitations To State Honestly

- Flood-risk resolution is coarse.
- Critical-road ranking depends on current graph assumptions.
- Some benchmark results are imported from project benchmarks rather than a full retraining report inside Django.
- Custom scales are interactive exploratory simulations and should not replace saved experiment artifacts in the final writeup.

## Optional Final Improvements

- Replace benchmark placeholders with retrained stored model metrics for every model.
- Export publication-ready plots directly from the pipeline.
- Refine critical-road methodology with additional centrality or spatial constraints.
- Add authentication/admin polish if the project will be demonstrated live.
