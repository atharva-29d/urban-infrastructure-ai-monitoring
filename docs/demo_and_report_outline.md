# Demo And Report Outline

Use this as a short final presentation structure for viva, demo day, or project report walkthrough.

## 1. Opening Problem Statement

Urban road networks are vulnerable to flood-driven disruptions.

The goal of this project is to:

- estimate flood stress on roads
- simulate cascading failures in the road graph
- predict road failures using machine learning
- identify critical roads and compare repair strategies
- present the whole analysis in an interactive Django dashboard

## 2. Data And Pipeline

Explain the project in this order:

1. GIS road-network data is loaded for Pune.
2. Rainfall-derived flood-risk values are attached to road segments.
3. A graph representation of the infrastructure is used for cascade simulation.
4. Repeated simulations generate ML datasets.
5. ML models are trained to predict failures.
6. Optimization logic compares repair strategies.
7. The Django app visualizes the results by scenario and severity scale.

## 3. Dashboard Demo Flow

Suggested live click path:

1. Homepage
   Show the infrastructure map, failed roads, critical roads, and severity selector.

2. Scenario Explorer
   Explain how different flood scales change failure patterns and cascade intensity.

3. Model Results
   Show the tuned model metrics, candidate-model comparison, and feature importance.

4. Optimization Analysis
   Show the greedy-vs-random repair comparison.

5. Run Detail
   Show that results are stored as structured experiment outputs.

## 4. Key Results To Highlight

- The tuned Random Forest strongly outperformed the original benchmark baseline.
- The dashboard supports preset scales plus custom exploratory scale simulation.
- Critical roads are not random; they cluster where graph centrality and traffic concentration are highest.
- Artifact-backed loading improves dashboard performance for preset severities.

## 5. Important Limitations To Say Clearly

- Flood-risk colors are data-driven but still spatially coarse.
- The failure labels are simulation-generated, so predictive performance must be interpreted in that context.
- Extremely high model scores can reflect an easier simulation label structure rather than true real-world generalization.
- Critical-road ranking depends on the current graph and traffic assumptions.

## 6. Future Work

- improve flood-risk resolution
- use more realistic traffic and infrastructure metadata
- add stronger holdout evaluation by scenario/year
- refine critical-road methodology
- compare more ML and spatio-temporal graph models under a stricter evaluation setup

## 7. Closing Line

This project demonstrates a full research prototype for urban infrastructure resilience analysis, combining geospatial data, graph simulation, ML prediction, optimization, and an interactive Django interface in one system.
