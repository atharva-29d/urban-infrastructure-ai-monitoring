import ast
import json
import pickle
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
MARKDOWN_PATH = DOCS_DIR / "project_complete_beginner_guide.md"
PDF_PATH = DOCS_DIR / "project_complete_beginner_guide.pdf"


GUIDE_INTRO = r"""
# Urban Infrastructure AI Monitoring and Resilience Analysis

Complete beginner guide for understanding the project from 0 to 100 percent.

Prepared for someone who is new to the codebase, new to the research pipeline, and possibly new to Python infrastructure projects.

## 1. What This Project Is

This project is a research-style prototype for studying how an urban road network can break down during flooding.

The project uses Pune road-network data, rainfall-derived flood-risk signals, graph simulation, machine learning, repair-strategy optimization, and a Django dashboard. In simple words, it asks:

- Which roads are exposed to rainfall or flood risk?
- If some roads fail first, can that failure spread to nearby roads?
- Can a machine-learning model predict which roads will fail?
- Which roads should be protected or repaired first when the budget is limited?
- Can all of this be shown in a dashboard for demo, viva, or report use?

The project is not only a website. The website is the final surface. Behind it is a full pipeline:

```text
Raw GIS data + rainfall files
        |
        v
Road and bridge preprocessing
        |
        v
Flood-risk calculation
        |
        v
NetworkX road graph
        |
        v
Flood failure and cascade simulation
        |
        v
ML dataset generation
        |
        v
Model training and evaluation
        |
        v
Repair optimization and critical-road ranking
        |
        v
Django research dashboard
```

## 2. The Problem Being Solved

Cities depend on roads, bridges, and junctions. During heavy rainfall or flooding, a few weak or exposed road segments may become unusable. That can overload nearby roads, force traffic to reroute, and create a wider failure pattern.

This project models that idea computationally. It treats roads as connected parts of a graph. Every road can have attributes such as traffic, length, rainfall, flood risk, capacity, stress, and failure state. Then the system simulates what happens when rainfall causes some roads to fail.

The project is useful as a final-year or research prototype because it combines:

- geospatial data processing
- rainfall and flood-risk scoring
- graph algorithms
- cascading failure simulation
- tabular machine learning
- graph neural-network experiments
- temporal failure prediction
- optimization under repair budgets
- dashboard presentation

## 3. Important Beginner Concepts

### GIS

GIS means Geographic Information System. In this project, roads and bridges are stored as geographic shapes. The main format is GeoJSON.

A road is usually a LineString or MultiLineString, which means it is drawn as one or more lines on a map.

### CRS

CRS means Coordinate Reference System. A common map CRS is EPSG:4326, which stores latitude and longitude. For measuring distance in meters, the notebooks project the data to a local metric CRS such as EPSG:32643.

### Graph

A graph is a structure made of nodes and edges.

In this project:

- a road segment is represented as a road node
- graph edges connect roads that touch or are considered neighbors
- graph attributes store road details such as traffic and flood risk

### Cascade

A cascade is a chain reaction. If a road fails, nearby roads may receive extra stress. If that stress becomes too high, they may fail too. This produces waves of failure across the network.

### ML Dataset

The machine-learning dataset is a table where each row represents a road in a simulated scenario. The model learns from road features and tries to predict whether the road failed.

### Django

Django is the web framework used for the dashboard. It handles URLs, database models, templates, pages, admin screens, and management commands.

## 4. Repository Layout

```text
EDI/
  data/
    gis/                  GeoJSON road, bridge, and boundary data
    graphs/               NetworkX graph files and simulation outputs
    weather/              raw rainfall NetCDF files
    dashboard_artifacts/  precomputed dashboard JSON files
  data_ingestion/         rainfall loading and preview utilities
  docs/                   project notes and this guide
  graph/                  cascade simulation logic
  infra_ai_system/        Django project and dashboard app
  ml/                     dataset generation and model training
  notebooks/              exploratory GIS and graph-building notebooks
  optimization/           repair and budget strategy scripts
  risk/                   rainfall-to-flood-risk logic
  static/                 static data copies for web use
  visualization/          plotting and map-generation scripts
  run_scenario.py         main standalone cascade simulation script
  requirements.txt        Python dependencies
```

## 5. Main Technologies

- pandas: table processing
- geopandas: GIS files and road geometry
- xarray: rainfall NetCDF files
- shapely: geometric operations
- networkx: graph representation and graph algorithms
- scikit-learn: tabular ML models and evaluation
- torch: neural networks
- torch-geometric: graph neural networks
- matplotlib and imageio: plots and animations
- Django: research dashboard
- SQLite: local dashboard database
- ReportLab: this PDF generator

## 6. Data Flow In Plain English

The project begins with raw geographic road data and rainfall data. The notebooks clean and clip the GIS data to Pune, assign road IDs, calculate length and traffic-like scores, and build a graph. The rainfall module samples rainfall near each road and converts it into a flood-risk score.

After that, scenario scripts simulate disasters. A storm severity scale controls how likely roads are to fail initially. Failed roads then stress their neighbors. The cascade function repeats until no new road failures appear or a maximum step count is reached.

Those simulation outputs are saved. They become:

- graph pickle files for later analysis
- JSON files for cascade summaries
- CSV files for ML datasets
- trained model pickle files
- metrics JSON files
- dashboard-ready JSON artifacts

Finally, the Django app reads the database and saved artifacts to show the scenario, maps, model results, optimization results, and route planning.

## 7. Building The Project From Scratch

This section explains how the project could be rebuilt from an empty folder.

### Step 1: Create The Project Structure

Create folders for data, graph logic, risk logic, ML logic, optimization, visualization, notebooks, docs, and the Django app.

```text
EDI/
  data/
  graph/
  risk/
  ml/
  optimization/
  visualization/
  notebooks/
  infra_ai_system/
```

### Step 2: Install Python Packages

The dependency list is in `requirements.txt`.

Typical setup:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Prepare GIS Data

Use `notebooks/01_prepare_gis.ipynb` to:

- load Pune boundary data
- load raw roads
- load raw bridges
- clip roads and bridges to Pune
- assign IDs such as `road_0`, `road_1`, and so on
- save `roads_final.geojson` and `bridges_final.geojson`

Use `notebooks/02_load_gis.ipynb` to:

- inspect GIS columns
- project roads into a meter-based CRS
- calculate road lengths
- assign traffic weights using the road type
- connect roads and bridges
- build and save the NetworkX graph

### Step 4: Attach Rainfall And Flood Risk

Use `risk/flood_risk.py`.

The script:

- finds a rainfall NetCDF file for a year
- clips rainfall to a Pune bounding box
- loads final road geometry
- picks a midpoint for each road
- samples rainfall at the nearest rainfall grid cell
- calculates mean rainfall
- calculates a flood-risk score between 0 and 1
- returns roads with `rain_mm_mean` and `flood_risk`

### Step 5: Build The Base Graph

The base graph is stored at:

```text
data/graphs/pune_base_graph_weather.gpickle
```

It is a Python pickle containing a NetworkX graph. Road nodes contain attributes such as:

- `type`
- `traffic`
- `length`
- `rain_mm_mean`
- `flood_risk`
- `failed`
- `capacity`
- `base_traffic`
- `stress`

### Step 6: Run A Flood Scenario

Use `run_scenario.py`.

The script loads the base graph, sets scenario parameters, seeds initial failures from flood risk, runs the cascade simulation, and saves results.

Important severity examples:

- `0.7`: mild scenario
- `1.5`: severe scenario
- `2.5`: extreme scenario

Output examples:

- `data/graphs/run_results_weather_0.7.json`
- `data/graphs/run_results_weather_1.5.json`
- `data/graphs/run_results_weather_2.5.json`
- `data/graphs/pune_after_cascade_weather_0.7.gpickle`

### Step 7: Generate Machine-Learning Datasets

Use:

```powershell
python -m ml.dataset
python -m ml.temporal_dataset
```

The tabular dataset saves to:

```text
data/ml_dataset.csv
```

The temporal dataset saves to:

```text
data/temporal_dataset.csv
```

### Step 8: Train Models

The strongest current training script is:

```powershell
python -m ml.train_model
```

It compares:

- Random Forest
- Extra Trees
- Logistic Regression

It saves:

- best model pickle
- model metrics JSON
- model comparison JSON
- feature importance CSV

Experimental neural-network scripts also exist:

- `ml/train_nn.py`
- `ml/train_temporal.py`
- `ml/train_gnn.py`
- `ml/train_gnn_generalization.py`
- `ml/train_stgnn.py`

### Step 9: Run Optimization

Use:

```powershell
python -m optimization.compare_strategies
```

This compares greedy repair against random repair for different repair budgets.

Greedy repair uses `optimization/greedy_repair.py`, which ranks roads by a combined normalized score of degree centrality and flood risk.

### Step 10: Prepare The Dashboard

From `infra_ai_system/`:

```powershell
python manage.py migrate
python manage.py ingest_research_data
python manage.py export_dashboard_artifacts
python manage.py runserver
```

Then open:

```text
http://127.0.0.1:8000/
```

## 8. Core Algorithm: Flood Failure Seeding

The first failure stage is probabilistic. For every road, the code looks at `flood_risk`. Higher risk means a higher chance that the road fails at the start of the scenario.

Plain-English logic:

```text
for every road:
    risk = road flood risk between 0 and 1
    probability = risk multiplied by scenario scale
    draw a random number between 0 and 1
    if random number is smaller than probability:
        mark road as failed
```

This is why scale matters. A scale of 0.7 creates fewer initial failures. A scale of 2.5 creates many more.

## 9. Core Algorithm: Cascade Simulation

The main cascade function lives in `graph/cascade.py`.

The idea is:

```text
find roads that already failed
for each road that is still working:
    count how many failed road neighbors it has
    calculate extra overload caused by those failed neighbors
    add overload/capacity to the road stress value
    if stress is greater than 1:
        convert stress into a failure probability
        maybe mark the road as failed
return the list of newly failed roads
```

Key variables:

- `base_traffic`: original traffic value for the road
- `capacity`: estimated road tolerance before failure
- `overload_factor`: how strongly neighbor failures increase stress
- `stress`: accumulated pressure over cascade steps
- `failed_neighbors`: nearby failed road nodes

This is a simple research heuristic, not a real hydraulic or traffic-engineering simulator. Its value is that it gives controlled failure labels and lets the project study network behavior.

## 10. Scenario Tuning

`run_scenario.py` adjusts capacity and overload settings based on severity.

Mild scenarios:

- higher capacity factor
- lower overload factor
- fewer cascades

Severe or extreme scenarios:

- lower capacity factor
- higher overload factor
- larger cascades

The current tuning is:

```text
scale <= 0.8:
    capacity factor = 1.5
    overload factor = 1.2

scale <= 1.6:
    capacity factor = 1.25
    overload factor = 1.5

scale > 1.6:
    capacity factor = 1.1
    overload factor = 1.8
```

## 11. Machine Learning Explained

The ML part tries to predict road failure from features.

Each row in `data/ml_dataset.csv` is one road under one simulated scenario. The target column is `failed`.

Important feature groups:

- physical or assigned road features: traffic, length
- rainfall features: rain, flood risk
- graph features: degree, clustering, core number, PageRank
- neighbor features: neighbor risk, neighbor traffic, neighbor degree
- failure context: failed neighbors and failed-neighbor ratio
- scenario context: scale and exposure score

The best tabular training script filters out roads that were initially failed, so the model focuses on predicting cascade-driven failures among roads that were alive at the start.

The train/validation/test split is grouped by `simulation_id`. This is important because rows from the same simulation are related. Grouped splitting reduces leakage between train and test sets.

## 12. Dashboard Explained

The dashboard is a Django app inside:

```text
infra_ai_system/dashboard/
```

The key pages are:

- `/`: dashboard overview with map, scenario selector, severity selector, cascade summary, training snapshot, and optimization snapshot
- `/scenarios/`: scenario explorer with severity comparison chart and cascade interpretation
- `/models/`: model results page with model metrics, candidate comparison, and feature importance
- `/optimization/`: repair optimization page with greedy-vs-random budget comparison
- `/routing/`: safe rerouting planner
- `/routing/search/`: JSON endpoint for road search
- `/runs/<slug>/`: experiment run detail page

The dashboard uses saved artifacts for speed. Instead of enriching every road on every request, `export_dashboard_artifacts` writes precomputed JSON files into `data/dashboard_artifacts/`.

## 13. Django Database Model

Django stores structured experiment information using these models:

- `ScenarioConfig`: a scenario definition such as Pune extreme flood 2020
- `ExperimentRun`: one run of a cascade, dataset build, training job, optimization job, or dashboard snapshot
- `RunArtifact`: a file produced by a run, such as a graph, CSV, JSON, model, plot, or report
- `ModelResult`: model metrics and metadata
- `OptimizationResult`: repair-strategy results for different budgets

This design is useful because it turns loose files into explainable experiment records.

## 14. Safe Rerouting Feature

The routing page lets a user choose a source road and target road. It then computes a route while avoiding roads that failed under the selected severity.

The route cost changes by mode:

- `safe`: strongly penalizes flood risk and criticality
- `balanced`: trades safety against route burden
- `fast`: leans toward shorter and lighter-traffic paths
- `emergency`: strongly avoids risky and critical segments

Internally it uses NetworkX shortest path with custom edge weights.

## 15. Outputs And Artifacts

Important output types:

- `.geojson`: geographic road and bridge data
- `.gpickle`: saved NetworkX graphs
- `.json`: simulation summaries, dashboard artifacts, model metrics
- `.csv`: ML datasets, critical-road rankings, feature importance
- `.pkl`: trained model files
- `.png` or `.gif`: visualization outputs

You should think of the repo as having two layers:

- source code that knows how to generate results
- generated artifacts that store the results for analysis and dashboard use

## 16. How To Explain This Project In A Viva

A strong short explanation:

This project builds a flood-resilience analysis system for Pune road infrastructure. It converts GIS road data and rainfall data into a graph where each road has traffic, length, rainfall, and flood-risk attributes. A flood scenario first fails roads probabilistically based on flood risk, then a cascade model spreads failures through neighboring roads based on overload and capacity. The resulting simulations are used to train ML models that predict road failure and to compare repair strategies under limited budgets. A Django dashboard presents the scenario maps, cascade metrics, model results, optimization results, and safe rerouting.

## 17. Limitations To State Honestly

- Flood risk is rainfall-derived and spatially coarse.
- Failure labels are simulation-generated, not real observed road failure labels.
- Very high ML scores may reflect the structure of the simulation labels.
- Critical-road rankings depend on the chosen centrality, traffic, and flood-risk assumptions.
- Custom dashboard scales are useful for exploration, but final report figures should use saved artifacts.
- Some graph and dashboard artifact names are not perfectly consistent; check file names before final submission.

## 18. Recommended Demo Order

1. Open the dashboard home page.
2. Show the severity selector and map layers.
3. Open Scenario Explorer and explain severity comparison.
4. Open Model Results and explain the trained model and feature importance.
5. Open Optimization Analysis and compare greedy repair with random repair.
6. Open Safe Rerouting and demonstrate how failed roads are avoided.
7. Open a run detail page to show experiment tracking.

## 19. How A Beginner Should Read The Code

Read the project in this order:

1. `README.md`
2. `risk/flood_risk.py`
3. `graph/cascade.py`
4. `run_scenario.py`
5. `ml/dataset.py`
6. `ml/train_model.py`
7. `optimization/greedy_repair.py`
8. `optimization/compare_strategies.py`
9. `infra_ai_system/dashboard/models.py`
10. `infra_ai_system/dashboard/views.py`
11. dashboard templates in `infra_ai_system/dashboard/templates/dashboard/`

That order follows the real story of the project: data, risk, graph, simulation, learning, optimization, interface.
"""


FILE_DESCRIPTIONS = {
    "data_ingestion/weather_loader.py": "Loads rainfall NetCDF files, previews rainfall variables, clips rainfall data to the Pune bounding box, and plots sample rainfall snapshots.",
    "graph/cascade.py": "Contains the shared cascade step that spreads failures from already failed road nodes to nearby road nodes using stress, capacity, and overload.",
    "risk/flood_risk.py": "Loads rainfall and road GIS data, samples rainfall near each road, calculates flood-risk scores, and attaches rainfall fields to roads.",
    "run_scenario.py": "Standalone script for running one flood scenario, seeding failures from flood risk, executing cascade steps, and saving result JSON and final graph pickle.",
    "ml/dataset.py": "Generates the main tabular ML dataset by running many simulated scenarios and recording road, graph, neighbor, exposure, and failure features.",
    "ml/temporal_dataset.py": "Generates a temporal dataset where each row describes a road at one cascade step and the target is whether it fails in the next step.",
    "ml/train_model.py": "Trains and compares tabular classifiers, selects the best model by validation PR-AUC, evaluates it on a held-out test split, and saves metrics and feature importance.",
    "ml/train_nn.py": "Trains a simple feed-forward neural network on the tabular road-failure dataset.",
    "ml/train_temporal.py": "Trains a temporal neural network to predict future failure for roads that are currently alive.",
    "ml/gnn_dataset.py": "Converts one saved graph into a PyTorch Geometric Data object for graph neural-network experiments.",
    "ml/gnn_dataset_multi.py": "Builds normalized graph datasets across multiple severity scales for generalization experiments.",
    "ml/stgnn_dataset.py": "Builds graph pairs/sequences for spatio-temporal graph neural-network experiments.",
    "ml/train_gnn.py": "Trains a GraphSAGE model on one graph dataset and evaluates ROC-AUC.",
    "ml/train_gnn_generalization.py": "Trains GraphSAGE on multiple scenarios and tests it on a held-out scenario.",
    "ml/train_stgnn.py": "Trains an experimental spatio-temporal GNN with GraphSAGE layers and a GRU.",
    "ml/generate_gnn_graphs.py": "Creates saved post-cascade graphs at different scales for graph neural-network experiments.",
    "ml/generate_temporal_graphs.py": "Saves step-by-step graph snapshots for temporal cascade analysis.",
    "ml/export_graph_to_json.py": "Exports a saved graph into JSON for web/static visualization use.",
    "optimization/greedy_repair.py": "Loads critical-road scores and selects the top K roads by a normalized centrality-plus-flood-risk score.",
    "optimization/compare_strategies.py": "Compares greedy repair against random repair by reinforcing selected roads and measuring final failed-road counts.",
    "optimization/optimize_budget.py": "Placeholder for future budget optimization logic.",
    "optimization/random_baseline.py": "Placeholder for future random-baseline logic.",
    "visualization/animate_cascade.py": "Creates an animated visualization of cascade progression.",
    "visualization/cascade_progression.py": "Plots cascade progression from saved JSON results.",
    "visualization/compare_scenarios.py": "Compares failure outcomes across saved scenario JSON files.",
    "visualization/critical_roads.py": "Computes or visualizes critical roads based on graph centrality and related attributes.",
    "visualization/heatmap_failures.py": "Creates spatial heatmaps for failed roads.",
    "visualization/map_failures.py": "Maps failed roads after a cascade.",
    "visualization/plot_cascade.py": "Plots cascade step counts from result JSON.",
    "visualization/plot_opt_vs_random.py": "Plots greedy-vs-random repair comparison.",
    "visualization/plot_repair_curve.py": "Plots repair-budget impact curves.",
    "visualization/repair_impact.py": "Measures how different repair budgets reduce failure under a scenario.",
    "visualization/resilience_curve.py": "Plots resilience or failure fraction curves from saved outputs.",
    "visualization/resilience_metrics.py": "Computes simple graph resilience metrics.",
    "visualization/visualize_cascade.py": "Visualizes cascade results on the graph.",
    "infra_ai_system/manage.py": "Django command-line entry point.",
    "infra_ai_system/infra_ai_system/settings.py": "Django settings for installed apps, database, templates, static files, and development configuration.",
    "infra_ai_system/infra_ai_system/urls.py": "Root URL router that sends dashboard URLs to the dashboard app and exposes the Django admin.",
    "infra_ai_system/dashboard/models.py": "Defines the database schema for scenarios, experiment runs, artifacts, model results, and optimization results.",
    "infra_ai_system/dashboard/views.py": "Loads artifacts, enriches road GeoJSON, simulates custom severities, prepares page contexts, and handles dashboard/routing pages.",
    "infra_ai_system/dashboard/urls.py": "Maps dashboard URLs to view functions.",
    "infra_ai_system/dashboard/admin.py": "Configures Django admin screens for research models.",
    "infra_ai_system/dashboard/tests.py": "Tests model behavior, page rendering, custom scale handling, artifact loading, and route search.",
    "infra_ai_system/dashboard/management/commands/ingest_research_data.py": "Imports saved research outputs into structured Django database rows.",
    "infra_ai_system/dashboard/management/commands/export_dashboard_artifacts.py": "Precomputes enriched map artifacts for faster dashboard page loading.",
}


TEMPLATE_DESCRIPTIONS = {
    "base.html": "Shared page shell, navigation, hero/header blocks, and severity/custom-scale form.",
    "dashboard.html": "Main dashboard page with scenario selector, severity selector, Leaflet infrastructure map, cascade summary, training snapshot, and optimization snapshot.",
    "scenario_explorer.html": "Severity comparison page with Chart.js visualizations, selected severity metrics, interpretation notes, and critical-road spatial summary.",
    "model_results.html": "Model page with summary cards, model performance chart, candidate comparison, feature importance, benchmark reference, and training runs.",
    "optimization_analysis.html": "Optimization page with budget sensitivity chart and greedy-vs-random result tables.",
    "route_planner.html": "Interactive safe-rerouting page with road search, map clicks, route comparison, and mode-specific route weighting.",
    "run_detail.html": "Experiment-run detail page showing metrics, inputs, artifacts, model results, and optimization rows.",
}


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def graph_stats(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as file:
            graph = pickle.load(file)
    except Exception:
        return None
    roads = [node for node, data in graph.nodes(data=True) if data.get("type") == "road"]
    failed = sum(1 for node in roads if graph.nodes[node].get("failed"))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "roads": len(roads),
        "failed": failed,
    }


def dynamic_project_summary():
    parts = ["\n## 20. Current Project Facts From Saved Files\n"]

    base_stats = graph_stats(ROOT / "data/graphs/pune_base_graph_weather.gpickle")
    if base_stats:
        parts.append(
            f"The current base graph contains {base_stats['nodes']} nodes, "
            f"{base_stats['edges']} edges, and {base_stats['roads']} road nodes."
        )

    scenario_paths = [
        ("0.7 mild", "data/graphs/pune_after_cascade_weather_0.7.gpickle", "data/graphs/run_results_weather_0.7.json"),
        ("1.2 severe", "data/graphs/pune_after_cascade_weather_1.2.gpickle", None),
        ("1.5 very severe", "data/graphs/pune_after_cascade_weather_1.5.gpickle", "data/graphs/run_results_weather_1.5.json"),
        ("2.5 extreme", "data/graphs/pune_after_cascade_weather.gpickle", "data/graphs/run_results_weather_2.5.json"),
    ]
    parts.append("\n### Saved Scenario Results\n")
    for label, graph_rel, json_rel in scenario_paths:
        stats = graph_stats(ROOT / graph_rel)
        results = read_json(ROOT / json_rel) if json_rel else None
        if not stats:
            continue
        text = f"- {label}: {stats['failed']} failed roads in `{graph_rel}`"
        if results:
            cascade = results.get("capacity_cascade", [])
            text += (
                f"; initial failures = {results.get('initial_failures', 0)}"
                f"; cascade steps = {len(cascade)}"
                f"; peak step failures = {max(cascade) if cascade else 0}"
            )
        parts.append(text)

    metrics = read_json(ROOT / "data/best_failure_model_metrics.json")
    if metrics:
        parts.append("\n### Current Best Model Metrics\n")
        parts.append(f"- selected model: {metrics.get('selected_model')}")
        parts.append(f"- ROC-AUC: {metrics.get('roc_auc')}")
        parts.append(f"- PR-AUC: {metrics.get('pr_auc')}")
        parts.append(f"- F1: {metrics.get('f1')}")
        parts.append(f"- balanced accuracy: {metrics.get('balanced_accuracy')}")
        parts.append(f"- train rows: {metrics.get('train_rows')}")
        parts.append(f"- validation rows: {metrics.get('validation_rows')}")
        parts.append(f"- test rows: {metrics.get('test_rows')}")

    comparison = read_json(ROOT / "data/model_comparison.json")
    if isinstance(comparison, list):
        parts.append("\n### Current Candidate Model Comparison\n")
        for row in comparison:
            parts.append(
                f"- {row.get('model_name')}: validation ROC-AUC {row.get('validation_roc_auc')}, "
                f"validation PR-AUC {row.get('validation_pr_auc')}, validation F1 {row.get('validation_f1')}"
            )

    optimization = read_json(ROOT / "data/graphs/opt_vs_random_scale_1.2.json")
    if isinstance(optimization, list):
        parts.append("\n### Current Greedy vs Random Repair Comparison\n")
        for row in optimization:
            parts.append(
                f"- K={row.get('K')}: greedy failed = {row.get('greedy_failed')}, "
                f"random failed = {row.get('random_failed')}"
            )

    repair = read_json(ROOT / "data/graphs/repair_impact_scale_1.2.json")
    if isinstance(repair, list):
        parts.append("\n### Current Repair Impact Curve\n")
        for row in repair:
            parts.append(
                f"- K={row.get('K')}: failed = {row.get('failed')} of {row.get('total')}, "
                f"fraction failed = {row.get('frac_failed')}"
            )

    return "\n".join(parts)


def ast_summary(path):
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except Exception as exc:
        return [f"- Could not parse: {exc}"]

    lines = []
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)

    if imports:
        short = ", ".join(imports[:10])
        if len(imports) > 10:
            short += ", ..."
        lines.append(f"- Main imports: {short}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(arg.arg for arg in node.args.args)
            lines.append(f"- Function `{node.name}({args})`, line {node.lineno}")
        elif isinstance(node, ast.ClassDef):
            lines.append(f"- Class `{node.name}`, line {node.lineno}")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = ", ".join(arg.arg for arg in child.args.args)
                    lines.append(f"  - Method `{child.name}({args})`, line {child.lineno}")

    return lines or ["- No top-level functions or classes."]


def code_map():
    parts = ["\n## 21. File-By-File Code Map\n"]
    skip_parts = {".git", ".venv", ".idea", "__pycache__", "staticfiles"}

    py_files = []
    for path in ROOT.rglob("*.py"):
        if any(part in skip_parts for part in path.parts):
            continue
        if "static" in path.parts and "data" in path.parts:
            continue
        if path == Path(__file__).resolve():
            continue
        py_files.append(path)

    for path in sorted(py_files, key=lambda item: item.as_posix()):
        path_rel = rel(path)
        parts.append(f"\n### {path_rel}\n")
        parts.append(FILE_DESCRIPTIONS.get(path_rel, "Project Python source file."))
        parts.extend(ast_summary(path))

    parts.append("\n## 22. Dashboard Template Map\n")
    template_dir = ROOT / "infra_ai_system/dashboard/templates/dashboard"
    for path in sorted(template_dir.glob("*.html")):
        path_rel = rel(path)
        parts.append(f"\n### {path_rel}\n")
        parts.append(TEMPLATE_DESCRIPTIONS.get(path.name, "Django HTML template."))

    parts.append("\n## 23. Notebook Map\n")
    parts.append("- `notebooks/01_prepare_gis.ipynb`: prepares/clips road and bridge GIS layers and saves final GeoJSON files.")
    parts.append("- `notebooks/02_load_gis.ipynb`: explores GIS data, assigns traffic/length, joins bridge information, and builds the base graph.")
    parts.append("- `notebooks/03_visualize_failures.ipynb`: explores and plots failure results after cascade simulations.")

    return "\n".join(parts)


def final_study_notes():
    return r"""

## 24. Common Questions And Answers

### Why use a graph?

Roads are connected. A graph gives the project a natural way to represent connectivity and neighbor effects. Without a graph, the project could still show flood risk, but it could not model cascading failures through the network.

### Why use random failure?

Flood impact is uncertain. The project uses risk multiplied by severity as a probability. That makes repeated simulations possible and creates datasets for ML.

### Why does the ML model score so high?

The labels come from a controlled simulation. The features also include strong graph and neighbor signals. That can make the classification problem easier than a real-world prediction problem. In a final report, present the score as simulation-label performance, not proof of real-world deployment readiness.

### What is the most important code file?

There is no single file. The core chain is:

```text
risk/flood_risk.py
graph/cascade.py
run_scenario.py
ml/dataset.py
ml/train_model.py
optimization/compare_strategies.py
infra_ai_system/dashboard/views.py
```

### What should be improved next?

- better flood-risk resolution
- real traffic data instead of traffic heuristics
- observed failure labels if available
- stronger spatial train/test splitting
- cleaner experiment artifact naming
- more robust route-planning validation
- deployment hardening for Django settings and secrets

## 25. Final Mental Model

Remember the project as five layers:

1. Data layer: roads, bridges, rainfall, graph files, datasets.
2. Simulation layer: flood risk seeds failures, cascade spreads failures.
3. Learning layer: models learn patterns from simulated failures.
4. Decision layer: optimization ranks or repairs roads under budgets.
5. Interface layer: Django turns the research outputs into pages and maps.

If you understand those five layers, the project becomes much easier to explain.

## 26. One-Page Beginner Summary

The project studies flood resilience in Pune's road network. It begins with GIS road and bridge data and rainfall files. The code assigns each road a flood-risk score, builds a graph where roads are connected, and simulates how floods can cause initial road failures. A cascade model then spreads failures to nearby roads when stress exceeds capacity. The generated simulations become datasets for machine learning, where models predict road failure from traffic, rainfall, graph, and neighbor features. Optimization scripts compare which roads should be reinforced first under limited budgets. A Django dashboard presents the results with maps, severity controls, model charts, optimization charts, experiment records, and safe rerouting.

That is the whole project in one sentence: it is an end-to-end urban infrastructure resilience prototype that connects geospatial data, rainfall risk, graph cascade simulation, machine learning, optimization, and dashboard visualization.
"""


def build_markdown():
    text = "\n\n".join(
        [
            GUIDE_INTRO.strip(),
            dynamic_project_summary().strip(),
            code_map().strip(),
            final_study_notes().strip(),
        ]
    )
    text = text.replace("\r\n", "\n")
    MARKDOWN_PATH.write_text(text, encoding="utf-8")
    return text


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="GuideTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor("#1f2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=22,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#0f766e"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#374151"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.5,
            leftIndent=16,
            firstLineIndent=0,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=7.2,
            leading=9,
            leftIndent=6,
            rightIndent=6,
            backColor=colors.HexColor("#f3f4f6"),
            borderColor=colors.HexColor("#e5e7eb"),
            borderWidth=0.5,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles


def flush_paragraph(buffer, story, style):
    if not buffer:
        return
    text = " ".join(item.strip() for item in buffer if item.strip())
    if text:
        story.append(Paragraph(escape(text), style))
    buffer.clear()


def add_code_block(text, story, style):
    lines = text.splitlines()
    chunk = []
    for line in lines:
        chunk.append(line)
        if len(chunk) >= 44:
            story.append(Preformatted(escape("\n".join(chunk)), style))
            chunk = []
    if chunk:
        story.append(Preformatted(escape("\n".join(chunk)), style))


def markdown_to_story(markdown_text):
    styles = make_styles()
    story = []
    paragraph = []
    in_code = False
    code_lines = []
    first_heading = True

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            if in_code:
                add_code_block("\n".join(code_lines), story, styles["GuideCode"])
                code_lines = []
                in_code = False
            else:
                flush_paragraph(paragraph, story, styles["GuideBody"])
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_paragraph(paragraph, story, styles["GuideBody"])
            story.append(Spacer(1, 3))
            continue

        if line.startswith("# "):
            flush_paragraph(paragraph, story, styles["GuideBody"])
            if not first_heading:
                story.append(PageBreak())
            story.append(Paragraph(escape(line[2:].strip()), styles["GuideTitle"]))
            first_heading = False
            continue

        if line.startswith("## "):
            flush_paragraph(paragraph, story, styles["GuideBody"])
            story.append(Paragraph(escape(line[3:].strip()), styles["GuideH1"]))
            continue

        if line.startswith("### "):
            flush_paragraph(paragraph, story, styles["GuideBody"])
            story.append(Paragraph(escape(line[4:].strip()), styles["GuideH2"]))
            continue

        if line.startswith("- "):
            flush_paragraph(paragraph, story, styles["GuideBody"])
            story.append(Paragraph(escape(line[2:].strip()), styles["GuideBullet"], bulletText="-"))
            continue

        numbered = re.match(r"^(\d+)\.\s+(.*)", line)
        if numbered:
            flush_paragraph(paragraph, story, styles["GuideBody"])
            story.append(
                Paragraph(
                    escape(numbered.group(2).strip()),
                    styles["GuideBullet"],
                    bulletText=f"{numbered.group(1)}.",
                )
            )
            continue

        paragraph.append(line)

    flush_paragraph(paragraph, story, styles["GuideBody"])
    if code_lines:
        add_code_block("\n".join(code_lines), story, styles["GuideCode"])
    return story


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(0.65 * inch, 0.45 * inch, "Urban Infrastructure AI Monitoring and Resilience Analysis")
    canvas.drawRightString(7.6 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def render_pdf(markdown_text):
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.75 * inch,
        title="Urban Infrastructure AI Monitoring and Resilience Analysis - Complete Beginner Guide",
        author="Atharva / Codex",
    )
    story = markdown_to_story(markdown_text)
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def main():
    markdown_text = build_markdown()
    render_pdf(markdown_text)
    print(f"Wrote Markdown: {MARKDOWN_PATH}")
    print(f"Wrote PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
