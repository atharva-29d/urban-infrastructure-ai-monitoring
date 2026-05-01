import pickle
import pandas as pd
import random
import sys
from functools import lru_cache
from pathlib import Path

import networkx as nx

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from graph.cascade import cascade_step_capacity

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
OUT_PATH = BASE_DIR / "data" / "ml_dataset.csv"

SCALE_RUNS = {
    0.2: 4,
    0.3: 4,
    0.4: 4,
    0.5: 4,
    0.6: 4,
    0.7: 4,
    0.8: 3,
    0.9: 3,
    1.0: 2,
    1.1: 1,
}


@lru_cache(maxsize=1)
def static_graph_features():
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    roads = [n for n, d in graph.nodes(data=True) if d.get("type") == "road"]
    road_graph = graph.subgraph(roads).copy()

    degrees = dict(road_graph.degree())
    clustering = nx.clustering(road_graph)
    core_numbers = nx.core_number(road_graph)
    avg_neighbor_degree = nx.average_neighbor_degree(road_graph)
    pagerank = nx.pagerank(road_graph, alpha=0.85, max_iter=100)
    articulation_points = set(nx.articulation_points(road_graph))

    component_size = {}
    for component in nx.connected_components(road_graph):
        size = len(component)
        for node in component:
            component_size[node] = size

    return {
        "degrees": degrees,
        "clustering": clustering,
        "core_numbers": core_numbers,
        "avg_neighbor_degree": avg_neighbor_degree,
        "pagerank": pagerank,
        "articulation_points": articulation_points,
        "component_size": component_size,
        "road_count": len(roads),
    }


def simulate_once(scale, run_index=0):
    rng = random.Random(42 + run_index + int(scale * 1000))

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    topology = static_graph_features()

    # Reset graph
    for n, d in G.nodes(data=True):
        if d.get("type") == "road":
            d["failed"] = False
            d["base_traffic"] = d["traffic"]
            d["capacity"] = d["traffic"] * 2.0

    # Flood seeding
    for n, d in G.nodes(data=True):
        if d.get("type") == "road":
            risk = d.get("flood_risk", 0)
            if rng.random() < risk * scale * 0.6:
                d["failed"] = True

    # Store initial failures
    initial_failed = set(
        n for n, d in G.nodes(data=True)
        if d.get("type") == "road" and d.get("failed")
    )

    # Cascade
    for _ in range(20):
        new = cascade_step_capacity(G, overload_factor=1.2)
        if not new:
            break

    # Neighbour statistics
    neighbor_risk = {}
    neighbor_traffic = {}
    neighbor_degree = {}
    neighbor_clustering = {}

    for n in G.nodes():
        nbrs = list(G.neighbors(n))

        if len(nbrs) == 0:
            neighbor_risk[n] = 0
            neighbor_traffic[n] = 0
            neighbor_degree[n] = 0
            neighbor_clustering[n] = 0
            continue

        risks = []
        traff = []
        degrees = []
        cluster_vals = []

        for nbr in nbrs:
            d = G.nodes[nbr]
            if d.get("type") == "road":
                risks.append(d.get("flood_risk", 0))
                traff.append(d.get("traffic", 0))
                degrees.append(topology["degrees"].get(nbr, 0))
                cluster_vals.append(topology["clustering"].get(nbr, 0))

        neighbor_risk[n] = sum(risks) / len(risks) if risks else 0
        neighbor_traffic[n] = sum(traff) / len(traff) if traff else 0
        neighbor_degree[n] = sum(degrees) / len(degrees) if degrees else 0
        neighbor_clustering[n] = sum(cluster_vals) / len(cluster_vals) if cluster_vals else 0

    # Initial failed neighbours
    failed_nbrs = {}

    for n in G.nodes():
        count = 0
        for nbr in G.neighbors(n):
            if nbr in initial_failed:
                count += 1
        failed_nbrs[n] = count

    rows = []
    simulation_id = f"scale_{scale}_run_{run_index}"

    for n, d in G.nodes(data=True):
        if d.get("type") != "road":
            continue

        degree = topology["degrees"].get(n, 0)
        component_fraction = (
            topology["component_size"].get(n, 0) / max(topology["road_count"], 1)
        )
        failed_nbr_count = failed_nbrs[n]
        failed_nbr_ratio = failed_nbr_count / max(degree, 1)
        exposure_score = float(d.get("flood_risk", 0)) * scale

        rows.append({
            "simulation_id": simulation_id,
            "run_index": run_index,
            "traffic": d.get("traffic", 0),
            "length": d.get("length", 0),
            "rain": d.get("rain_mm_mean", 0),
            "flood_risk": d.get("flood_risk", 0),
            "degree": degree,
            "clustering": topology["clustering"].get(n, 0),
            "core_number": topology["core_numbers"].get(n, 0),
            "avg_neighbor_degree": topology["avg_neighbor_degree"].get(n, 0),
            "pagerank": topology["pagerank"].get(n, 0),
            "is_articulation_point": int(n in topology["articulation_points"]),
            "component_fraction": component_fraction,
            "nbr_risk": neighbor_risk[n],
            "nbr_traffic": neighbor_traffic[n],
            "nbr_degree": neighbor_degree[n],
            "nbr_clustering": neighbor_clustering[n],
            "failed_nbrs": failed_nbr_count,
            "failed_nbr_ratio": failed_nbr_ratio,
            "exposure_score": exposure_score,
            "scale": scale,
            "initial_failed": int(n in initial_failed),
            "failed": int(d.get("failed", False)),
        })

    return rows, G


def build_dataset():

    all_rows = []

    for scale, run_count in SCALE_RUNS.items():
        print("Running scenario:", scale)

        for run_index in range(run_count):
            rows, _ = simulate_once(scale, run_index=run_index)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df.replace([float("inf"), -float("inf")], 0)
    df = df.fillna(0)

    # Train the main baseline on roads that were alive before the cascade.
    # This gives the model a harder and more useful prediction target.
    alive_df = df[df["initial_failed"] == 0].copy()

    alive_df.to_csv(OUT_PATH, index=False)

    print("\nSaved dataset:", OUT_PATH)
    print("\nOriginal class balance:")
    print(df["failed"].value_counts())
    print("\nAlive-road class balance:")
    print(alive_df["failed"].value_counts())


if __name__ == "__main__":
    build_dataset()
