import pickle
import pandas as pd
import random
from pathlib import Path

from graph.cascade import cascade_step_capacity

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
OUT_PATH = BASE_DIR / "data" / "ml_dataset.csv"

SCALES = [0.5, 0.7, 1.0, 1.2, 1.5]


def simulate_once(scale):

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

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
            if random.random() < risk * scale * 0.6:
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

    # Degree
    degrees = dict(G.degree())

    # Neighbour statistics
    neighbor_risk = {}
    neighbor_traffic = {}

    for n in G.nodes():
        nbrs = list(G.neighbors(n))

        if len(nbrs) == 0:
            neighbor_risk[n] = 0
            neighbor_traffic[n] = 0
            continue

        risks = []
        traff = []

        for nbr in nbrs:
            d = G.nodes[nbr]
            if d.get("type") == "road":
                risks.append(d.get("flood_risk", 0))
                traff.append(d.get("traffic", 0))

        neighbor_risk[n] = sum(risks) / len(risks) if risks else 0
        neighbor_traffic[n] = sum(traff) / len(traff) if traff else 0

    # Initial failed neighbours
    failed_nbrs = {}

    for n in G.nodes():
        count = 0
        for nbr in G.neighbors(n):
            if nbr in initial_failed:
                count += 1
        failed_nbrs[n] = count

    rows = []

    for n, d in G.nodes(data=True):
        if d.get("type") != "road":
            continue

        rows.append({
            "traffic": d.get("traffic", 0),
            "length": d.get("length", 0),
            "rain": d.get("rain_mm_mean", 0),
            "flood_risk": d.get("flood_risk", 0),
            "degree": degrees[n],
            "nbr_risk": neighbor_risk[n],
            "nbr_traffic": neighbor_traffic[n],
            "failed_nbrs": failed_nbrs[n],
            "scale": scale,
            "failed": int(d.get("failed", False)),
        })

    return rows, G


def build_dataset():

    all_rows = []

    for scale in SCALES:
        print("Running scenario:", scale)

        for _ in range(5):
            rows, _ = simulate_once(scale)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df.replace([float("inf"), -float("inf")], 0)
    df = df.fillna(0)

    df.to_csv(OUT_PATH, index=False)

    print("\nSaved dataset:", OUT_PATH)
    print("\nClass balance:")
    print(df["failed"].value_counts())


if __name__ == "__main__":
    build_dataset()