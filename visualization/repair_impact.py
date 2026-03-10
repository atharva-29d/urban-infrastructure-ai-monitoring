import pickle
import json
import random
from pathlib import Path
import pandas as pd

from graph.cascade import cascade_step_capacity

BASE_DIR = Path(__file__).resolve().parents[1]

# ----------------------------------
# Config
# ----------------------------------

SCALE = 1

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
CRITICAL_PATH = BASE_DIR / "data" / "graphs" / "critical_roads.csv"

OUT_JSON = BASE_DIR / "data" / "graphs" / "repair_impact_scale_1.2.json"


# ----------------------------------
# Failure seeding
# ----------------------------------

def seed_failures(G, scale):

    failed = []

    for n, d in G.nodes(data=True):

        if d.get("type") != "road":
            continue

        risk = d.get("flood_risk", 0)
        risk = max(0.0, min(1.0, risk))

        if random.random() < risk * scale:
            d["failed"] = True
            failed.append(n)

    return failed


# ----------------------------------
# Reset + capacity
# ----------------------------------

def reset_graph(G):

    for _, d in G.nodes(data=True):

        if d.get("type") == "road":
            d["failed"] = False
            d["overloaded"] = False
            d["base_traffic"] = d["traffic"]
            d["capacity"] = d["traffic"] * 1.25   # severe tuning


# ----------------------------------
# Main experiment
# ----------------------------------

def run_for_k(K, critical_ids):

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    reset_graph(G)

    # protect top-K roads
    protected = set(critical_ids[:K])

    for n in protected:
        if n in G.nodes:
            G.nodes[n]["protected"] = True
            G.nodes[n]["capacity"] *= 2.0


    # initial flood failures (skip protected)
    failed = []

    for n, d in G.nodes(data=True):

        if d.get("type") != "road":
            continue

        if d.get("protected"):
            continue

        risk = d.get("flood_risk", 0)

        if random.random() < risk * SCALE:
            d["failed"] = True
            failed.append(n)

    # run cascade
    # run cascade
    for _ in range(20 ):
        new = cascade_step_capacity(G, overload_factor=1.5)
        if not new:
            break


    roads = [n for n, d in G.nodes(data=True) if d.get("type") == "road"]
    failed = [n for n in roads if G.nodes[n].get("failed")]

    return len(failed), len(roads)


# ----------------------------------
# Run sweep
# ----------------------------------

critical = pd.read_csv(CRITICAL_PATH)

Ks = [0, 500, 2000, 5000, 10000]


rows = []

for K in Ks:

    print("Running for K =", K)

    f, total = run_for_k(K, critical["road_id"].tolist())

    rows.append({
        "K": K,
        "failed": f,
        "total": total,
        "frac_failed": f / total,
    })

with open(OUT_JSON, "w") as f:
    json.dump(rows, f, indent=2)

print("Saved:", OUT_JSON)

print("\nResults:")
for r in rows:
    print(r)
