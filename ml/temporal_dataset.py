import pickle
import pandas as pd
import random
from pathlib import Path

from graph.cascade import cascade_step_capacity

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
OUT_PATH = BASE_DIR / "data" / "temporal_dataset.csv"

SCALES = [0.5, 0.7, 0.9, 1.1, 1.3]
RUNS_PER_SCALE = 2
MAX_STEPS = 8
MAX_ROWS = 2_000_000


def simulate_temporal(scale):

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # reset graph
    for n, d in G.nodes(data=True):
        if d.get("type") == "road":
            d["failed"] = False
            d["base_traffic"] = d["traffic"]
            d["capacity"] = d["traffic"] * 2.0

    # flood seeding
    for n, d in G.nodes(data=True):
        if d.get("type") == "road":
            risk = d.get("flood_risk", 0)
            if random.random() < risk * scale * 0.6:
                d["failed"] = True

    rows = []
    degrees = dict(G.degree())

    for step in range(MAX_STEPS):

        current_state = {
            n: d.get("failed", False)
            for n, d in G.nodes(data=True)
        }

        new_failures = cascade_step_capacity(G, overload_factor=1.2)

        # compute neighbor features
        num_failed_neighbors = {}
        neighbor_avg_risk = {}
        neighbor_avg_traffic = {}

        for n in G.nodes():
            nbrs = list(G.neighbors(n))

            if not nbrs:
                num_failed_neighbors[n] = 0
                neighbor_avg_risk[n] = 0
                neighbor_avg_traffic[n] = 0
                continue

            failed_count = 0
            risks = []
            traffics = []

            for nbr in nbrs:
                d_nbr = G.nodes[nbr]

                if current_state.get(nbr, False):
                    failed_count += 1

                if d_nbr.get("type") == "road":
                    risks.append(d_nbr.get("flood_risk", 0))
                    traffics.append(d_nbr.get("traffic", 0))

            num_failed_neighbors[n] = failed_count
            neighbor_avg_risk[n] = sum(risks) / len(risks) if risks else 0
            neighbor_avg_traffic[n] = sum(traffics) / len(traffics) if traffics else 0

        # record rows
        for n, d in G.nodes(data=True):

            if d.get("type") != "road":
                continue

            rows.append({
                "traffic": d.get("traffic", 0),
                "length": d.get("length", 0),
                "rain": d.get("rain_mm_mean", 0),
                "flood_risk": d.get("flood_risk", 0),
                "scale": scale,
                "step": step,
                "degree": degrees[n],
                "num_failed_neighbors": num_failed_neighbors[n],
                "neighbor_avg_risk": neighbor_avg_risk[n],
                "neighbor_avg_traffic": neighbor_avg_traffic[n],
                "failed_now": int(current_state[n]),
                "failed_next": int(d.get("failed", False)),
            })

        if not new_failures:
            break

        if len(rows) > MAX_ROWS:
            break

    return rows


def build_dataset():

    all_rows = []

    for scale in SCALES:
        print("Scenario:", scale)

        for _ in range(RUNS_PER_SCALE):

            rows = simulate_temporal(scale)

            if len(rows) > 200000:
                rows = random.sample(rows, 200000)

            all_rows.extend(rows)

            if len(all_rows) > MAX_ROWS:
                break

        if len(all_rows) > MAX_ROWS:
            break

    df = pd.DataFrame(all_rows)

    df = df.replace([float("inf"), -float("inf")], 0)
    df = df.fillna(0)

    df.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH)
    print("\nClass balance:")
    print(df["failed_next"].value_counts())


if __name__ == "__main__":
    build_dataset()