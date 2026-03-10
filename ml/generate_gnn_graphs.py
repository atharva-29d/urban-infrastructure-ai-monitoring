import pickle
import random
from pathlib import Path

from graph.cascade import cascade_step_capacity

BASE_DIR = Path(__file__).resolve().parents[1]

BASE_GRAPH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
OUT_DIR = BASE_DIR / "data" / "graphs"

SCALES = [0.5, 0.7, 1.0, 1.2, 1.5]


def generate_graph(scale):

    with open(BASE_GRAPH, "rb") as f:
        G = pickle.load(f)

    # Reset
    for n, d in G.nodes(data=True):
        if d.get("type") == "road":
            d["failed"] = False
            d["stress"] = 0  # ← ADD THIS
            d["capacity"] = d.get("traffic", 1) * 2.2

    # Flood seed
    for n, d in G.nodes(data=True):
        if d.get("type") == "road":
            risk = d.get("flood_risk", 0)
            if random.random() < risk * scale * 0.6:
                d["failed"] = True

    # Cascade
    for _ in range(20):
        new = cascade_step_capacity(G, overload_factor=1.2)
        if not new:
            break

    return G


def main():

    print("Generating GNN training graphs...")

    for s in SCALES:
        print("Running scenario:", s)

        G = generate_graph(s)

        path = OUT_DIR / f"graph_scale_{s}.gpickle"

        with open(path, "wb") as f:
            pickle.dump(G, f)

        print("Saved:", path)


if __name__ == "__main__":
    main()