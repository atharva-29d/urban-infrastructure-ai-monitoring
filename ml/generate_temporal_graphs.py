import pickle
import random
from pathlib import Path

from graph.cascade import cascade_step_capacity

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
OUT_DIR = BASE_DIR / "data" / "temporal_graphs"

SCALES = [0.5, 0.7, 0.9, 1.1]
MAX_STEPS = 8
RUNS_PER_SCALE = 2


def simulate_and_save(scale, run_id):

    print(f"Scale {scale} Run {run_id}")

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

    # directory
    save_dir = OUT_DIR / f"scale_{scale}" / f"run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # temporal steps
    for step in range(MAX_STEPS):

        # save snapshot
        path = save_dir / f"step_{step}.gpickle"
        with open(path, "wb") as f:
            pickle.dump(G, f)

        print("Saved:", path)

        new_failures = cascade_step_capacity(G, overload_factor=1.2)

        if not new_failures:
            break


def main():

    for scale in SCALES:
        for run in range(RUNS_PER_SCALE):
            simulate_and_save(scale, run)


if __name__ == "__main__":
    main()