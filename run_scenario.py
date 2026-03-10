import pickle
import json
import random
from pathlib import Path

from graph.cascade import cascade_step_capacity

# -------------------------------------------------
# Scenario severity
# -------------------------------------------------

SCALE = 2.5  # try 0.7 (mild), 1.5 (severe), 2.5 (extreme)


# ---------------------------------------------
# Scenario tuning
# ---------------------------------------------

if SCALE <= 0.8:          # mild
    CAPACITY_FACTOR = 1.5
    OVERLOAD_FACTOR = 1.2
elif SCALE <= 1.6:        # severe
    CAPACITY_FACTOR = 1.25
    OVERLOAD_FACTOR = 1.5
else:                     # extreme
    CAPACITY_FACTOR = 1.1
    OVERLOAD_FACTOR = 1.8

# -------------------------------------------------
# Paths (robust to run location)
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"

RESULTS_PATH = BASE_DIR / "data" / "graphs" / f"run_results_weather_{SCALE}.json"

FINAL_GRAPH_PATH = BASE_DIR / "data" / "graphs" / f"pune_after_cascade_weather_{SCALE}.gpickle"


# -------------------------------------------------
# Initial failure injection using flood risk
# -------------------------------------------------

def seed_failures_from_flood_risk(G, scale=1.0):
    """
    Fail road nodes probabilistically based on flood_risk.
    scale controls severity of storm.
    """

    failed = []

    for n, d in G.nodes(data=True):

        if d.get("type") != "road":
            continue

        risk = d.get("flood_risk", 0.0)

        # clamp safety
        risk = max(0.0, min(1.0, risk))

        if random.random() < risk * scale:
            d["failed"] = True
            failed.append(n)

    return failed


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    # ---------------------------------------------
    # Load weather-aware base graph
    # ---------------------------------------------

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    print("Graph loaded.")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # ---------------------------------------------
    # Assign capacity + reset states
    # ---------------------------------------------

    for n, d in G.nodes(data=True):

        if d.get("type") == "road":

            # base traffic used for rerouting logic
            d["base_traffic"] = d["traffic"]
            d["capacity"] = d["traffic"] * CAPACITY_FACTOR

            d["failed"] = False
            d["overloaded"] = False

    # ---------------------------------------------
    # Seed failures from rainfall risk
    # ---------------------------------------------



    initial_failed = seed_failures_from_flood_risk(G, scale=SCALE)

    print(f"Initial failed roads from flood risk: {len(initial_failed)}")

    # ---------------------------------------------
    # Run cascade simulation
    # ---------------------------------------------

    history = []

    for step in range(20):

        new_failures = cascade_step_capacity(G, overload_factor=OVERLOAD_FACTOR)

        history.append(len(new_failures))

        print(f"Step {step+1}: new failures = {len(new_failures)}")

        if len(new_failures) == 0:
            print("System stabilized.")
            break

    # ---------------------------------------------
    # Save results JSON
    # ---------------------------------------------

    results = {
        "initial_failures": len(initial_failed),
        "capacity_cascade": history,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved results to:", RESULTS_PATH)

    # ---------------------------------------------
    # Save final post-cascade graph for visualization
    # ---------------------------------------------

    with open(FINAL_GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)

    print("Saved final graph to:", FINAL_GRAPH_PATH)


if __name__ == "__main__":
    main()
