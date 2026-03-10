import pickle
import random
import json
from pathlib import Path

from optimization.greedy_repair import greedy_select
from graph.cascade import cascade_step_capacity

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
OUT_PATH = BASE_DIR / "data" / "graphs" / "opt_vs_random_scale_1.2.json"


SCALE = 1.2
Ks = [0, 500, 2000, 5000]


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def copy_graph(G):
    return pickle.loads(pickle.dumps(G))


def seed_flood_failures(G):
    for n, d in G.nodes(data=True):
        if d.get("type") == "road":
            if d.get("flood_risk", 0) * SCALE > random.random():
                d["failed"] = True
            else:
                d["failed"] = False


def reinforce(G, road_ids):
    for n in road_ids:
        if n in G.nodes:
            G.nodes[n]["protected"] = True
            G.nodes[n]["failed"] = False
            G.nodes[n]["capacity"] *= 2.0


def run_cascade(G, steps=20):
    for _ in range(steps):
        new = cascade_step_capacity(G, overload_factor=1.5)
        if not new:
            break


def count_failed(G):
    return sum(1 for _, d in G.nodes(data=True)
               if d.get("type") == "road" and d.get("failed"))


def experiment(K, strategy="greedy"):
    base = load_graph()
    G = copy_graph(base)

    # reset traffic & capacity
    for n, d in G.nodes(data=True):
        if d.get("type") == "road":
            d["base_traffic"] = d["traffic"]
            d["capacity"] = d["traffic"] * 1.25  # same as severe tuning
            d["failed"] = False
            d["protected"] = False

    # choose roads
    if strategy == "greedy":
        chosen = greedy_select(K)
    elif strategy == "random":
        roads = [n for n, d in G.nodes(data=True) if d.get("type") == "road"]
        chosen = random.sample(roads, K)
    else:
        chosen = []

    reinforce(G, chosen)
    seed_flood_failures(G)
    run_cascade(G)

    return count_failed(G)


def main():

    results = []

    for K in Ks:
        print(f"\nBudget K={K}")

        N = 5

        g_vals = [experiment(K, "greedy") for _ in range(N)]
        r_vals = [experiment(K, "random") for _ in range(N)]

        f_greedy = sum(g_vals) / N
        f_rand = sum(r_vals) / N

        results.append({
            "K": K,
            "greedy_failed": f_greedy,
            "random_failed": f_rand
        })

        print("Greedy:", f_greedy)
        print("Random:", f_rand)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
