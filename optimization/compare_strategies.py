import json
import pickle
import random
import statistics
from pathlib import Path

from optimization.greedy_repair import greedy_select
from graph.cascade import cascade_step_capacity

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
OUT_PATH = BASE_DIR / "data" / "graphs" / "opt_vs_random_scale_1.2.json"

SCALE = 1.2
BUDGETS = [0, 500, 2000, 5000]
N_REPLICATIONS = 7
CAPACITY_FACTOR = 1.35
OVERLOAD_FACTOR = 1.25
PROTECTION_CAPACITY_MULTIPLIER = 3.0
PROTECTION_RISK_MULTIPLIER = 0.15
PROTECTION_STRESS_MULTIPLIER = 0.25
MAX_STEPS = 20
BASE_SEED = 20260504


def load_graph():
    with open(GRAPH_PATH, "rb") as file:
        return pickle.load(file)


def copy_graph(graph):
    return pickle.loads(pickle.dumps(graph))


def road_nodes(graph):
    return [node_id for node_id, data in graph.nodes(data=True) if data.get("type") == "road"]


def reset_graph(graph):
    for _, data in graph.nodes(data=True):
        if data.get("type") != "road":
            continue
        traffic = float(data.get("traffic", 0.0))
        data["base_traffic"] = traffic
        data["capacity"] = max(traffic * CAPACITY_FACTOR, 1.0)
        data["failed"] = False
        data["protected"] = False
        data["stress"] = 0.0


def reinforce(graph, road_ids):
    for road_id in road_ids:
        if road_id not in graph.nodes:
            continue
        data = graph.nodes[road_id]
        if data.get("type") != "road":
            continue
        data["protected"] = True
        data["failed"] = False
        data["capacity"] = max(data.get("capacity", 1.0) * PROTECTION_CAPACITY_MULTIPLIER, 1.0)


def seed_flood_failures(graph, rng):
    initial_failures = 0
    for _, data in graph.nodes(data=True):
        if data.get("type") != "road":
            continue

        risk = max(0.0, min(1.0, float(data.get("flood_risk", 0.0))))
        if data.get("protected"):
            risk *= PROTECTION_RISK_MULTIPLIER

        failed = (risk * SCALE) > rng.random()
        data["failed"] = failed
        if failed:
            initial_failures += 1

    return initial_failures


def protected_cascade_step(graph, rng):
    new_failures = []
    failed_nodes = [node_id for node_id, data in graph.nodes(data=True) if data.get("failed")]
    if not failed_nodes:
        return []

    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "road" or data.get("failed"):
            continue

        base = data.get("base_traffic", data.get("traffic", 0))
        capacity = max(data.get("capacity", 1.0), 1.0)

        failed_neighbors = 0
        for neighbor in graph.neighbors(node_id):
            neighbor_data = graph.nodes[neighbor]
            if neighbor_data.get("type") == "road" and neighbor_data.get("failed"):
                failed_neighbors += 1

        if failed_neighbors == 0:
            continue

        overload = base + OVERLOAD_FACTOR * base * failed_neighbors
        prev_stress = data.get("stress", 0.0)
        stress = prev_stress + overload / capacity
        if data.get("protected"):
            stress *= PROTECTION_STRESS_MULTIPLIER
        data["stress"] = stress

        if stress > 1.0:
            probability = min(1.0, (stress - 1.0) * 1.5)
            if data.get("protected"):
                probability *= PROTECTION_STRESS_MULTIPLIER
            if rng.random() < probability:
                data["failed"] = True
                new_failures.append(node_id)

    return new_failures


def run_cascade(graph, rng, steps=MAX_STEPS):
    history = []
    for _ in range(steps):
        new_failures = protected_cascade_step(graph, rng)
        history.append(len(new_failures))
        if not new_failures:
            break
    return history


def count_failed(graph):
    return sum(
        1
        for _, data in graph.nodes(data=True)
        if data.get("type") == "road" and data.get("failed")
    )


def choose_roads(graph, budget_k, strategy, rng):
    if budget_k <= 0:
        return []
    if strategy == "greedy":
        return greedy_select(budget_k)
    if strategy == "random":
        roads = road_nodes(graph)
        sample_size = min(budget_k, len(roads))
        return rng.sample(roads, sample_size)
    raise ValueError(f"Unknown strategy: {strategy}")


def experiment(budget_k, strategy="greedy", replication=0):
    base_graph = load_graph()
    graph = copy_graph(base_graph)
    reset_graph(graph)

    seed = BASE_SEED + (budget_k * 17) + (replication * 101) + (0 if strategy == "greedy" else 1)
    rng = random.Random(seed)

    chosen = choose_roads(graph, budget_k, strategy, rng)
    reinforce(graph, chosen)
    initial_failures = seed_flood_failures(graph, rng)
    cascade_history = run_cascade(graph, rng)
    total_failed = count_failed(graph)
    total_roads = len(road_nodes(graph))

    return {
        "strategy": strategy,
        "budget_k": budget_k,
        "replication": replication,
        "protected_count": len(chosen),
        "initial_failures": initial_failures,
        "cascade_history": cascade_history,
        "failed": total_failed,
        "resilience_score": max(0.0, 1.0 - (total_failed / max(total_roads, 1))),
        "total_roads": total_roads,
    }


def summarize_runs(runs):
    failed_values = [run["failed"] for run in runs]
    resilience_values = [run["resilience_score"] for run in runs]
    initial_values = [run["initial_failures"] for run in runs]
    peak_step_values = [max(run["cascade_history"]) if run["cascade_history"] else 0 for run in runs]

    return {
        "mean_failed": round(statistics.mean(failed_values), 2),
        "std_failed": round(statistics.pstdev(failed_values), 2),
        "mean_resilience": round(statistics.mean(resilience_values), 4),
        "std_resilience": round(statistics.pstdev(resilience_values), 4),
        "mean_initial_failures": round(statistics.mean(initial_values), 2),
        "mean_peak_step_failures": round(statistics.mean(peak_step_values), 2),
        "replications": len(runs),
    }


def compare_budget(budget_k):
    greedy_runs = [experiment(budget_k, "greedy", replication=i) for i in range(N_REPLICATIONS)]
    random_runs = [experiment(budget_k, "random", replication=i) for i in range(N_REPLICATIONS)]

    greedy_summary = summarize_runs(greedy_runs)
    random_summary = summarize_runs(random_runs)

    failed_reduction = round(random_summary["mean_failed"] - greedy_summary["mean_failed"], 2)
    resilience_gain = round(greedy_summary["mean_resilience"] - random_summary["mean_resilience"], 4)

    return {
        "K": budget_k,
        "greedy_failed": greedy_summary["mean_failed"],
        "greedy_std_failed": greedy_summary["std_failed"],
        "greedy_resilience": greedy_summary["mean_resilience"],
        "greedy_initial_failures": greedy_summary["mean_initial_failures"],
        "greedy_peak_step_failures": greedy_summary["mean_peak_step_failures"],
        "random_failed": random_summary["mean_failed"],
        "random_std_failed": random_summary["std_failed"],
        "random_resilience": random_summary["mean_resilience"],
        "random_initial_failures": random_summary["mean_initial_failures"],
        "random_peak_step_failures": random_summary["mean_peak_step_failures"],
        "failed_reduction_vs_random": failed_reduction,
        "resilience_gain_vs_random": resilience_gain,
        "replications": N_REPLICATIONS,
    }


def main():
    results = []
    for budget_k in BUDGETS:
        print(f"\nBudget K={budget_k}")
        summary = compare_budget(budget_k)
        results.append(summary)
        print("Greedy mean failed:", summary["greedy_failed"])
        print("Random mean failed:", summary["random_failed"])
        print("Failed reduction vs random:", summary["failed_reduction_vs_random"])
        print("Resilience gain vs random:", summary["resilience_gain_vs_random"])

    with open(OUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
