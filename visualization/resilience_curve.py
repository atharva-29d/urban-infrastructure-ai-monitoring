import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "graphs" / "repair_impact_scale_1.2.json"
OUT_PATH = BASE_DIR / "visualization" / "resilience_curve.png"


def main():

    print("Loading repair impact data...")

    with open(DATA_PATH) as f:
        results = json.load(f)

    budgets = []
    frac_failed = []

    for r in results:
        budgets.append(r["K"])
        frac_failed.append(r["frac_failed"])

    resilience = [1 - f for f in frac_failed]

    plt.figure(figsize=(8,5))

    plt.plot(budgets, resilience, marker="o")

    plt.xlabel("Repair Budget (Number of Roads Reinforced)")
    plt.ylabel("Network Resilience (Fraction Surviving)")
    plt.title("Infrastructure Resilience vs Repair Budget")

    plt.grid(True)

    plt.savefig(OUT_PATH, dpi=300)

    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()