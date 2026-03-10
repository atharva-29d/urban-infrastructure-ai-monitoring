import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_PATH = BASE_DIR / "data" / "graphs" / "run_results_weather_2.5.json"
OUT_PATH = BASE_DIR / "visualization" / "cascade_progression.png"


def main():

    print("Loading cascade results...")

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    failures = data["capacity_cascade"]

    steps = list(range(1, len(failures) + 1))

    plt.figure(figsize=(8,5))

    plt.plot(steps, failures, marker="o")

    plt.xlabel("Cascade Step")
    plt.ylabel("New Failures")
    plt.title("Cascade Failure Progression")

    plt.grid(True)

    plt.savefig(OUT_PATH, dpi=300)

    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()