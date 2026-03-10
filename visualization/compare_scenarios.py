import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

SCALES = [0.7, 1.2, 2.5]

plt.figure()

for s in SCALES:

    path = BASE_DIR / "data" / "graphs" / f"run_results_weather_{s}.json"

    if not path.exists():
        print("Missing:", path)
        continue

    with open(path) as f:
        results = json.load(f)

    history = results["capacity_cascade"]
    steps = list(range(1, len(history) + 1))

    plt.plot(steps, history, marker="o", label=f"Scale {s}")

plt.xlabel("Cascade Step")
plt.ylabel("New Failures")
plt.title("Cascade Severity Under Different Rainfall Scenarios")
plt.legend()
plt.grid(True)

plt.show()
