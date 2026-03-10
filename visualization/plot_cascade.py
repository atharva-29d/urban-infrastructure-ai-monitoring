import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = BASE_DIR / "data" / "graphs" / "run_results_weather.json"

with open(RESULTS_PATH) as f:
    results = json.load(f)

history = results["capacity_cascade"]

steps = list(range(1, len(history) + 1))

plt.figure()
plt.plot(steps, history, marker="o")
plt.xlabel("Cascade Step")
plt.ylabel("New Failures")
plt.title("Cascading Failures Over Time (Flood Scenario)")
plt.grid(True)

plt.show()
