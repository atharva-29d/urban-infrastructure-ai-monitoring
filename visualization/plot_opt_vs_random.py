import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

PATH = BASE_DIR / "data" / "graphs" / "opt_vs_random_scale_1.2.json"

with open(PATH) as f:
    rows = json.load(f)

Ks = [r["K"] for r in rows]

g = [r["greedy_failed"] for r in rows]
r = [r["random_failed"] for r in rows]

plt.figure()
plt.plot(Ks, g, marker="o", label="Greedy Optimizer")
plt.plot(Ks, r, marker="o", label="Random Selection")

plt.xlabel("Repair Budget (K roads)")
plt.ylabel("Number of Roads Failed")
plt.title("Optimized vs Random Repair Strategies (Severe Flood)")
plt.legend()
plt.grid(True)

plt.show()
