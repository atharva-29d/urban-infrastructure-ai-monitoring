import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

PATH = BASE_DIR / "data" / "graphs" / "repair_impact_scale_1.2.json"

with open(PATH) as f:
    rows = json.load(f)

Ks = [r["K"] for r in rows]
frac = [r["frac_failed"] for r in rows]

plt.figure()
plt.plot(Ks, frac, marker="o")
plt.xlabel("Number of Roads Reinforced")
plt.ylabel("Fraction of Roads Failed")
plt.title("Effect of Targeted Repairs Under Severe Flood")
plt.grid(True)

plt.show()
