import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_after_cascade_weather.gpickle"
ROADS_PATH = BASE_DIR / "data" / "gis" / "roads_final.geojson"

# -------------------------------------------------
# Load graph
# -------------------------------------------------

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

failed_roads = {
    n for n, d in G.nodes(data=True)
    if d.get("type") == "road" and d.get("failed")
}

print("Failed roads in graph:", len(failed_roads))

# -------------------------------------------------
# Load GIS
# -------------------------------------------------

roads = gpd.read_file(ROADS_PATH)

# -------------------------------------------------
# Attach failure flag
# -------------------------------------------------

roads["failed"] = roads["road_id"].isin(failed_roads)

print("Failed roads in GIS:", roads["failed"].sum())

# -------------------------------------------------
# Plot (optimized)
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 10))

# plot subset of intact roads for speed
roads_ok = (
    roads[~roads["failed"]]
    .sample(frac=0.25, random_state=42)
    if len(roads) > 20000
    else roads[~roads["failed"]]
)

roads_ok.plot(ax=ax, color="lightgray", linewidth=0.3)

roads[roads["failed"]].plot(ax=ax, color="red", linewidth=1.2)

ax.set_title("Road Failures After Flood Cascade")
ax.axis("off")

plt.show()
