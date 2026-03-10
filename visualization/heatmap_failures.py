import pickle
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_after_cascade_weather_0.7.gpickle"

ROADS_PATH = BASE_DIR / "data" / "gis" / "roads_final.geojson"

# -----------------------------
# Load graph
# -----------------------------

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

failed = {
    n for n, d in G.nodes(data=True)
    if d.get("type") == "road" and d.get("failed")
}

# -----------------------------
# Load roads GIS
# -----------------------------

roads = gpd.read_file(ROADS_PATH)

roads["failed"] = roads["road_id"].isin(failed)

roads = roads.to_crs(epsg=32643)  # UTM for Pune

# -----------------------------
# Build spatial grid
# -----------------------------

xmin, ymin, xmax, ymax = roads.total_bounds

CELL = 1000 # meters

xs = list(range(int(xmin), int(xmax), CELL))
ys = list(range(int(ymin), int(ymax), CELL))

from shapely.geometry import box

polys = []

for x in xs:
    for y in ys:
        polys.append(
            box(x, y, x + CELL, y + CELL)
        )

grid = gpd.GeoDataFrame({"geometry": polys}, crs=roads.crs)

# -----------------------------
# Spatial join
# -----------------------------

join = gpd.sjoin(roads, grid, predicate="intersects")

agg = (
    join.groupby(join.index_right)["failed"]
    .mean()
    .reset_index()
)

grid["fail_frac"] = 0.0
grid.loc[agg["index_right"], "fail_frac"] = agg["failed"]

## -----------------------------
# Plot heatmap (clean)
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 10))

# only cells that contain roads
grid_plot = grid[grid["fail_frac"] > 0]

grid_plot.plot(
    column="fail_frac",
    cmap="inferno",
    linewidth=0.3,
    edgecolor="gray",
    ax=ax,
    legend=True,
)

# very light road outline for context
roads.boundary.plot(ax=ax, color="white", linewidth=0.05, alpha=0.3)

ax.set_title("Spatial Heatmap of Road Failures (Severe Flood)")
ax.axis("off")

plt.show()
