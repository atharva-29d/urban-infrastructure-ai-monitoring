import pickle
from pathlib import Path
import pandas as pd
import networkx as nx

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
OUT_PATH = BASE_DIR / "data" / "graphs" / "critical_roads.csv"

print("Loading graph...")

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

# -----------------------------------
# Roads only
# -----------------------------------

roads = [n for n, d in G.nodes(data=True) if d.get("type") == "road"]
H = G.subgraph(roads)

print("Computing degree centrality...")

deg = nx.degree_centrality(H)

rows = []

for n in roads:
    d = G.nodes[n]

    rows.append({
        "road_id": n,
        "degree_centrality": deg.get(n, 0),
        "traffic": d.get("traffic", 0),
        "flood_risk": d.get("flood_risk", 0),
    })

df = pd.DataFrame(rows)

# -----------------------------------
# Normalize + combine into score
# -----------------------------------

for col in ["degree_centrality", "traffic", "flood_risk"]:
    df[col + "_norm"] = df[col].rank(pct=True)

df["criticality"] = (
    0.4 * df["degree_centrality_norm"]
    + 0.35 * df["traffic_norm"]
    + 0.25 * df["flood_risk_norm"]
)

df = df.sort_values("criticality", ascending=False)

df.to_csv(OUT_PATH, index=False)

print("Saved critical road ranking to:", OUT_PATH)
print("\nTop 10 Critical Roads:")
print(df.head(10))
