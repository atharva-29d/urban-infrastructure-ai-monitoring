import pickle
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_after_cascade_weather_0.7.gpickle"
OUT_PATH = BASE_DIR / "static" / "data" / "graphs" / "graph_live.json"


def export():

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    nodes = []
    edges = []

    for n, d in G.nodes(data=True):

        if d.get("type") != "road":
            continue

        nodes.append({
            "id": n,
            "lat": d.get("lat", 18.52),
            "lon": d.get("lon", 73.85),
            "failed": d.get("failed", False),
            "traffic": d.get("traffic", 0),
            "risk": d.get("flood_risk", 0)
        })

    for u, v, d in G.edges(data=True):
        edges.append({
            "u": u,
            "v": v
        })

    with open(OUT_PATH, "w") as f:
        json.dump({
            "nodes": nodes,
            "edges": edges
        }, f)

    print("Saved JSON →", OUT_PATH)


if __name__ == "__main__":
    export()