import pickle
from pathlib import Path
import networkx as nx

BASE_DIR = Path(__file__).resolve().parents[1]

SCALES = [0.7, 1.2, 2.5]

for s in SCALES:

    path = BASE_DIR / "data" / "graphs" / f"pune_after_cascade_weather_{s}.gpickle"

    if not path.exists():
        print("Missing graph for scale", s)
        continue

    with open(path, "rb") as f:
        G = pickle.load(f)

    roads = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "road"
    ]

    failed = [
        n for n in roads
        if G.nodes[n].get("failed")
    ]

    frac_failed = len(failed) / len(roads)

    # surviving subgraph
    alive = [n for n in roads if n not in failed]

    H = G.subgraph(alive)

    if len(H) > 0:
        lcc = max(nx.connected_components(H), key=len)
        lcc_frac = len(lcc) / len(roads)
    else:
        lcc_frac = 0.0

    print("\n--- Scale", s, "---")
    print("Roads total:", len(roads))
    print("Failed:", len(failed))
    print("Fraction failed:", round(frac_failed, 3))
    print("Largest surviving component fraction:", round(lcc_frac, 3))
