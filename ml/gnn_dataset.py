import pickle
import torch
from pathlib import Path
from torch_geometric.data import Data

BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_after_cascade_weather_1.2.gpickle"


def build_gnn_data():

    print("Loading graph...")

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # ---- Only road nodes ----
    node_list = [n for n, d in G.nodes(data=True) if d.get("type") == "road"]
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    # ---- Graph statistics ----
    degrees = dict(G.degree())

    neighbor_risk = {}
    neighbor_traffic = {}

    for n in node_list:
        nbrs = list(G.neighbors(n))

        if len(nbrs) == 0:
            neighbor_risk[n] = 0
            neighbor_traffic[n] = 0
            continue

        risks = []
        traff = []

        for nbr in nbrs:
            d = G.nodes[nbr]
            if d.get("type") == "road":
                risks.append(d.get("flood_risk", 0))
                traff.append(d.get("traffic", 0))

        neighbor_risk[n] = sum(risks) / len(risks) if risks else 0
        neighbor_traffic[n] = sum(traff) / len(traff) if traff else 0

    # ---- Features and labels ----
    features = []
    labels = []

    for n in node_list:
        d = G.nodes[n]

        cap_ratio = 0
        if d.get("capacity", 0) > 0:
            cap_ratio = d.get("traffic", 0) / d.get("capacity", 1)

        x = [
            d.get("traffic", 0),
            d.get("length", 0),
            d.get("rain_mm_mean", 0),
            d.get("flood_risk", 0),
            degrees[n],
            neighbor_risk[n],
            neighbor_traffic[n],
            cap_ratio,
        ]

        features.append(x)
        labels.append(int(d.get("failed", False)))

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # ---- Edges ----
    edge_index = []

    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_index.append([node_to_idx[v], node_to_idx[u]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)

    print(data)
    return data


if __name__ == "__main__":
    data = build_gnn_data()