import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import networkx as nx
import torch
from torch_geometric.data import Data

BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_DIR = BASE_DIR / "data" / "graphs"

SCALES = [0.5, 0.7, 1.0, 1.2, 1.5]


def compute_neighbor_stats(G):

    nbr_risk = {}
    nbr_traffic = {}

    for n in G.nodes():
        risks = []
        traff = []

        for nbr in G.neighbors(n):
            d = G.nodes[nbr]
            if d.get("type") == "road":
                risks.append(d.get("flood_risk", 0))
                traff.append(d.get("traffic", 0))

        if len(risks) == 0:
            nbr_risk[n] = 0
            nbr_traffic[n] = 0
        else:
            nbr_risk[n] = np.mean(risks)
            nbr_traffic[n] = np.mean(traff)

    return nbr_risk, nbr_traffic


def build_graph(scale):

    path = GRAPH_DIR / f"graph_scale_{scale}.gpickle"
    print("Loading:", path.name)

    with open(path, "rb") as f:
        G = pickle.load(f)

    # Degree and clustering
    degree = dict(G.degree())
    clustering = nx.clustering(G)

    # Neighbor aggregation
    nbr_risk, nbr_traffic = compute_neighbor_stats(G)

    nodes = []
    features = []
    labels = []

    for n, d in G.nodes(data=True):

        if d.get("type") != "road":
            continue

        nodes.append(n)

        traffic = d.get("traffic", 0)
        capacity = d.get("capacity", 1)
        cap_ratio = traffic / capacity if capacity > 0 else 0

        features.append([
            traffic,
            d.get("rain_mm_mean", 0),
            d.get("flood_risk", 0),
            degree[n],
            nbr_risk[n],
            nbr_traffic[n],
            clustering[n],
            cap_ratio,
        ])

        labels.append(int(d.get("failed", False)))

    # Normalize features
    X = np.array(features, dtype=np.float32)

    # Replace bad values BEFORE scaling
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0

    # Clip extreme values
    X = np.clip(X, -1e6, 1e6)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Final safety
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0

    # Build PyTorch tensors
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Edge index
    node_map = {n: i for i, n in enumerate(nodes)}

    edges = []
    for u, v in G.edges():
        if u in node_map and v in node_map:
            edges.append([node_map[u], node_map[v]])
            edges.append([node_map[v], node_map[u]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def load_all_graphs():
    graphs = []
    for s in SCALES:
        graphs.append(build_graph(s))
    return graphs