# ml/stgnn_dataset.py

import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_DIR = BASE_DIR / "data" / "graphs"

SCALES = [0.5, 0.7, 0.9]


def load_graph(scale):
    path = GRAPH_DIR / f"graph_scale_{scale}.gpickle"
    print("Loading:", path.name)

    with open(path, "rb") as f:
        G = pickle.load(f)

    return G


def neighbor_stats(G, node):

    nbrs = list(G.neighbors(node))

    if len(nbrs) == 0:
        return 0, 0, 0

    risks = []
    traff = []
    failed = 0

    for n in nbrs:
        d = G.nodes[n]

        risks.append(d.get("flood_risk", 0))
        traff.append(d.get("traffic", 0))

        if d.get("failed", False):
            failed += 1

    return (
        np.mean(risks),
        np.mean(traff),
        failed
    )


def build_graph_pair(prev_G, curr_G):

    node_map = {}
    i = 0

    for n, d in curr_G.nodes(data=True):
        if d.get("type") == "road":
            node_map[n] = i
            i += 1

    X = []
    y = []

    for n, d in curr_G.nodes(data=True):

        if d.get("type") != "road":
            continue

        prev_d = prev_G.nodes[n]

        nbr_risk, nbr_traffic, failed_nbrs = neighbor_stats(curr_G, n)

        X.append([
            d.get("traffic", 0),
            d.get("length", 0),
            d.get("rain_mm_mean", 0),
            d.get("flood_risk", 0),
            curr_G.degree(n),
            nbr_risk,
            nbr_traffic,
            failed_nbrs
        ])

        prev_failed = prev_d.get("failed", False)
        curr_failed = d.get("failed", False)

        y.append(int(curr_failed and not prev_failed))

    X = np.array(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = np.nan_to_num(X)

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    edges = []

    for u, v in curr_G.edges():
        if u in node_map and v in node_map:
            edges.append([node_map[u], node_map[v]])
            edges.append([node_map[v], node_map[u]])

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    data = Data(
        x=X,
        edge_index=edge_index,
        y=y
    )

    return data


def build_sequences():

    graphs = [load_graph(s) for s in SCALES]

    seq = []

    for i in range(len(graphs) - 1):
        data = build_graph_pair(graphs[i], graphs[i+1])
        seq.append(data)

    print("Total sequences:", len(seq))
    print(seq[0])

    return seq


if __name__ == "__main__":
    build_sequences()