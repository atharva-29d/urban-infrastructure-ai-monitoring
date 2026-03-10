import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_after_cascade_weather_1.2.gpickle"
OUTPUT_DIR = BASE_DIR / "visualization"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return G


def main():

    print("Loading graph...")
    G = load_graph()

    print("Nodes:", len(G.nodes))
    print("Edges:", len(G.edges))

    failed_nodes = []
    working_nodes = []

    for n, d in G.nodes(data=True):
        if d.get("failed"):
            failed_nodes.append(n)
        else:
            working_nodes.append(n)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12,12))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=working_nodes,
        node_size=5,
        node_color="lightgray"
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=failed_nodes,
        node_size=5,
        node_color="red"
    )

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.1,
        width=0.2
    )

    plt.title("Cascade Failures in Pune Road Network")
    plt.axis("off")

    out = OUTPUT_DIR / "cascade_network.png"
    plt.savefig(out, dpi=300)

    print("Saved:", out)


if __name__ == "__main__":
    main()