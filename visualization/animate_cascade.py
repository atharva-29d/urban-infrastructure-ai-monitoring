import pickle
import networkx as nx
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_after_cascade_weather_1.2.gpickle"
OUT_DIR = BASE_DIR / "visualization" / "cascade_animation"
OUT_DIR.mkdir(exist_ok=True)


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return G


def simulate_cascade(G, steps=10):

    history = []

    failed = set(n for n, d in G.nodes(data=True) if d.get("failed"))

    for step in range(steps):

        history.append(set(failed))

        new_failed = set()

        for n in G.nodes():

            if n in failed:
                continue

            neighbors = list(G.neighbors(n))

            failed_neighbors = sum(1 for nbr in neighbors if nbr in failed)

            if failed_neighbors > 1:
                new_failed.add(n)

        failed = failed.union(new_failed)

    return history


def main():

    print("Loading graph...")
    G = load_graph()

    print("Computing layout...")
    pos = nx.spring_layout(G, seed=42)

    print("Simulating cascade...")
    history = simulate_cascade(G, steps=8)

    frames = []

    for step, failed in enumerate(history):

        print("Rendering frame", step)

        plt.figure(figsize=(10,10))

        working = [n for n in G.nodes() if n not in failed]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=working,
            node_size=5,
            node_color="lightgray"
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(failed),
            node_size=5,
            node_color="red"
        )

        nx.draw_networkx_edges(
            G,
            pos,
            width=0.2,
            alpha=0.1
        )

        plt.title(f"Cascade Step {step}")
        plt.axis("off")

        frame_path = OUT_DIR / f"frame_{step}.png"
        plt.savefig(frame_path, dpi=200)
        plt.close()

        frames.append(imageio.imread(frame_path))

    gif_path = OUT_DIR / "cascade_animation.gif"
    imageio.mimsave(gif_path, frames, duration=1)

    print("Saved animation:", gif_path)


if __name__ == "__main__":
    main()