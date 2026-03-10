import pickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

GRAPH_PATH = BASE_DIR / "data" / "graphs" / "pune_base_graph_weather.gpickle"
CRITICAL_PATH = BASE_DIR / "data" / "graphs" / "critical_roads.csv"


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def load_scores():
    return pd.read_csv(CRITICAL_PATH)


def greedy_select(K, alpha=0.5, beta=0.5):
    df = load_scores()

    # -------------------------------
    # Ensure required columns exist
    # -------------------------------
    for col in ["degree_centrality", "flood_risk"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in critical_roads.csv: {col}")

    # -------------------------------
    # Normalize to [0,1]
    # -------------------------------
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    df["degree_norm"] = norm(df["degree_centrality"])
    df["flood_norm"] = norm(df["flood_risk"])

    # -------------------------------
    # Composite greedy score
    # -------------------------------
    df["score"] = (
        alpha * df["degree_norm"]
        + beta * df["flood_norm"]
    )

    top = df.sort_values("score", ascending=False).head(K)

    return top["road_id"].tolist()



if __name__ == "__main__":

    roads = greedy_select(1000)

    print("Top 10 selected roads:")
    print(roads[:10])
