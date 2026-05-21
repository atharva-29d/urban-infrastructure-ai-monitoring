import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CRITICAL_PATH = BASE_DIR / "data" / "graphs" / "critical_roads.csv"


def load_scores():
    return pd.read_csv(CRITICAL_PATH)


def greedy_select(K, alpha=0.5, beta=0.5):
    df = load_scores()

    required_cols = ["road_id", "degree_centrality", "flood_risk"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in critical_roads.csv: {col}")

    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    df["degree_norm"] = norm(df["degree_centrality"])
    df["flood_norm"] = norm(df["flood_risk"])
    traffic_weight = 0.0
    criticality_weight = 0.0

    if "traffic" in df.columns:
        df["traffic_norm"] = norm(df["traffic"])
        traffic_weight = 0.15
    else:
        df["traffic_norm"] = 0.0

    if "criticality" in df.columns:
        df["criticality_norm"] = norm(df["criticality"])
        criticality_weight = 0.35
    else:
        df["criticality_norm"] = 0.0

    base_weight = alpha + beta + traffic_weight + criticality_weight
    if base_weight == 0:
        base_weight = 1.0

    df["score"] = (
        alpha * df["degree_norm"]
        + beta * df["flood_norm"]
        + traffic_weight * df["traffic_norm"]
        + criticality_weight * df["criticality_norm"]
    )
    df["score"] = df["score"] / base_weight

    top = df.sort_values("score", ascending=False).head(K)

    return top["road_id"].tolist()



if __name__ == "__main__":

    roads = greedy_select(1000)

    print("Top 10 selected roads:")
    print(roads[:10])
