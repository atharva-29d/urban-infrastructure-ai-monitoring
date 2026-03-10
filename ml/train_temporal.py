import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "temporal_dataset.csv"


class TemporalNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Original size:", len(df))

    # 🔥 Only predict future failure for roads that are alive now
    df = df[df["failed_now"] == 0]

    print("\nAfter filtering alive roads:")
    print(df["failed_next"].value_counts())

    # features
    FEATURES = [
        "traffic",
        "length",
        "rain",
        "flood_risk",
        "scale",
        "step",
        "degree",
        "num_failed_neighbors",
        "neighbor_avg_risk",
        "neighbor_avg_traffic",
    ]

    X = df[FEATURES].values
    y = df["failed_next"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = TemporalNN(X_train.shape[1])

    # handle imbalance
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining...")

    for epoch in range(25):

        optimizer.zero_grad()

        logits = model(X_train)
        loss = loss_fn(logits, y_train)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} Loss:", loss.item())

    # evaluation
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy().flatten()

    auc = roc_auc_score(y_test, probs)

    print("\n🔥 GENERALIZATION ROC-AUC:", auc)


if __name__ == "__main__":
    main()