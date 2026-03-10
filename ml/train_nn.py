import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "ml_dataset.csv"


class FailureNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    features = [
        "traffic",
        "length",
        "rain",
        "flood_risk",
        "degree",
        "nbr_risk",
        "nbr_traffic",
        "failed_nbrs",
        "scale",
    ]

    X = df[features].values
    y = df["failed"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = FailureNet(len(features))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Training neural network...")

    for epoch in range(30):
        model.train()
        logits = model(X_train)

        loss = loss_fn(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} Loss:", loss.item())

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).numpy().flatten()

    auc = roc_auc_score(y_test.numpy(), probs)
    print("\nNN ROC AUC:", auc)


if __name__ == "__main__":
    main()