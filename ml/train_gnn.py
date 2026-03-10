import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
import numpy as np

from ml.gnn_dataset import build_gnn_data


class GraphSAGE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, 64)
        self.conv2 = SAGEConv(64, 32)
        self.conv3 = SAGEConv(32, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return self.out(x)


def safe_normalize(x):
    mean = x.mean(0)
    std = x.std(0)

    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)


def train():

    print("Building graph dataset...")
    data = build_gnn_data()

    # ---- Clean + normalize ----
    data.x = torch.nan_to_num(data.x, nan=0.0, posinf=1.0, neginf=-1.0)
    data.x = safe_normalize(data.x)

    model = GraphSAGE(data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    y = data.y.float().view(-1, 1)

    print("Training GraphSAGE...")

    for epoch in range(100):

        model.train()
        logits = model(data)

        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss:", loss.item())

    # ---- Evaluation ----
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    probs = np.nan_to_num(probs)

    auc = roc_auc_score(data.y.cpu().numpy(), probs)

    print("\nGraphSAGE ROC AUC:", auc)


if __name__ == "__main__":
    train()