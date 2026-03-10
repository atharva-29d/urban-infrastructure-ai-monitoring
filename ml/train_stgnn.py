import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from stgnn_dataset import build_sequences

SAMPLE = 8000


class STGNN(nn.Module):

    def __init__(self, in_dim=8, hidden=96):
        super().__init__()

        self.gnn1 = SAGEConv(in_dim, hidden)
        self.gnn2 = SAGEConv(hidden, hidden)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.gru = nn.GRU(hidden, hidden, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden, 1)

    def forward(self, data_seq, node_idx):

        h_seq = []

        for data in data_seq:

            edge_index, _ = subgraph(
                node_idx,
                data.edge_index,
                relabel_nodes=True
            )

            x = data.x[node_idx]

            h = self.gnn1(x, edge_index)
            h = torch.relu(self.norm1(h))
            h = self.dropout(h)

            h = self.gnn2(h, edge_index)
            h = torch.relu(self.norm2(h))

            h_seq.append(h.unsqueeze(0))

        h_seq = torch.cat(h_seq, dim=0)
        h_seq = h_seq.permute(1, 0, 2)

        out, _ = self.gru(h_seq)

        final = out[:, -1, :]

        logits = self.fc(final).squeeze()

        return logits


def train():

    print("Loading sequences...")
    seq = build_sequences()

    model = STGNN()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    pos = seq[0].y.sum()
    neg = len(seq[0].y) - pos

    pos_weight = neg / (pos + 1e-6)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight)
    )

    print("\nTraining EARLY cascade prediction (0.5 → 0.7)...")

    for epoch in range(25):

        model.train()

        node_idx = torch.randperm(seq[0].num_nodes)[:SAMPLE]

        # input = scale 0.5
        logits = model([seq[0]], node_idx)

        # target = scale 0.7
        y = seq[1].y[node_idx]

        loss = loss_fn(logits, y)

        if torch.isnan(loss):
            print("Skipping NaN batch")
            continue

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} Loss:", loss.item())

    print("\nEvaluating...")

    model.eval()

    node_idx = torch.randperm(seq[0].num_nodes)[:SAMPLE]

    logits = model([seq[0]], node_idx)

    # prevent numerical explosion
    logits = torch.clamp(logits, -10, 10)

    probs = torch.sigmoid(logits).detach().numpy()

    # remove invalid values
    probs = np.nan_to_num(probs)

    y_true = seq[1].y[node_idx].numpy()

    auc = roc_auc_score(y_true, probs)

    print("\n🔥 ST-GNN ROC-AUC:", auc)


if __name__ == "__main__":
    train()