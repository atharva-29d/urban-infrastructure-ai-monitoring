import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
import numpy as np

from ml.gnn_dataset_multi import load_all_graphs


class GraphSAGE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, 64)
        self.conv2 = SAGEConv(64, 32)
        self.conv3 = SAGEConv(32, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return torch.clamp(x, -20, 20)


def train():

    graphs = load_all_graphs()

    # Choose test scenario automatically (middle)
    test_graph = graphs[2]
    train_graphs = [g for i, g in enumerate(graphs) if i != 2]

    print("\nClass balance in test graph:")
    print(np.bincount(test_graph.y.numpy()))

    model = GraphSAGE(test_graph.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    print("\nTraining on multiple disaster scenarios...")

    for epoch in range(100):

        model.train()
        total_loss = 0

        for g in train_graphs:
            optimizer.zero_grad()
            out = model(g.x, g.edge_index)
            loss = loss_fn(out, g.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {total_loss:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(test_graph.x, test_graph.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs[np.isnan(probs)] = 0.5

    auc = roc_auc_score(test_graph.y.numpy(), probs)
    print("\nGENERALIZATION ROC AUC:", auc)


if __name__ == "__main__":
    train()