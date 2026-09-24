"""
GCN encoder, trained as a graph autoencoder (GAE) since there are no
fraud labels available. The model learns to compress each node into an
embedding that can reconstruct the graph's edges — nodes with unusual
connectivity patterns end up with unusual embeddings, which is what the
downstream anomaly detector (see anomaly_detection.py) picks up on.
"""

import torch
from torch_geometric.nn import GCNConv, GAE


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def build_model(in_channels, hidden_channels=32, out_channels=16):
    encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
    return GAE(encoder)
