"""
Train the GCN graph autoencoder and save the learned node embeddings.

Usage:
    python src/train.py
"""

import numpy as np
import torch
from torch_geometric.data import Data

from build_graph import load_data, build_graph, build_user_features, build_product_features
from models import build_model


def build_pyg_data(G, user_df, product_df):
    """Convert the NetworkX graph + feature tables into a PyG Data object."""
    node_list = list(G.nodes())
    node_index = {node: i for i, node in enumerate(node_list)}

    feature_dim = 4
    X = np.zeros((len(node_list), feature_dim))

    user_lookup = user_df.set_index("UserId").to_dict("index")
    product_lookup = product_df.set_index("ProductId").to_dict("index")

    for node, idx in node_index.items():
        if node.startswith("user_"):
            uid = node.replace("user_", "")
            f = user_lookup.get(uid)
            if f:
                X[idx] = [f["review_count"], f["avg_score"],
                          f["time_span_days"], f["helpfulness_ratio"]]
        else:
            pid = node.replace("product_", "")
            f = product_lookup.get(pid)
            if f:
                X[idx] = [f["review_count"], f["avg_score"], f["unique_reviewers"], 0]

    edge_list = [(node_index[u], node_index[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # make bidirectional

    x = torch.tensor(X, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data, node_list


def train(data, epochs=100, lr=0.01):
    model = build_model(in_channels=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index).numpy()
    return model, embeddings


if __name__ == "__main__":
    df = load_data()
    G = build_graph(df)
    user_df = build_user_features(df)
    product_df = build_product_features(df)

    data, node_list = build_pyg_data(G, user_df, product_df)
    model, embeddings = train(data)

    np.save("results/embeddings.npy", embeddings)
    with open("results/node_list.txt", "w") as f:
        f.write("\n".join(node_list))

    torch.save(model.state_dict(), "results/gcn_weights.pt")
    print(f"Saved embeddings for {len(node_list)} nodes to results/")
