"""
Build a bipartite reviewer-product graph from the raw Amazon Fine Food
Reviews CSV, and engineer per-user behavioral features.

Usage:
    from src.build_graph import build_graph, build_user_features
"""

import pandas as pd
import networkx as nx


def load_data(csv_path="data/Reviews.csv", sample_size=50000, random_state=42):
    """Load and lightly clean the raw reviews CSV."""
    df = pd.read_csv(csv_path)
    df = df[["UserId", "ProductId", "Score", "Time",
              "HelpfulnessNumerator", "HelpfulnessDenominator"]]
    df = df.dropna()
    df_sample = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df_sample


def build_graph(df):
    """Build a bipartite user-product graph: an edge per review."""
    G = nx.Graph()
    for _, row in df.iterrows():
        user_node = f"user_{row['UserId']}"
        product_node = f"product_{row['ProductId']}"
        G.add_node(user_node, type="user")
        G.add_node(product_node, type="product")
        G.add_edge(user_node, product_node, score=row["Score"], time=row["Time"])
    return G


def build_user_features(df):
    """Engineer per-user behavioral features: burst timing, rating extremity, etc."""
    rows = []
    for user_id, group in df.groupby("UserId"):
        review_count = len(group)
        unique_products = group["ProductId"].nunique()
        avg_score = group["Score"].mean()
        time_span_days = (group["Time"].max() - group["Time"].min()) / 86400
        helpfulness_ratio = (
            group["HelpfulnessNumerator"].sum()
            / max(group["HelpfulnessDenominator"].sum(), 1)
        )
        rows.append({
            "UserId": user_id,
            "review_count": review_count,
            "unique_products": unique_products,
            "avg_score": avg_score,
            "time_span_days": time_span_days,
            "helpfulness_ratio": helpfulness_ratio,
        })
    return pd.DataFrame(rows)


def build_product_features(df):
    """Engineer per-product features, used to build the GCN's node feature matrix."""
    rows = []
    for product_id, group in df.groupby("ProductId"):
        rows.append({
            "ProductId": product_id,
            "review_count": len(group),
            "avg_score": group["Score"].mean(),
            "unique_reviewers": group["UserId"].nunique(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_data()
    G = build_graph(df)
    user_df = build_user_features(df)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Users analyzed: {len(user_df)}")
