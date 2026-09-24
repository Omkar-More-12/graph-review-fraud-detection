"""
Baseline fraud-flagging methods, run BEFORE the GNN so there's an honest
point of comparison.

1. Rule-based: hand-written heuristic (burst timing + extreme rating + low helpfulness)
2. Louvain: unsupervised community detection on the graph structure
"""

import community as community_louvain


def rule_based_flag(user_df, min_reviews=5, max_span_days=3,
                     min_avg_score=4.5, max_helpfulness=0.3):
    """Flag users matching a simple, hand-written suspicious pattern."""
    suspicious = user_df[
        (user_df["review_count"] >= min_reviews) &
        (user_df["time_span_days"] <= max_span_days) &
        (user_df["avg_score"] >= min_avg_score) &
        (user_df["helpfulness_ratio"] < max_helpfulness)
    ]
    return suspicious


def louvain_communities(G):
    """Unsupervised community detection — no labels or training involved."""
    partition = community_louvain.best_partition(G)
    return partition


if __name__ == "__main__":
    from build_graph import load_data, build_graph, build_user_features

    df = load_data()
    G = build_graph(df)
    user_df = build_user_features(df)

    suspicious = rule_based_flag(user_df)
    print(f"Rule-based baseline flagged {len(suspicious)} users out of {len(user_df)}")

    partition = louvain_communities(G)
    print(f"Louvain found {len(set(partition.values()))} communities "
          f"across {len(partition)} nodes")
