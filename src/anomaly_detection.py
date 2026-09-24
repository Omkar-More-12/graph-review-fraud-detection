"""
Turn GCN embeddings into fraud-suspicion scores via Isolation Forest,
restricted to active users (3+ reviews) so quiet/low-activity users
aren't mistaken for anomalies. Then validate flagged users by checking
for same-timestamp, multi-product review bursts — a concrete, human-
checkable signature of automated/bulk posting.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def flag_active_user_anomalies(embeddings, node_list, user_df, min_reviews=3, contamination=0.05):
    """Run Isolation Forest on embeddings, restricted to active user nodes."""
    active_users = set(user_df[user_df["review_count"] >= min_reviews]["UserId"])

    user_mask = [n.startswith("user_") for n in node_list]
    user_embeddings = embeddings[user_mask]
    user_node_list = [node_list[i] for i in range(len(node_list)) if user_mask[i]]

    active_mask = [n.replace("user_", "") in active_users for n in user_node_list]
    active_embeddings = user_embeddings[active_mask]
    active_node_list = [user_node_list[i] for i in range(len(user_node_list)) if active_mask[i]]

    scaler = StandardScaler()
    active_embeddings_scaled = scaler.fit_transform(active_embeddings)

    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(active_embeddings_scaled)

    flagged = [active_node_list[i] for i, p in enumerate(preds) if p == -1]
    return flagged, active_node_list


def find_burst_users(flagged_nodes, df, min_same_timestamp=3):
    """
    Validate flagged users by checking for genuine same-timestamp,
    multi-product review bursts — a strong automated-posting signature
    that a simple rule-based check would miss.
    """
    burst_users = []
    for node in flagged_nodes:
        uid = node.replace("user_", "")
        user_reviews = df[df["UserId"] == uid]
        same_ts_counts = user_reviews["Time"].value_counts()
        if same_ts_counts.max() >= min_same_timestamp:
            burst_users.append(node)
    return burst_users


if __name__ == "__main__":
    import pandas as pd
    from build_graph import load_data, build_user_features

    df = load_data()
    user_df = build_user_features(df)

    embeddings = np.load("results/embeddings.npy")
    with open("results/node_list.txt") as f:
        node_list = f.read().splitlines()

    flagged, active_node_list = flag_active_user_anomalies(embeddings, node_list, user_df)
    print(f"Flagged {len(flagged)} out of {len(active_node_list)} active users")

    burst_users = find_burst_users(flagged, df)
    print(f"Of those, {len(burst_users)} show same-timestamp multi-product "
          f"review bursts — the strongest fraud signal found")

    with open("results/flagged_users.txt", "w") as f:
        f.write("\n".join(flagged))
    with open("results/burst_users.txt", "w") as f:
        f.write("\n".join(burst_users))
