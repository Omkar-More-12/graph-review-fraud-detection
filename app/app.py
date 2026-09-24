"""
Interactive demo: visualize the review network with GCN-flagged
suspicious users highlighted, and inspect why any given user was flagged.

Run with:
    streamlit run app/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from build_graph import load_data, build_graph

st.set_page_config(page_title="Graph Review Fraud Detection", layout="wide")
st.title("Fake review detection — graph neural network")
st.caption(
    "A GCN learns reviewer/product network structure (not review text) to "
    "flag accounts with unusual connectivity patterns, validated against "
    "same-timestamp multi-product review bursts."
)


@st.cache_data
def load_everything():
    df = load_data()
    G = build_graph(df)
    embeddings = np.load("results/embeddings.npy")
    with open("results/node_list.txt") as f:
        node_list = f.read().splitlines()
    with open("results/flagged_users.txt") as f:
        flagged = set(f.read().splitlines())
    with open("results/burst_users.txt") as f:
        burst_users = set(f.read().splitlines())
    return df, G, node_list, flagged, burst_users


try:
    df, G, node_list, flagged, burst_users = load_everything()

    st.sidebar.header("View")
    n_nodes_to_show = st.sidebar.slider("Nodes to display", 50, 500, 200)

    top_priority = list(burst_users) + list(flagged - burst_users)
    node_subset = set(top_priority[:n_nodes_to_show // 2])
    remaining = [n for n in node_list if n not in node_subset]
    node_subset.update(np.random.choice(remaining, size=min(n_nodes_to_show, len(remaining)), replace=False))

    subG = G.subgraph(node_subset)
    pos = nx.spring_layout(subG, seed=42)

    edge_x, edge_y = [], []
    for src, dst in subG.edges():
        edge_x += [pos[src][0], pos[dst][0], None]
        edge_y += [pos[src][1], pos[dst][1], None]

    node_x = [pos[n][0] for n in subG.nodes()]
    node_y = [pos[n][1] for n in subG.nodes()]

    def color_for(n):
        if n in burst_users:
            return "#B03A2E"  # confirmed burst pattern
        elif n in flagged:
            return "#E67E22"  # flagged, unconfirmed
        elif n.startswith("user_"):
            return "#5B7FA6"
        return "#B4B2A9"

    node_color = [color_for(n) for n in subG.nodes()]
    node_text = [
        f"{n}<br>"
        + ("Confirmed burst pattern" if n in burst_users
           else "Flagged by GCN" if n in flagged
           else "Normal")
        for n in subG.nodes()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(width=0.5, color="#D8D6CC"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                              marker=dict(size=8, color=node_color),
                              text=node_text, hoverinfo="text"))
    fig.update_layout(showlegend=False, height=600,
                       xaxis=dict(visible=False), yaxis=dict(visible=False),
                       margin=dict(l=0, r=0, t=0, b=0))

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dark red = confirmed burst pattern · Orange = flagged, unconfirmed · Blue = normal user · Gray = product")

    st.subheader("Case study")
    if burst_users:
        example = list(burst_users)[0]
        uid = example.replace("user_", "")
        case = df[df["UserId"] == uid].sort_values("Time")
        st.write(f"**{example}** — {len(case)} reviews")
        st.dataframe(case[["ProductId", "Score", "Time", "HelpfulnessNumerator", "HelpfulnessDenominator"]])

except FileNotFoundError as e:
    st.warning(
        f"Missing file: {e}\n\nRun the pipeline first:\n\n"
        "```\npython src/train.py\npython src/anomaly_detection.py\n```"
    )
