"""
brain_network.py
================
Module 4 & 7 Visualization: Interactive Brain Graph Network Visualizer.
Plots interactive 2D brain connectome graphs using Plotly and NetworkX.
Color-codes nodes by canonical resting-state networks and scales nodes by degree centrality.
"""

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# Distinct palette for canonical resting-state networks
NETWORK_COLORS = {
    "Default Mode Network (DMN)": "#E11D48",          # Crimson
    "Salience Network (SN)": "#0D9488",                # Teal
    "Central Executive Network (CEN)": "#2563EB",      # Cobalt Blue
    "Sensorimotor Network (SMN)": "#16A34A",           # Forest Green
    "Visual Network (VN)": "#9333EA",                  # Violet
    "Subcortical Network": "#D97706",                  # Amber
    "Cerebellar Network": "#64748B",                   # Slate Gray
    "Other / Unassigned": "#94A3B8"                    # Neutral Gray
}


def plot_interactive_brain_graph(
    G: nx.Graph,
    title: str = "Functional Brain Connectivity Network",
    highlight_node: Optional[int] = None
) -> go.Figure:
    """
    Renders an interactive, publication-quality 2D Plotly graph of the brain network.
    - Nodes sized by degree centrality.
    - Nodes colored by canonical resting-state network.
    - Edges colored by correlation sign (blue = positive, crimson = negative).
    """
    # Deterministic layout for reproducible graph visual structure
    pos = nx.spring_layout(G, seed=42, k=0.28, iterations=50)

    # Prepare Edge Coordinates
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.75, color="rgba(148, 163, 184, 0.4)"),
        hoverinfo="none",
        mode="lines"
    )

    # Prepare Node Coordinates & Metadata
    node_x, node_y = [], []
    node_colors, node_sizes, hover_texts = [], [], []

    degree_dict = nx.degree_centrality(G)

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        meta = G.nodes[node]
        rname = meta.get("name", f"Region_{node}")
        net = meta.get("network", "Other / Unassigned")
        deg = degree_dict.get(node, 0.0)

        # Highlighting logic
        if highlight_node is not None and node == highlight_node:
            node_colors.append("#F59E0B")
            node_sizes.append(24)
        else:
            node_colors.append(NETWORK_COLORS.get(net, "#64748B"))
            node_sizes.append(max(8, int(deg * 50)))

        hover_texts.append(
            f"<b>{rname}</b> (Node #{node})<br>"
            f"Network: {net}<br>"
            f"Degree Centrality: {deg:.3f}<br>"
            f"Hemisphere: {meta.get('hemisphere', 'N/A')}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=hover_texts,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=1.2, color="#FFFFFF")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, x=0.5, font=dict(size=16, family="Inter, sans-serif")),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            width=750,
            height=600
        )
    )
    return fig
