"""
connectivity_heatmap.py
=======================
Module 7 Visualization: Interactive and Static Functional Connectivity Heatmaps.
Generates Plotly interactive heatmaps with hover metadata and Matplotlib figures.
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_interactive_heatmap(
    matrix: np.ndarray,
    region_names: List[str],
    title: str = "Functional Connectivity Matrix (AAL-116)",
    colorscale: str = "RdBu_r",
    zmin: float = -0.8,
    zmax: float = 0.8
) -> go.Figure:
    """Creates a high-performance interactive Plotly heatmap with anatomical hover annotations."""
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=region_names,
        y=region_names,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title="Fisher z / r", len=0.85),
        hoverongaps=False,
        hovertemplate="Region 1: %{y}<br>Region 2: %{x}<br>Connectivity: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, family="Inter, sans-serif")),
        width=750,
        height=700,
        xaxis=dict(showticklabels=False, title="Brain Regions (AAL 1-116)"),
        yaxis=dict(showticklabels=False, title="Brain Regions (AAL 1-116)", autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white"
    )
    return fig


def plot_static_heatmap(
    matrix: np.ndarray,
    title: str = "Functional Connectivity Matrix",
    cmap: str = "coolwarm",
    vmin: float = -0.7,
    vmax: float = 0.7
) -> plt.Figure:
    """Creates a publication-quality Matplotlib heatmap figure."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    cax = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Fisher z / Pearson r")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("AAL Brain Regions (1–116)", fontsize=10)
    ax.set_ylabel("AAL Brain Regions (1–116)", fontsize=10)
    plt.tight_layout()
    return fig
