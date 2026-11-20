"""
comparison.py
=============
Module 7: Healthy Control vs ASD Brain Connectivity Comparison and Statistics.
Performs two-sample independent t-tests with Benjamini-Hochberg FDR correction,
identifies significant hyper/hypo-connected brain pathways, and creates comparison charts.
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any

from config import AAL_REGIONS, BRAIN_NETWORKS


def compare_cohort_connectivity(
    matrices: np.ndarray,
    labels: np.ndarray,
    region_names: List[str] = AAL_REGIONS
) -> Dict[str, Any]:
    """
    Computes rigorous mass-univariate statistical comparison between ASD and TC cohorts.
    - Two-sample independent t-test for each pair of regions
    - Benjamini-Hochberg FDR correction for multiple comparisons
    - Top hypo-connected and hyper-connected edges in ASD
    """
    asd_mask = (labels == 1)
    tc_mask = (labels == 0)

    mats_asd = matrices[asd_mask]
    mats_tc = matrices[tc_mask]

    mean_asd = np.mean(mats_asd, axis=0)
    mean_tc = np.mean(mats_tc, axis=0)
    diff_matrix = mean_asd - mean_tc

    n_regions = matrices.shape[1]
    triu_i, triu_j = np.triu_indices(n_regions, k=1)

    t_stats = np.zeros((n_regions, n_regions))
    p_values = np.ones((n_regions, n_regions))

    p_list = []
    edges_info = []

    for i, j in zip(triu_i, triu_j):
        vals_asd = mats_asd[:, i, j]
        vals_tc = mats_tc[:, i, j]
        
        t_val, p_val = stats.ttest_ind(vals_asd, vals_tc, equal_var=False)
        if np.isnan(p_val):
            p_val = 1.0
            t_val = 0.0

        t_stats[i, j] = t_stats[j, i] = t_val
        p_values[i, j] = p_values[j, i] = p_val
        p_list.append(p_val)

        edges_info.append({
            "region_1": region_names[i],
            "region_2": region_names[j],
            "node_1": i,
            "node_2": j,
            "mean_tc": float(mean_tc[i, j]),
            "mean_asd": float(mean_asd[i, j]),
            "diff": float(diff_matrix[i, j]),
            "t_stat": float(t_val),
            "p_val": float(p_val)
        })

    # Benjamini-Hochberg FDR procedure
    sorted_indices = np.argsort(p_list)
    m = len(p_list)
    q_values = np.zeros(m)
    
    for rank, idx in enumerate(sorted_indices):
        q_val = p_list[idx] * m / (rank + 1)
        q_values[idx] = q_val

    # Ensure monotonicity of q-values
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    q_values = np.clip(q_values, 0.0, 1.0)

    for idx, q_val in enumerate(q_values):
        edges_info[idx]["fdr_q"] = float(q_val)

    df_edges = pd.DataFrame(edges_info)

    # Top Hypo-connected pathways in ASD (strong in TC, weaker in ASD: diff < 0)
    top_hypo = df_edges.sort_values(by="diff", ascending=True).head(10).to_dict(orient="records")

    # Top Hyper-connected pathways in ASD (weaker in TC, stronger in ASD: diff > 0)
    top_hyper = df_edges.sort_values(by="diff", ascending=False).head(10).to_dict(orient="records")

    return {
        "mean_tc": mean_tc,
        "mean_asd": mean_asd,
        "diff_matrix": diff_matrix,
        "t_stats": t_stats,
        "p_values": p_values,
        "df_edges": df_edges,
        "top_hypoconnected": top_hypo,
        "top_hyperconnected": top_hyper
    }


def plot_differential_heatmap(
    diff_matrix: np.ndarray,
    region_names: List[str] = AAL_REGIONS,
    title: str = "Differential Connectome: (ASD Mean - Control Mean)"
) -> go.Figure:
    """Plotly heatmap showing hyper-connectivity (red) and hypo-connectivity (blue) in ASD."""
    max_abs = max(0.1, float(np.max(np.abs(diff_matrix))))
    fig = go.Figure(data=go.Heatmap(
        z=diff_matrix,
        x=region_names,
        y=region_names,
        colorscale="RdBu_r",
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(title="Delta Fisher z", len=0.85),
        hovertemplate="Region 1: %{y}<br>Region 2: %{x}<br>Delta (ASD - TC): %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, family="Inter, sans-serif")),
        width=750,
        height=700,
        xaxis=dict(showticklabels=False, title="AAL Brain Regions (1–116)"),
        yaxis=dict(showticklabels=False, title="AAL Brain Regions (1–116)", autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white"
    )
    return fig


def plot_top_altered_pathways_bar(
    top_edges: List[Dict[str, Any]],
    title: str = "Top Significant Connection Differences"
) -> go.Figure:
    """Creates a horizontal bar chart of top differential pathways."""
    labels = [f"{e['region_1']} <-> {e['region_2']}" for e in top_edges]
    diffs = [e['diff'] for e in top_edges]
    colors = ["#2563EB" if d < 0 else "#DC2626" for d in diffs]

    fig = go.Figure(go.Bar(
        x=diffs,
        y=labels,
        orientation="h",
        marker=dict(color=colors)
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, family="Inter, sans-serif")),
        xaxis=dict(title="Connectivity Difference (ASD - TC)", zeroline=True, zerolinewidth=1.5),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=30, t=40, b=40),
        height=400,
        template="plotly_white"
    )
    return fig
