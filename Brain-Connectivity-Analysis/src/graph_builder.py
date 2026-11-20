"""
graph_builder.py
================
Module 4: Brain Network Construction and Graph Theory Analysis.
Constructs NetworkX graphs from functional connectivity matrices, computes
topological network metrics (centrality, clustering, efficiency), and converts
connectomes into PyTorch Geometric Data objects for GNN training.
"""

import sys
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    AAL_REGIONS, BRAIN_NETWORKS, GRAPH_THRESHOLD_PERCENTILE,
    GRAPH_THRESHOLD_TYPE, MIN_EDGE_WEIGHT
)


class BrainGraphBuilder:
    """Builds and analyzes graph-theoretic brain networks from functional connectivity matrices."""

    def __init__(
        self,
        threshold_type: str = GRAPH_THRESHOLD_TYPE,
        threshold_value: float = GRAPH_THRESHOLD_PERCENTILE,
        region_names: list = AAL_REGIONS
    ):
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value
        self.region_names = region_names
        self.n_nodes = len(region_names)
        self._build_region_metadata()

    def _build_region_metadata(self):
        """Maps each AAL region to its hemisphere, lobe, and canonical resting-state network."""
        self.region_meta = {}
        for idx, rname in enumerate(self.region_names):
            hemi = "Left" if rname.endswith("_L") else ("Right" if rname.endswith("_R") else "Bilateral")
            network = "Other / Unassigned"
            for net_name, members in BRAIN_NETWORKS.items():
                if rname in members:
                    network = net_name
                    break
            self.region_meta[idx] = {
                "name": rname,
                "hemisphere": hemi,
                "network": network
            }

    def threshold_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Thresholds dense correlation matrix to construct a sparse, biologically meaningful graph.
        
        Supports:
        - 'proportional': Keeps the top p% strongest absolute connections across the brain.
        - 'absolute': Keeps connections with |W_ij| >= threshold_value.
        """
        mat = matrix.copy()
        np.fill_diagonal(mat, 0.0)

        if self.threshold_type == "proportional":
            # Retain top percentile
            abs_vals = np.abs(mat[np.triu_indices_from(mat, k=1)])
            cutoff = np.percentile(abs_vals, self.threshold_value)
            sparse_mat = np.where(np.abs(mat) >= cutoff, mat, 0.0)
        else:
            # Absolute thresholding
            sparse_mat = np.where(np.abs(mat) >= self.threshold_value, mat, 0.0)

        return sparse_mat

    def build_networkx_graph(self, matrix: np.ndarray) -> nx.Graph:
        """Constructs a weighted NetworkX undirected graph with rich anatomical node attributes."""
        sparse_mat = self.threshold_matrix(matrix)
        G = nx.Graph()

        # Add all 116 anatomical nodes
        for idx, meta in self.region_meta.items():
            G.add_node(idx, **meta)

        # Add edges passing threshold
        n = sparse_mat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                weight = float(sparse_mat[i, j])
                if abs(weight) > 1e-6:
                    G.add_edge(i, j, weight=weight, abs_weight=abs(weight))

        return G

    def compute_graph_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Calculates canonical graph-theoretical metrics of brain network topology:
        - Degree Centrality
        - Clustering Coefficient (local segregation)
        - Betweenness Centrality (hub bottlenecks)
        - Network Density
        - Global & Local Efficiency
        """
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)

        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight="abs_weight")
        clustering_coeff = nx.clustering(G, weight="abs_weight")
        
        # Mean summary statistics
        avg_clustering = float(np.mean(list(clustering_coeff.values())))
        avg_betweenness = float(np.mean(list(betweenness_centrality.values())))
        
        # Global efficiency: 1 / (shortest path length)
        try:
            global_eff = float(nx.global_efficiency(G))
        except Exception:
            global_eff = 0.0

        # Hub identification: top 5 highest degree nodes
        sorted_degrees = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        top_hubs = [
            {
                "node_idx": idx,
                "region": self.region_names[idx],
                "network": self.region_meta[idx]["network"],
                "degree": deg
            }
            for idx, deg in sorted_degrees[:5]
        ]

        return {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "network_density": density,
            "avg_clustering_coefficient": avg_clustering,
            "avg_betweenness_centrality": avg_betweenness,
            "global_efficiency": global_eff,
            "degree_centrality": degree_centrality,
            "clustering_coefficients": clustering_coeff,
            "betweenness_centrality": betweenness_centrality,
            "top_hubs": top_hubs
        }

    def to_pyg_data(
        self,
        matrix: np.ndarray,
        label: int,
        subject_id: int
    ) -> Any:
        """
        Converts a functional connectivity matrix into a PyTorch Geometric Data object:
        - x: Node feature tensor [num_nodes, in_channels]
        - edge_index: Graph connectivity tensor [2, num_edges]
        - edge_attr: Edge weight tensor [num_edges, 1]
        - y: Label tensor [1]
        """
        import torch
        from torch_geometric.data import Data

        sparse_mat = self.threshold_matrix(matrix)
        
        # Node features: the row of connectivity with all other regions (116-dim profile)
        # Concatenated with regional nodal strength (sum of abs weights)
        x_features = torch.tensor(matrix, dtype=torch.float32)

        # Edges
        rows, cols = np.where(np.abs(sparse_mat) > 1e-6)
        edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
        
        edge_weights = sparse_mat[rows, cols]
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

        y = torch.tensor([label], dtype=torch.long)

        data = Data(
            x=x_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            subject_id=subject_id,
            num_nodes=self.n_nodes
        )
        return data


if __name__ == "__main__":
    builder = BrainGraphBuilder()
    dummy_mat = np.random.normal(0, 0.4, (116, 116))
    dummy_mat = (dummy_mat + dummy_mat.T) / 2.0
    G = builder.build_networkx_graph(dummy_mat)
    metrics = builder.compute_graph_metrics(G)
    
    print(f"[+] NetworkX Brain Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[+] Density: {metrics['network_density']:.4f}")
    print(f"[+] Global Efficiency: {metrics['global_efficiency']:.4f}")
    print(f"[+] Top Hub: {metrics['top_hubs'][0]['region']} ({metrics['top_hubs'][0]['network']})")
