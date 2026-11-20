"""
feature_extraction.py
=====================
Extracts and packages features for Traditional Machine Learning:
- Vectorized Functional Connectivity (Upper Triangular = 6,670 connections)
- Graph-Theoretical Topological Features (Degree, Clustering, Centrality = 348 features)
- Feature Selection via ANOVA F-statistic for high-dimensional neuroimaging data.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from connectivity import FunctionalConnectivity
from graph_builder import BrainGraphBuilder


class ConnectomeFeatureExtractor:
    """Extracts tabular edge and topological features from brain connectivity matrices."""

    def __init__(self, n_top_features: int = 200):
        self.n_top_features = n_top_features
        self.fc_extractor = FunctionalConnectivity()
        self.graph_builder = BrainGraphBuilder()
        self.feature_selector: Optional[SelectKBest] = None

    def extract_edge_features(self, matrices: np.ndarray) -> np.ndarray:
        """
        Extracts upper triangular vectorized correlation coefficients.
        Input: (n_subjects, 116, 116)
        Output: (n_subjects, 6670)
        """
        n_subs = matrices.shape[0]
        vectors = []
        for i in range(n_subs):
            vec = self.fc_extractor.vectorize_upper(matrices[i])
            vectors.append(vec)
        return np.array(vectors)

    def extract_graph_features(self, matrices: np.ndarray) -> np.ndarray:
        """
        Extracts nodal graph metrics (Degree Centrality, Clustering, Betweenness).
        Input: (n_subjects, 116, 116)
        Output: (n_subjects, 116 * 3 = 348)
        """
        n_subs = matrices.shape[0]
        topo_features = []

        for i in range(n_subs):
            G = self.graph_builder.build_networkx_graph(matrices[i])
            metrics = self.graph_builder.compute_graph_metrics(G)
            
            deg = np.array(list(metrics["degree_centrality"].values()))
            clust = np.array(list(metrics["clustering_coefficients"].values()))
            betw = np.array(list(metrics["betweenness_centrality"].values()))
            
            combined_nodal = np.concatenate([deg, clust, betw])
            topo_features.append(combined_nodal)

        return np.array(topo_features)

    def fit_transform_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, SelectKBest]:
        """
        Fits ANOVA F-test feature selector on training split to mitigate curse of dimensionality
        WITHOUT data leakage.
        """
        k = min(self.n_top_features, X_train.shape[1])
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X_train, y_train)
        return X_selected, self.feature_selector

    def transform_features(self, X_test: np.ndarray) -> np.ndarray:
        """Transforms test set using fitted selector."""
        if self.feature_selector is None:
            raise ValueError("Feature selector has not been fitted yet.")
        return self.feature_selector.transform(X_test)


if __name__ == "__main__":
    extractor = ConnectomeFeatureExtractor(n_top_features=100)
    dummy_mats = np.random.normal(0, 0.4, (10, 116, 116))
    dummy_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    edges = extractor.extract_edge_features(dummy_mats)
    print(f"[+] Extracted edge features shape: {edges.shape}")
    selected, _ = extractor.fit_transform_features(edges, dummy_labels)
    print(f"[+] Selected top features shape: {selected.shape}")
