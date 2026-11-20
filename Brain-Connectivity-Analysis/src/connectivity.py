"""
connectivity.py
===============
Module 3: Functional Connectivity Matrix Computation and Connectome Analysis.
Computes Pearson correlation matrices, applies Fisher r-to-z transformation,
extracts vectorized edge features, and analyzes regional connectivity strengths.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AAL_REGIONS, FISHER_Z_TRANSFORM


class FunctionalConnectivity:
    """Computes and analyzes whole-brain functional connectivity matrices."""

    def __init__(self, apply_fisher_z: bool = FISHER_Z_TRANSFORM):
        self.apply_fisher_z = apply_fisher_z

    def compute_matrix(
        self,
        timeseries: np.ndarray,
        zero_diagonal: bool = True
    ) -> np.ndarray:
        """
        Computes Pearson correlation matrix between all pairs of brain regions.
        
        Parameters:
            timeseries: Array of shape (timepoints, n_regions)
            zero_diagonal: If True, sets self-connections (diagonal) to 0.0.
            
        Returns:
            Symmetric connectivity matrix of shape (n_regions, n_regions)
        """
        # Pearson correlation: (n_regions, n_regions)
        corr = np.corrcoef(timeseries.T)
        
        # Handle numerical precision edge cases (NaNs or values slightly outside [-1, 1])
        corr = np.nan_to_num(corr, nan=0.0)
        corr = np.clip(corr, -0.999999, 0.999999)

        if self.apply_fisher_z:
            # Fisher r-to-z transform: z = arctanh(r)
            matrix = np.arctanh(corr)
        else:
            matrix = corr

        if zero_diagonal:
            np.fill_diagonal(matrix, 0.0)
        else:
            np.fill_diagonal(matrix, 1.0 if not self.apply_fisher_z else 0.0)

        return matrix

    def vectorize_upper(self, matrix: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Extracts the upper triangular elements of the connectivity matrix.
        For 116 AAL regions, yields exactly (116 * 115) / 2 = 6,670 unique connection features.
        """
        triu_indices = np.triu_indices_from(matrix, k=k)
        return matrix[triu_indices]

    def reconstruct_from_vector(self, vector: np.ndarray, n_nodes: int = len(AAL_REGIONS)) -> np.ndarray:
        """Reconstructs a symmetric n_nodes x n_nodes matrix from an upper-triangular vector."""
        matrix = np.zeros((n_nodes, n_nodes))
        triu_indices = np.triu_indices(n_nodes, k=1)
        matrix[triu_indices] = vector
        matrix = matrix + matrix.T
        return matrix

    def compute_nodal_strength(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculates total, positive, and negative connectivity strength for each brain region.
        
        Nodal Strength S_i = sum_j |W_ij|
        """
        pos_matrix = np.maximum(matrix, 0)
        neg_matrix = np.minimum(matrix, 0)

        total_strength = np.sum(np.abs(matrix), axis=1)
        pos_strength = np.sum(pos_matrix, axis=1)
        neg_strength = np.sum(np.abs(neg_matrix), axis=1)

        return {
            "total_strength": total_strength,
            "positive_strength": pos_strength,
            "negative_strength": neg_strength
        }

    def compute_group_averages(
        self,
        matrices: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes mean connectivity matrix for Healthy Control (TC) and ASD groups,
        as well as the differential connectivity matrix (ASD - TC).
        
        Parameters:
            matrices: Array of shape (n_subjects, n_regions, n_regions)
            labels: 1D array of subject labels (1 = ASD, 0 = TC)
            
        Returns:
            Tuple of (mean_tc, mean_asd, diff_matrix)
        """
        asd_mask = (labels == 1)
        tc_mask = (labels == 0)

        mean_asd = np.mean(matrices[asd_mask], axis=0)
        mean_tc = np.mean(matrices[tc_mask], axis=0)
        diff_matrix = mean_asd - mean_tc

        return mean_tc, mean_asd, diff_matrix


if __name__ == "__main__":
    fc = FunctionalConnectivity(apply_fisher_z=True)
    dummy_ts = np.random.normal(0, 1, (180, 116))
    mat = fc.compute_matrix(dummy_ts)
    vec = fc.vectorize_upper(mat)
    strengths = fc.compute_nodal_strength(mat)
    
    print(f"[+] Functional Connectivity Matrix: {mat.shape}")
    print(f"[+] Vectorized Upper Triangular Elements: {vec.shape[0]} (Expected 6,670)")
    print(f"[+] Mean Nodal Strength: {np.mean(strengths['total_strength']):.3f}")
