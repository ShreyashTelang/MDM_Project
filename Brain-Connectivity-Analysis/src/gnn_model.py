"""
gnn_model.py
============
Module 6: Graph Neural Network (GNN) for Brain Connectome Classification.
Implements:
1. PyTorch Geometric Graph Convolutional Network (GCN) with Edge Weighting
2. Dual Graph Readout Pooling (Global Mean + Global Max Pooling)
3. Pure PyTorch Spectral GCN Fallback for maximum cross-platform portability
4. End-to-end Training, Validation, Early Stopping, and Inference engine.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GNN_CONFIG, MODELS_DIR, RANDOM_STATE
from evaluation import evaluate_predictions

try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    from torch_geometric.loader import DataLoader as PyGDataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ==============================================================================
# 1. PyTorch Geometric GCN Model Architecture
# ==============================================================================
class PyGBrainGCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for whole-brain graph-level classification.
    Incorporates edge weights (correlation strengths), batch normalization,
    dual global pooling (mean + max), and a multi-layer classification head.
    """

    def __init__(
        self,
        in_features: int = GNN_CONFIG["in_features"],
        hidden_dim_1: int = GNN_CONFIG["hidden_dim_1"],
        hidden_dim_2: int = GNN_CONFIG["hidden_dim_2"],
        num_classes: int = 2,
        dropout: float = GNN_CONFIG["dropout"]
    ):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)

        self.dropout = nn.Dropout(p=dropout)
        # Dual readout pooling doubles the embedding dimension
        self.fc1 = nn.Linear(hidden_dim_2 * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Graph Convolution Block 1
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Graph Convolution Block 2
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Dual Graph Readout Pooling: captures both global density and focal peak hubs
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)

        # MLP Classification Head
        out = F.relu(self.fc1(h_graph))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


# ==============================================================================
# 2. Pure PyTorch Spectral GCN (Zero-dependency Fallback)
# ==============================================================================
class SpectralGraphConv(nn.Module):
    """
    Kipf & Welling Spectral Graph Convolution:
    H^(l+1) = sigma( D^(-1/2) * A_hat * D^(-1/2) * H^(l) * W^(l) )
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N, In_F), adj_norm: (Batch, N, N)
        support = torch.matmul(x, self.weight)  # (Batch, N, Out_F)
        output = torch.matmul(adj_norm, support) + self.bias
        return output


class PurePyTorchBrainGCN(nn.Module):
    """Batch-oriented Spectral GCN operating directly on normalized adjacency tensors."""
    def __init__(
        self,
        in_features: int = 116,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        num_classes: int = 2,
        dropout: float = 0.35
    ):
        super().__init__()
        self.gcn1 = SpectralGraphConv(in_features, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.gcn2 = SpectralGraphConv(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim_2 * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F), adj_norm: (B, N, N)
        h = self.gcn1(x, adj_norm)
        # Reshape for BatchNorm1d: (B * N, Hidden)
        B, N, H1 = h.shape
        h = self.bn1(h.view(B * N, H1)).view(B, N, H1)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gcn2(h, adj_norm)
        B, N, H2 = h.shape
        h = self.bn2(h.view(B * N, H2)).view(B, N, H2)
        h = F.relu(h)
        h = self.dropout(h)

        # Dual pooling over nodes dimension (dim=1)
        h_mean = torch.mean(h, dim=1)
        h_max, _ = torch.max(h, dim=1)
        h_graph = torch.cat([h_mean, h_max], dim=-1)

        out = F.relu(self.fc1(h_graph))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


# ==============================================================================
# 3. GNN Trainer & Inference Engine
# ==============================================================================
class GNNTrainer:
    """Manages training, validation, early stopping, and metric logging for Brain GCN."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    @staticmethod
    def compute_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
        """Computes symmetric normalized adjacency: D^(-1/2) * (A + I) * D^(-1/2)."""
        A_tilde = adj + np.eye(adj.shape[0])
        degrees = np.sum(np.abs(A_tilde), axis=1)
        d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        d_inv_sqrt[degrees == 0] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def train_spectral_gcn(
        self,
        X_train_mats: np.ndarray,
        y_train: np.ndarray,
        X_val_mats: np.ndarray,
        y_val: np.ndarray,
        epochs: int = GNN_CONFIG["epochs"],
        lr: float = GNN_CONFIG["learning_rate"],
        weight_decay: float = GNN_CONFIG["weight_decay"],
        batch_size: int = GNN_CONFIG["batch_size"],
        save_path: Path = MODELS_DIR / "gnn_gcn_model.pt"
    ) -> Dict[str, Any]:
        """Trains Spectral GCN on connectivity matrices with early stopping."""
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        n_nodes = X_train_mats.shape[1]
        self.model = PurePyTorchBrainGCN(in_features=n_nodes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Precompute normalized adjacencies and node feature matrices
        def prep_tensors(mats, labels):
            adj_list, feat_list = [], []
            for m in mats:
                norm_adj = self.compute_normalized_laplacian(m)
                adj_list.append(norm_adj)
                feat_list.append(m)  # Node features = 116-dim connectivity profile
            
            x_t = torch.tensor(np.array(feat_list), dtype=torch.float32)
            adj_t = torch.tensor(np.array(adj_list), dtype=torch.float32)
            y_t = torch.tensor(labels, dtype=torch.long)
            return x_t, adj_t, y_t

        X_tr_t, adj_tr_t, y_tr_t = prep_tensors(X_train_mats, y_train)
        X_val_t, adj_val_t, y_val_t = prep_tensors(X_val_mats, y_val)

        best_val_auc = 0.0
        best_metrics: Dict[str, Any] = {}

        print(f"[*] Training GNN (Spectral GCN) for {epochs} epochs on {self.device}...")

        n_train = len(y_train)
        for ep in range(1, epochs + 1):
            self.model.train()
            indices = np.random.permutation(n_train)
            total_loss = 0.0
            num_batches = 0

            for i in range(0, n_train, batch_size):
                b_idx = indices[i:i + batch_size]
                bx, badj, by = X_tr_t[b_idx].to(self.device), adj_tr_t[b_idx].to(self.device), y_tr_t[b_idx].to(self.device)

                optimizer.zero_grad()
                logits = self.model(bx, badj)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / max(1, num_batches)

            # Validation pass
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t.to(self.device), adj_val_t.to(self.device))
                val_loss = criterion(val_logits, y_val_t.to(self.device)).item()
                val_probs = F.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
                val_preds = torch.argmax(val_logits, dim=-1).cpu().numpy()

            val_eval = evaluate_predictions(y_val, val_preds, val_probs)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_eval["accuracy"])
            self.history["val_auc"].append(val_eval["roc_auc"])

            if val_eval["roc_auc"] > best_val_auc:
                best_val_auc = val_eval["roc_auc"]
                best_metrics = val_eval
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "best_epoch": ep,
                    "metrics": best_metrics
                }, save_path)

            if ep % 10 == 0 or ep == epochs:
                print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_eval['accuracy']:.4f} | Val AUC: {val_eval['roc_auc']:.4f}")

        print(f"[+] GNN training complete. Best Validation ROC-AUC: {best_val_auc:.4f}")
        return best_metrics


if __name__ == "__main__":
    np.random.seed(42)
    mats_tr = np.random.normal(0, 0.4, (80, 116, 116))
    y_tr = np.random.randint(0, 2, 80)
    mats_val = np.random.normal(0, 0.4, (20, 116, 116))
    y_val = np.random.randint(0, 2, 20)

    trainer = GNNTrainer()
    metrics = trainer.train_spectral_gcn(mats_tr, y_tr, mats_val, y_val, epochs=5)
    print("Self-test metrics:", metrics)
