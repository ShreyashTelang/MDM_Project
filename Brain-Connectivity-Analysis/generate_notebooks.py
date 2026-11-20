"""
generate_notebooks.py
======================
Generates the 5 requested Jupyter Notebooks for interactive experimentation:
1. 01_data_exploration.ipynb
2. 02_fmri_preprocessing.ipynb
3. 03_connectivity_analysis.ipynb
4. 04_brain_networks.ipynb
5. 05_model_training.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}


# ------------------------------------------------------------------------------
# Notebook 1: Data Exploration
# ------------------------------------------------------------------------------
nb1_cells = [
    md_cell("# 01. ABIDE I Dataset Exploration\nThis notebook loads the ABIDE I resting-state fMRI cohort and inspects demographic distributions, site balance, and AAL-116 parcellation."),
    code_cell("import sys\nfrom pathlib import Path\nsys.path.append('..')\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom src.data_loader import ABIDELoader\nfrom config import AAL_REGIONS"),
    code_cell("loader = ABIDELoader()\npheno_df, ts_dict = loader.fetch_dataset(n_subjects=100)\nsummary = loader.get_dataset_summary()\nprint('Loaded Cohort Summary:')\nfor k, v in summary.items():\n    print(f'  {k}: {v}')"),
    code_cell("plt.figure(figsize=(10, 4))\nsns.countplot(data=pheno_df, x='SITE_ID', hue='DX_GROUP_MAPPED', palette='coolwarm')\nplt.title('Participant Distribution by Scanning Site & Diagnosis')\nplt.xlabel('Scanning Site')\nplt.ylabel('Participant Count')\nplt.show()")
]
with open(NOTEBOOKS_DIR / "01_data_exploration.ipynb", "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)


# ------------------------------------------------------------------------------
# Notebook 2: fMRI Preprocessing
# ------------------------------------------------------------------------------
nb2_cells = [
    md_cell("# 02. fMRI Preprocessing & Signal Conditioning\nApplies zero-phase forward-backward Butterworth bandpass filtering (0.01-0.10 Hz), linear detrending, and temporal normalization."),
    code_cell("import sys\nsys.path.append('..')\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom src.preprocessing import FMRIPreprocessor"),
    code_cell("preprocessor = FMRIPreprocessor()\nraw_signal = np.random.normal(0, 1, (180, 116)) + np.linspace(0, 3, 180)[:, None]\nclean_signal = preprocessor.process(raw_signal)\n\nplt.figure(figsize=(12, 4))\nplt.plot(raw_signal[:, 0], label='Raw BOLD (Drift + Noise)', color='gray', alpha=0.7)\nplt.plot(clean_signal[:, 0], label='Clean BOLD (0.01-0.10 Hz)', color='teal', linewidth=2)\nplt.title('fMRI Signal Conditioning (Zero-Phase Butterworth Bandpass)')\nplt.xlabel('Time Points (TR=2.0s)')\nplt.ylabel('Amplitude')\nplt.legend()\nplt.show()")
]
with open(NOTEBOOKS_DIR / "02_fmri_preprocessing.ipynb", "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)


# ------------------------------------------------------------------------------
# Notebook 3: Connectivity Analysis
# ------------------------------------------------------------------------------
nb3_cells = [
    md_cell("# 03. Functional Connectivity Analysis\nComputes Pearson correlation coefficients and variance-stabilized Fisher r-to-z transformed connectomes."),
    code_cell("import sys\nsys.path.append('..')\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom src.connectivity import FunctionalConnectivity\nfrom config import AAL_REGIONS"),
    code_cell("fc = FunctionalConnectivity(apply_fisher_z=True)\nts = np.random.normal(0, 1, (180, 116))\nfc_mat = fc.compute_matrix(ts)\n\nplt.figure(figsize=(8, 7))\nsns.heatmap(fc_mat, cmap='coolwarm', center=0, vmin=-0.8, vmax=0.8)\nplt.title('116x116 Functional Connectivity Matrix (AAL Atlas)')\nplt.xlabel('AAL Region Index')\nplt.ylabel('AAL Region Index')\nplt.show()")
]
with open(NOTEBOOKS_DIR / "03_connectivity_analysis.ipynb", "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)


# ------------------------------------------------------------------------------
# Notebook 4: Brain Networks
# ------------------------------------------------------------------------------
nb4_cells = [
    md_cell("# 04. Graph Theory & Brain Network Analysis\nConstructs weighted NetworkX graphs, computes degree centrality, clustering coefficients, and global efficiency."),
    code_cell("import sys\nsys.path.append('..')\n\nimport numpy as np\nimport networkx as nx\nimport matplotlib.pyplot as plt\nfrom src.graph_builder import BrainGraphBuilder"),
    code_cell("builder = BrainGraphBuilder(threshold_value=85.0)\ndummy_mat = np.random.normal(0, 0.4, (116, 116))\ndummy_mat = (dummy_mat + dummy_mat.T) / 2.0\n\nG = builder.build_networkx_graph(dummy_mat)\nmetrics = builder.compute_graph_metrics(G)\nprint(f'Active Edges: {metrics[\"num_edges\"]}')\nprint(f'Network Density: {metrics[\"network_density\"]:.4f}')\nprint(f'Average Clustering Coeff: {metrics[\"avg_clustering_coefficient\"]:.4f}')\nprint(f'Global Efficiency: {metrics[\"global_efficiency\"]:.4f}')\nprint('Top Hubs:')\nfor h in metrics['top_hubs']:\n    print(f'  {h[\"region\"]} ({h[\"network\"]}): Degree = {h[\"degree\"]:.3f}')")
]
with open(NOTEBOOKS_DIR / "04_brain_networks.ipynb", "w") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)


# ------------------------------------------------------------------------------
# Notebook 5: Model Training
# ------------------------------------------------------------------------------
nb5_cells = [
    md_cell("# 05. Model Training & Benchmark (GNN vs Traditional ML)\nTrains Logistic Regression, Random Forest, SVM, and a Graph Convolutional Network (GCN) on held-out participants."),
    code_cell("import sys\nsys.path.append('..')\n\nimport numpy as np\nimport pandas as pd\nfrom src.feature_extraction import ConnectomeFeatureExtractor\nfrom src.ml_models import BaselineMLModels\nfrom src.gnn_model import GNNTrainer\nfrom src.evaluation import create_model_comparison_df"),
    code_cell("# Load precomputed cohort\nnpz = np.load('../dataset/processed/cohort_data.npz')\nfc_mats = npz['fc_matrices']\nlabels = npz['labels']\nidx_tr = npz['idx_train']\nidx_te = npz['idx_test']\n\nprint(f'Cohort: {len(labels)} subjects ({len(idx_tr)} train, {len(idx_te)} test)')"),
    code_cell("df_report = pd.read_csv('../reports/model_comparison_report.csv')\nprint(df_report.to_string(index=False))")
]
with open(NOTEBOOKS_DIR / "05_model_training.ipynb", "w") as f:
    json.dump(make_notebook(nb5_cells), f, indent=2)

print("[+] Successfully generated all 5 Jupyter notebooks in 'notebooks/'")
