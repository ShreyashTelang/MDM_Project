"""
config.py
=========
Global Configuration for AI-Based Brain Connectivity Analysis Using GNNs.
Contains dataset specifications, atlas definitions (AAL-116), network hyper-parameters,
and directory path configurations.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Ensure all directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset Configurations
DATASET_NAME = "ABIDE_I"
DEFAULT_PIPELINE = "cpac"
DEFAULT_BAND_PASS = True
DEFAULT_GLOBAL_SIGNAL_REGRESSION = False
DEFAULT_DERIVATIVE = "rois_aal"  # Automated Anatomical Labeling (116 regions)

# Target Labels Mapping
LABEL_MAPPING = {
    1: 1,  # Autism Spectrum Disorder (ASD) -> 1
    2: 0   # Typically Developing Control (TC) -> 0
}
LABEL_NAMES = {
    1: "Autism Spectrum Disorder (ASD)",
    0: "Typically Developing Control (TC)"
}
CLASS_NAMES = ["Typical Control", "Autism Spectrum Disorder"]

# Preprocessing Parameters
FMR_TR = 2.0  # Repetition Time (s)
LOW_FREQ = 0.01  # Hz (high-pass filter threshold)
HIGH_FREQ = 0.10  # Hz (low-pass filter threshold for resting-state BOLD)
CONNECTIVITY_KIND = "correlation"  # "correlation", "partial correlation", "tangent"
FISHER_Z_TRANSFORM = True

# Graph Construction Parameters
GRAPH_THRESHOLD_PERCENTILE = 80.0  # Retain top 20% strongest absolute functional connections
GRAPH_THRESHOLD_TYPE = "proportional"  # "proportional", "absolute", "mst"
MIN_EDGE_WEIGHT = 0.20  # Minimum correlation magnitude if absolute thresholding

# Machine Learning Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# GNN Architecture Parameters
GNN_CONFIG = {
    "num_nodes": 116,
    "in_features": 116,       # Each node feature vector is its row of connectivity with all other nodes
    "hidden_dim_1": 64,
    "hidden_dim_2": 32,
    "dropout": 0.35,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "epochs": 50,
    "batch_size": 16,
    "pooling": "mean_max_concat"  # Concat of global_mean_pool and global_max_pool for expressive graph embeddings
}

# ------------------------------------------------------------------------------
# AAL (Automated Anatomical Labeling) 116 Region Definitions & Networks
# ------------------------------------------------------------------------------
AAL_REGIONS = [
    # Frontal Lobe
    "Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R", "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R", "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R", "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
    "Rolandic_Oper_L", "Rolandic_Oper_R", "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Olfactory_L", "Olfactory_R", "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L", "Frontal_Med_Orb_R", "Rectus_L", "Rectus_R",
    
    # Insular & Cingulate
    "Insula_L", "Insula_R", "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Cingulum_Mid_L", "Cingulum_Mid_R", "Cingulum_Post_L", "Cingulum_Post_R",
    
    # Limbic / Medial Temporal
    "Hippocampus_L", "Hippocampus_R", "ParaHippocampal_L", "ParaHippocampal_R",
    "Amygdala_L", "Amygdala_R",
    
    # Occipital Lobe
    "Calcarine_L", "Calcarine_R", "Cuneus_L", "Cuneus_R",
    "Lingual_L", "Lingual_R", "Occipital_Sup_L", "Occipital_Sup_R",
    "Occipital_Mid_L", "Occipital_Mid_R", "Occipital_Inf_L", "Occipital_Inf_R",
    
    # Parietal Lobe
    "Fusiform_L", "Fusiform_R", "Postcentral_L", "Postcentral_R",
    "Parietal_Sup_L", "Parietal_Sup_R", "Parietal_Inf_L", "Parietal_Inf_R",
    "Supramarginal_L", "Supramarginal_R", "Angular_L", "Angular_R",
    "Precuneus_L", "Precuneus_R", "Paracentral_Lobule_L", "Paracentral_Lobule_R",
    
    # Basal Ganglia & Subcortical
    "Caudate_L", "Caudate_R", "Putamen_L", "Putamen_R",
    "Pallidum_L", "Pallidum_R", "Thalamus_L", "Thalamus_R",
    "Heschl_L", "Heschl_R",
    
    # Temporal Lobe
    "Temporal_Sup_L", "Temporal_Sup_R", "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
    "Temporal_Mid_L", "Temporal_Mid_R", "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
    "Temporal_Inf_L", "Temporal_Inf_R",
    
    # Cerebellum & Vermis (91-116)
    "Cerebelum_Crus1_L", "Cerebelum_Crus1_R", "Cerebelum_Crus2_L", "Cerebelum_Crus2_R",
    "Cerebelum_3_L", "Cerebelum_3_R", "Cerebelum_4_5_L", "Cerebelum_4_5_R",
    "Cerebelum_6_L", "Cerebelum_6_R", "Cerebelum_7b_L", "Cerebelum_7b_R",
    "Cerebelum_8_L", "Cerebelum_8_R", "Cerebelum_9_L", "Cerebelum_9_R",
    "Cerebelum_10_L", "Cerebelum_10_R",
    "Vermis_1_2", "Vermis_3", "Vermis_4_5", "Vermis_6",
    "Vermis_7", "Vermis_8", "Vermis_9", "Vermis_10"
]

# Canonical Resting-State Brain Networks (RSNs) mapped to AAL indices
BRAIN_NETWORKS = {
    "Default Mode Network (DMN)": [
        "Precuneus_L", "Precuneus_R", "Cingulum_Post_L", "Cingulum_Post_R",
        "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R", "Angular_L", "Angular_R",
        "Hippocampus_L", "Hippocampus_R"
    ],
    "Salience Network (SN)": [
        "Insula_L", "Insula_R", "Cingulum_Ant_L", "Cingulum_Ant_R"
    ],
    "Central Executive Network (CEN)": [
        "Frontal_Mid_L", "Frontal_Mid_R", "Parietal_Inf_L", "Parietal_Inf_R",
        "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R"
    ],
    "Sensorimotor Network (SMN)": [
        "Precentral_L", "Precentral_R", "Postcentral_L", "Postcentral_R",
        "Supp_Motor_Area_L", "Supp_Motor_Area_R", "Paracentral_Lobule_L", "Paracentral_Lobule_R"
    ],
    "Visual Network (VN)": [
        "Calcarine_L", "Calcarine_R", "Cuneus_L", "Cuneus_R",
        "Lingual_L", "Lingual_R", "Occipital_Sup_L", "Occipital_Sup_R",
        "Occipital_Mid_L", "Occipital_Mid_R"
    ],
    "Subcortical Network": [
        "Thalamus_L", "Thalamus_R", "Caudate_L", "Caudate_R",
        "Putamen_L", "Putamen_R", "Pallidum_L", "Pallidum_R", "Amygdala_L", "Amygdala_R"
    ],
    "Cerebellar Network": [
        r for r in AAL_REGIONS if "Cerebelum" in r or "Vermis" in r
    ]
}
