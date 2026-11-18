"""
train.py
========
Phase 5: 1D Convolutional Neural Network (1D CNN) Training Pipeline.

Architecture:
  Input (360, 1)
  → Conv1D(32, k=7) + BatchNorm + ReLU + MaxPool(2)
  → Conv1D(64, k=5) + BatchNorm + ReLU + MaxPool(2)
  → Conv1D(128, k=3) + BatchNorm + ReLU + MaxPool(2)
  → GlobalAveragePooling1D
  → Dense(64, ReLU) + Dropout(0.35)
  → Dense(4, Softmax)

Features:
- Handles class imbalance via cost-sensitive weighting
- Real-time training monitoring with EarlyStopping & ReduceLROnPlateau
- Best model serialization to models/ecg_cnn_model.keras
- Training & validation history visualization
- Inference configuration export to models/inference_config.json
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Set backend and environment variables before imports
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
from keras import layers, callbacks

CLASS_NAMES = ['Normal Beat', 'Supraventricular Ectopic (SVEB)', 'Premature Ventricular (PVC)', 'Other Arrhythmia']


def build_1d_cnn(input_shape=(360, 1), num_classes=4):
    """
    Constructs the 1D Convolutional Neural Network architecture.
    
    Layer-by-Layer Rationale:
    -------------------------
    1. InputLayer (360, 1):
       Accepts 1.0 second of ECG signal sampled at 360 Hz.
       
    2. Conv1D(32, kernel_size=7, padding='same'):
       7-sample kernel (~19.4 ms window) captures steep QRS inflection slopes
       and rapid high-frequency cardiac transitions.
       
    3. BatchNormalization:
       Zero-centers and normalizes internal feature distributions, stabilizing
       gradient propagation and accelerating convergence.
       
    4. MaxPooling1D(pool_size=2):
       Downsamples temporal sequence from 360 to 180 samples. Provides local
       temporal translation invariance.
       
    5. Conv1D(64, kernel_size=5, padding='same'):
       5-sample kernel on downsampled feature map covers ~27.8 ms effective
       receptive field. Assembles low-level edge features into sub-wave detectors
       (P-wave, QRS, T-wave).
       
    6. MaxPooling1D(pool_size=2):
       Downsamples temporal sequence from 180 to 90 samples.
       
    7. Conv1D(128, kernel_size=3, padding='same'):
       Captures wider multi-wave dependencies (PR interval length, ST segment
       deviations, T-wave discordance).
       
    8. MaxPooling1D(pool_size=2):
       Downsamples temporal sequence from 90 to 45 samples.
       
    9. GlobalAveragePooling1D:
       Averages the 45 temporal steps into a compact 128-dimensional embedding.
       Eliminates parameter explosion (preventing overfitting) and enforces
       global shift-invariance.
       
    10. Dense(64, activation='relu'):
        Dense projection layer for non-linear feature interaction.
        
    11. Dropout(0.35):
        Randomly drops 35% of neurons during training to prevent co-adaptation.
        
    12. Dense(4, activation='softmax'):
        Outputs normalized posterior probability distribution across all 4 classes:
        P(Normal), P(SVEB), P(PVC), P(Other), summing to 1.0.
    """
    inputs = keras.Input(shape=input_shape, name="ecg_input")
    
    # Block 1
    x = layers.Conv1D(filters=32, kernel_size=7, padding='same', use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool1")(x)  # Shape: (180, 32)
    
    # Block 2
    x = layers.Conv1D(filters=64, kernel_size=5, padding='same', use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool2")(x)  # Shape: (90, 64)
    
    # Block 3
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool3")(x)  # Shape: (45, 128)
    
    # Spatial Aggregation & Classification Head
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)  # Shape: (128,)
    x = layers.Dense(64, activation='relu', name="dense1")(x)
    x = layers.Dropout(0.35, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="softmax_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="1D_CNN_ECG_Classifier")
    return model


def train_model(data_dir='data/processed', models_dir='models', figures_dir='figures',
                batch_size=64, epochs=30, learning_rate=0.001):
    """
    Trains the 1D CNN model and serializes all artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 65)
    print(" PHASE 5: 1D CNN MODEL ARCHITECTURE & TRAINING PIPELINE ")
    print("=" * 65)
    
    # 1. Load Data
    print("[*] Loading processed tensors from 'data/processed/'...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    
    with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    raw_weights = metadata.get('class_weights', {})
    class_weights = {int(k): float(v) for k, v in raw_weights.items()}
    
    print(f"    • Training Batches   : {len(X_train)} samples (shape: {X_train.shape})")
    print(f"    • Validation Batches : {len(X_val)} samples (shape: {X_val.shape})")
    print(f"    • Batch Size         : {batch_size}")
    print(f"    • Class Weights      : {class_weights}")
    
    # 2. Build Model
    model = build_1d_cnn(input_shape=(360, 1), num_classes=4)
    model.summary()
    
    # 3. Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # 4. Callbacks
    model_checkpoint_path = os.path.join(models_dir, "ecg_cnn_model.keras")
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )
    
    # 5. Fit
    print(f"\n[*] Starting 1D CNN Model Training (up to {epochs} epochs)...", flush=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
        verbose=2
    )
    
    print(f"\n[+] Model training finished! Best checkpoint saved to: {model_checkpoint_path}")
    
    # 6. Save Training History
    hist_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'lr': [float(x) for x in history.history.get('lr', history.history.get('learning_rate', []))]
    }
    with open(os.path.join(models_dir, "training_history.json"), 'w') as f:
        json.dump(hist_dict, f, indent=4)
        
    # 7. Save Inference Configuration
    inference_config = {
        'model_file': 'models/ecg_cnn_model.keras',
        'sampling_rate': 360,
        'window_size': 360,
        'window_before': 144,
        'window_after': 216,
        'lead_used': 'MLII',
        'preprocessing': {
            'lowcut_hz': 0.5,
            'highcut_hz': 45.0,
            'notch_hz': 60.0,
            'normalization': 'zscore'
        },
        'classes': CLASS_NAMES,
        'class_descriptions': {
            'Normal Beat': 'Normal sinus rhythm; no abnormal conduction pattern detected.',
            'Supraventricular Ectopic (SVEB)': 'Atrial/junctional premature contraction originating above ventricles.',
            'Premature Ventricular (PVC)': 'Ventricular ectopic beat originating in ventricles (wide QRS).',
            'Other Arrhythmia': 'Ventricular fusion, paced rhythm, or unclassified cardiac pattern.'
        }
    }
    with open(os.path.join(models_dir, "inference_config.json"), 'w') as f:
        json.dump(inference_config, f, indent=4)
    print(f"[+] Saved inference configuration to: {os.path.join(models_dir, 'inference_config.json')}")
    
    # 8. Plot Training History Curves
    plot_training_curves(history.history, figures_dir=figures_dir)
    
    return model, history


def plot_training_curves(hist, figures_dir='figures'):
    """
    Plots training vs validation accuracy and loss progression curves.
    """
    epochs_range = range(1, len(hist['loss']) + 1)
    
    # Plot Loss Curve
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, hist['loss'], 'b-o', lw=2.0, label='Training Loss')
    plt.plot(epochs_range, hist['val_loss'], 'r-s', lw=2.0, label='Validation Loss')
    plt.title("1D CNN Training & Validation Loss Progression", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Sparse Categorical Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    loss_path = os.path.join(figures_dir, "training_validation_loss.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"[+] Loss progression plot saved to: {loss_path}")
    
    # Plot Accuracy Curve
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, [acc * 100 for acc in hist['accuracy']], 'b-o', lw=2.0, label='Training Accuracy')
    plt.plot(epochs_range, [acc * 100 for acc in hist['val_accuracy']], 'g-s', lw=2.0, label='Validation Accuracy')
    plt.title("1D CNN Training & Validation Accuracy Progression", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    acc_path = os.path.join(figures_dir, "training_validation_accuracy.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()
    print(f"[+] Accuracy progression plot saved to: {acc_path}")


if __name__ == '__main__':
    train_model(batch_size=64, epochs=25, learning_rate=0.001)
