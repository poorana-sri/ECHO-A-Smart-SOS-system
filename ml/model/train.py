"""
Training Pipeline Script for ECHO Sequence Classifier
=====================================================

Loads preprocessed YAMNet sequence datasets (train_data.npz, val_data.npz), 
configures class weights, trains the recurrent Echo Sequence Classifier model, 
and exports the trained model weights.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Include parent paths for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.data.prepare_dataset import create_dataset
from ml.model.sequence_classifier import build_echo_sequence_classifier

def train_model(data_dir="ml/data/splits", model_output_path="ml/model/saved_model.keras", epochs=20, batch_size=32):
    """
    Executes training loop for the Echo Sequence Classifier.
    """
    train_path = os.path.join(data_dir, "train_data.npz")
    val_path = os.path.join(data_dir, "val_data.npz")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[train] Dataset splits not found. Generating dataset...")
        create_dataset(output_dir=data_dir)
        
    print(f"[train] Loading dataset splits from {data_dir}...")
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    
    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    
    print(f"  - Training samples: {X_train.shape[0]} (Feature shape: {X_train.shape[1:]})")
    print(f"  - Validation samples: {X_val.shape[0]}")
    
    # Compute balanced class weights to handle any class imbalance
    classes = np.unique(y_train)
    total_samples = len(y_train)
    class_counts = np.bincount(y_train)
    class_weights = {i: float(total_samples / (len(classes) * class_counts[i])) for i in classes}
    print("[train] Computed Class Weights:", class_weights)
    
    sequence_length = X_train.shape[1]
    embedding_dim = X_train.shape[2]
    
    model = build_echo_sequence_classifier(
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        num_classes=4
    )
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]
    
    print(f"[train] Training Echo Sequence Classifier for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(model_output_path)
    print(f"[train] Model successfully saved to {model_output_path}")
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[train] Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc * 100:.2f}%")
    
    return model, history

if __name__ == "__main__":
    train_model()
