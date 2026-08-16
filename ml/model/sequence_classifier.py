"""
Echo Sequence Classifier Architecture Module
============================================

Implements the Echo recurrent sequence classifier model consuming temporal YAMNet 
1024-dimensional feature embeddings and outputting probabilities for the 4 Echo emergency classes.

Contract Specification:
- Input shape: [batch_size, sequence_length, 1024] (float32)
- Output shape: [batch_size, 4] (float32)
- Class index ordering:
    0: NORMAL
    1: ACCIDENT
    2: DISTRESS
    3: VIOLENT_INCIDENT
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_echo_sequence_classifier(sequence_length=6, embedding_dim=1024, num_classes=4):
    """
    Builds and compiles the Keras recurrent sequence classifier model.
    
    Args:
        sequence_length: Int or None for dynamic sequence length (e.g. 6)
        embedding_dim: Int, dimension of YAMNet embeddings (1024)
        num_classes: Int, number of target emergency categories (4)
        
    Returns:
        Compiled Keras tf.keras.Model instance
    """
    inputs = layers.Input(shape=(sequence_length, embedding_dim), name="yamnet_embeddings_input", dtype=tf.float32)
    
    # Dynamic sequence length masking (supports padded sequences)
    x = layers.Masking(mask_value=0.0, name="sequence_masking")(inputs)
    
    # Recurrent temporal sequence feature extraction
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=False, dropout=0.2, unroll=True),
        name="bidirectional_gru"
    )(x)
    
    # Dense projection layer with regularization
    x = layers.Dense(64, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    
    # Softmax output over the 4 Echo emergency classes
    outputs = layers.Dense(num_classes, activation="softmax", name="echo_class_probabilities")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="EchoSequenceClassifier")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    model = build_echo_sequence_classifier(sequence_length=6)
    model.summary()
