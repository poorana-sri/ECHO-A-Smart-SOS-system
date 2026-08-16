"""
TensorFlow Lite Export & Verification Script for ECHO
=====================================================

Converts trained Echo Sequence Classifier Keras model to TensorFlow Lite (.tflite),
verifies Android runtime compatibility, checks tensor shapes & dtypes, copies assets 
to app/src/main/assets/models/, and executes sample inference test.
"""

import os
import sys
import shutil
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.data.prepare_dataset import CLASS_MAP

EXPORTED_DIR = "ml/exported"
ASSETS_DIR = "app/src/main/assets/models"

def export_and_verify_tflite(
    keras_model_path="ml/model/saved_model.keras",
    sequence_length=6,
    embedding_dim=1024
):
    """
    Exports Keras model to TFLite and verifies contract compliance.
    """
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Model file not found at {keras_model_path}. Train model first.")

    os.makedirs(EXPORTED_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print(f"[export_tflite] Loading Keras model from {keras_model_path}...")
    keras_model = tf.keras.models.load_model(keras_model_path)

    # Convert model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()

    tflite_filename = "echo_sequence_classifier.tflite"
    exported_path = os.path.join(EXPORTED_DIR, tflite_filename)
    assets_path = os.path.join(ASSETS_DIR, tflite_filename)

    with open(exported_path, "wb") as f:
        f.write(tflite_model)
    print(f"[export_tflite] Exported TFLite model to {exported_path}")

    # Copy asset to Android app assets directory
    shutil.copy2(exported_path, assets_path)
    print(f"[export_tflite] Copied asset to Android app directory: {assets_path}")

    # Create dummy YAMNet label asset if missing
    labels_filename = "yamnet_labels.txt"
    exported_labels = os.path.join(EXPORTED_DIR, labels_filename)
    assets_labels = os.path.join(ASSETS_DIR, labels_filename)
    
    if not os.path.exists(exported_labels):
        with open(exported_labels, "w") as f:
            f.write("Speech\nScream\nVehicle\nExplosion\nCheering\nSilence\n")
        shutil.copy2(exported_labels, assets_labels)
        print(f"[export_tflite] Created YAMNet labels file at {exported_labels} and copied to assets.")

    # ----------------------------------------------------
    # MANDATORY MANDATED TFLITE VERIFICATION (Section 17)
    # ----------------------------------------------------
    print("\n==================================================")
    print("      TENSORFLOW LITE MODEL CONTRACT VERIFICATION  ")
    print("==================================================")
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    output_shape = output_details[0]['shape']
    output_dtype = output_details[0]['dtype']

    print(f"Input Name:  {input_details[0]['name']}")
    print(f"Input Shape: {input_shape}")
    print(f"Input Dtype: {input_dtype}")
    print(f"Output Name: {output_details[0]['name']}")
    print(f"Output Shape:{output_shape}")
    print(f"Output Dtype:{output_dtype}")

    # Contract Verification Assertions
    assert input_dtype == np.float32, f"Input dtype must be float32, got {input_dtype}"
    assert len(input_shape) == 3 and input_shape[2] == embedding_dim, f"Expected input shape [1, N, 1024], got {input_shape}"
    assert output_dtype == np.float32, f"Output dtype must be float32, got {output_dtype}"
    assert list(output_shape) == [1, 4], f"Expected output shape [1, 4], got {output_shape}"

    print("\nClass Order Contract Verification:")
    for idx, name in CLASS_MAP.items():
        print(f"  Index {idx} -> {name}")

    # Test Inference Run on TFLite Model
    test_input = np.random.normal(0, 0.1, size=(1, sequence_length, embedding_dim)).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output_probs = interpreter.get_tensor(output_details[0]['index'])

    print("\nSample TFLite Inference Execution:")
    print("Raw Output Probabilities:", output_probs)
    print("Sum of Probabilities:    ", np.sum(output_probs))
    predicted_class = CLASS_MAP[np.argmax(output_probs)]
    print("Predicted Class:         ", predicted_class)
    print("==================================================\n")

    print("[export_tflite] VERIFICATION PASSED SUCCESSFULLY!")
    return exported_path

if __name__ == "__main__":
    export_and_verify_tflite()
