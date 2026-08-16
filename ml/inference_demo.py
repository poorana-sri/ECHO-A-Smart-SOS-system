"""
ECHO Reference End-to-End Inference Demonstration
=================================================

Demonstrates complete acoustic inference pipeline using the actual exported 
TensorFlow Lite models:
1. Audio Waveform (16kHz mono float32)
2. YAMNet Base Acoustic Model (yamnet.tflite) -> 1024-D temporal embeddings
3. Echo Sequence Classifier (echo_sequence_classifier.tflite) -> 4 emergency class probabilities
4. Displays formatted predictions, confidence, and class probabilities
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.features.yamnet_feature_extractor import YAMNetFeatureExtractor
from ml.data.prepare_dataset import CLASS_MAP, generate_source_audio

YAMNET_TFLITE_PATH = "ml/exported/yamnet.tflite"
CLASSIFIER_TFLITE_PATH = "ml/exported/echo_sequence_classifier.tflite"

def run_inference_demo(audio_path=None, test_scenario="NORMAL"):
    """
    Executes end-to-end inference from raw audio waveform to emergency class prediction.
    """
    print("================================================================================")
    print(f"               ECHO END-TO-END INFERENCE DEMO: {test_scenario:<18}")
    print("================================================================================")

    # 1. Obtain 16kHz mono audio waveform
    if audio_path and os.path.exists(audio_path):
        print(f"[inference_demo] Loading audio waveform from {audio_path}...")
        from scipy.io import wavfile
        sr, audio_data = wavfile.read(audio_path)
    else:
        print(f"[inference_demo] Generating 4.5-second realistic audio track ({test_scenario})...")
        scenario_map = {
            "NORMAL": (0, "fireworks"),           # Hard negative test
            "ACCIDENT": (1, "vehicle_crash"),
            "DISTRESS": (2, "scream"),
            "VIOLENT_INCIDENT": (3, "impulse_plus_shout")
        }
        class_idx, subtype = scenario_map.get(test_scenario, (0, "speech"))
        rng = np.random.RandomState(100)
        audio_data = generate_source_audio(class_idx, subtype, rng)
        sr = 16000

    # 2. Extract YAMNet 1024-D temporal embeddings
    extractor = YAMNetFeatureExtractor(model_path=YAMNET_TFLITE_PATH)
    features = extractor.extract_features(audio_data, sample_rate=sr)
    embeddings = features["embeddings"]
    
    # 3. Format into exact 6-frame sequence [1, 6, 1024]
    sequence_length = 6
    if len(embeddings) < sequence_length:
        pad_len = sequence_length - len(embeddings)
        embeddings = np.pad(embeddings, ((0, pad_len), (0, 0)), mode="constant")
    else:
        embeddings = embeddings[:sequence_length]
        
    input_tensor = np.expand_dims(embeddings, axis=0).astype(np.float32)

    # 4. Execute Echo Sequence Classifier TFLite Model
    if not os.path.exists(CLASSIFIER_TFLITE_PATH):
        raise FileNotFoundError(f"Exported classifier missing at {CLASSIFIER_TFLITE_PATH}. Run export_tflite.py first.")

    interpreter = tf.lite.Interpreter(model_path=CLASSIFIER_TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output_probs = interpreter.get_tensor(output_details[0]["index"])[0]

    # 5. Display Formatted Results
    print("\n--------------------------------------------------------------------------------")
    print("Class Probabilities:")
    for idx in range(4):
        class_name = CLASS_MAP[idx]
        prob_pct = output_probs[idx] * 100.0
        bar = "#" * int(prob_pct / 4)
        print(f"  [{idx}] {class_name:<18}: {prob_pct:6.2f}% | {bar}")
    print("--------------------------------------------------------------------------------")
    
    predicted_idx = int(np.argmax(output_probs))
    predicted_class = CLASS_MAP[predicted_idx]
    confidence = output_probs[predicted_idx] * 100.0
    
    print(f"PREDICTED CLASS: {predicted_class} (Confidence: {confidence:.2f}%)")
    print("================================================================================\n")
    
    return predicted_class, output_probs

if __name__ == "__main__":
    for scenario in ["NORMAL", "ACCIDENT", "DISTRESS", "VIOLENT_INCIDENT"]:
        run_inference_demo(test_scenario=scenario)
