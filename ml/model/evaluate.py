"""
Comprehensive Evaluation & Hard-Negative Stress Testing for ECHO
================================================================

Evaluates the trained Echo Sequence Classifier on the strictly held-out test split (test_data.npz).
Computes:
1. Overall Accuracy & Macro/Weighted F1 Scores.
2. Per-class Precision, Recall, F1-Score, and Support counts.
3. Full 4x4 Confusion Matrix.
4. Granular Hard-Negative Stress Analysis across non-emergency loud acoustic categories 
   (Fireworks, Cheering, Construction, Door Slams, Traffic, Music).
5. False-Positive Rate (FPR) specifically for NORMAL acoustic clips misclassified as emergencies.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.data.prepare_dataset import CLASS_MAP

def evaluate_model(data_dir="ml/data/splits", model_path="ml/model/saved_model.keras"):
    """
    Runs comprehensive evaluation and hard-negative stress testing on held-out test set.
    """
    test_path = os.path.join(data_dir, "test_data.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found at {test_path}. Run prepare_dataset.py first.")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found at {model_path}. Run train.py first.")
        
    print(f"[evaluate] Loading held-out test data from {test_path}...")
    test_data = np.load(test_path, allow_pickle=True)
    X_test = test_data["X"]
    y_test = test_data["y"]
    test_meta = test_data["meta"] if "meta" in test_data else np.array(["unknown"] * len(y_test))
    test_hard_neg = test_data["hard_neg"] if "hard_neg" in test_data else np.array([False] * len(y_test))
    
    print(f"[evaluate] Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"[evaluate] Executing inference across {len(X_test)} test sequences...")
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    
    # 1. Overall & Classification Metrics
    accuracy = float(np.mean(y_pred == y_test))
    target_names = [CLASS_MAP[i] for i in range(4)]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    
    # 2. General False Positive Analysis
    normal_mask = (y_test == 0)
    total_normal = int(np.sum(normal_mask))
    normal_false_positives = int(np.sum(y_pred[normal_mask] != 0))
    overall_fpr = float(normal_false_positives / total_normal) if total_normal > 0 else 0.0
    
    # 3. Dedicated Hard-Negative Analysis
    subtypes = np.unique(test_meta)
    hard_neg_results = {}
    
    print("\n================================================================================")
    print("                      ECHO MODEL EVALUATION REPORT                             ")
    print("================================================================================")
    print(f"Total Held-Out Test Samples: {len(y_test)}")
    print(f"Overall Test Accuracy:       {accuracy * 100:.2f}%")
    print(f"Macro F1-Score:              {report_dict['macro avg']['f1-score'] * 100:.2f}%")
    print(f"Weighted F1-Score:           {report_dict['weighted avg']['f1-score'] * 100:.2f}%")
    print(f"Overall Normal False Alarm:  {overall_fpr * 100:.2f}% ({normal_false_positives}/{total_normal} normal clips)")
    print("\n--------------------------------------------------------------------------------")
    print("Classification Metrics by Category:")
    print(report_str)
    print("--------------------------------------------------------------------------------")
    print("Confusion Matrix:")
    print("Row: Actual Ground Truth | Column: Model Prediction")
    print(f"{'':<18} | {'NORMAL':<10} | {'ACCIDENT':<10} | {'DISTRESS':<10} | {'VIOLENT':<10}")
    print("-" * 75)
    for idx, row in enumerate(cm):
        class_label = CLASS_MAP[idx]
        print(f"{class_label:<18} | {row[0]:<10} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10}")
        
    print("\n================================================================================")
    print("                 HARD-NEGATIVE ACOUSTIC STRESS ANALYSIS                         ")
    print("================================================================================")
    print("Evaluating false-positive resilience against non-emergency loud sound categories:\n")
    print(f"{'Sound Category':<20} | {'True Class':<10} | {'Count':<6} | {'Correct':<8} | {'False Alarms':<14} | {'Error Rate':<10}")
    print("-" * 80)
    
    for sub in subtypes:
        sub_mask = (test_meta == sub)
        sub_count = int(np.sum(sub_mask))
        sub_actual = int(y_test[sub_mask][0])
        sub_preds = y_pred[sub_mask]
        sub_correct = int(np.sum(sub_preds == sub_actual))
        sub_errors = sub_count - sub_correct
        err_rate = float(sub_errors / sub_count) * 100 if sub_count > 0 else 0.0
        
        hard_neg_results[sub] = {
            "count": sub_count,
            "correct": sub_correct,
            "errors": sub_errors,
            "error_rate_pct": err_rate,
            "true_class": CLASS_MAP[sub_actual]
        }
        print(f"{sub:<20} | {CLASS_MAP[sub_actual]:<10} | {sub_count:<6} | {sub_correct:<8} | {sub_errors:<14} | {err_rate:6.1f}%")
        
    print("================================================================================\n")
    
    return {
        "accuracy": accuracy,
        "macro_f1": report_dict['macro avg']['f1-score'],
        "weighted_f1": report_dict['weighted avg']['f1-score'],
        "overall_fpr": overall_fpr,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "hard_neg_results": hard_neg_results
    }

if __name__ == "__main__":
    evaluate_model()
