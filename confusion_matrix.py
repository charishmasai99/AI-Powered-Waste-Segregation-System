# =============================================================================
# confusion_matrix.py  ─  Evaluate model and plot confusion matrix
# Run:  python confusion_matrix.py
# NOTE: Requires  waste_classifier_mobilenet.h5  to already exist
# =============================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "waste_classifier_mobilenet.h5"
TEST_DIR   = "final_dataset/test"

# ── Guard checks ──────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"❌  Model not found: {MODEL_PATH}")
    print("    Run  python train_mobilenet.py  first.")
    exit(1)

if not os.path.exists(TEST_DIR):
    print(f"❌  Test folder not found: {TEST_DIR}")
    exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
print("⏳  Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ── Load class names ──────────────────────────────────────────────────────────
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
print("✅  Classes:", class_names)

# ── Test data generator (no augmentation) ─────────────────────────────────────
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = (224, 224),
    batch_size  = 32,
    class_mode  = "categorical",
    shuffle     = False
)

# ── Predict ───────────────────────────────────────────────────────────────────
print("⏳  Running predictions on test set ...")
predictions = model.predict(test_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# ── Per-class accuracy ────────────────────────────────────────────────────────
print("\n📊  Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

overall_acc = np.mean(y_pred == y_true) * 100
print(f"✅  Overall Test Accuracy: {overall_acc:.2f}%\n")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm   = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(9, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=True)

ax.set_title(f"Confusion Matrix – Waste Classification\n"
             f"Overall Accuracy: {overall_acc:.1f}%",
             fontsize=14, fontweight="bold")

plt.tight_layout()

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/confusion_matrix.png", dpi=150)
print("📁  Saved to  assets/confusion_matrix.png")
plt.show()