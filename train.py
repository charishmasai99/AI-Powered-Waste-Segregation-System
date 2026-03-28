# =============================================================================
# train.py  ─  Train the custom CNN waste classifier
# Run:  python train.py
# =============================================================================

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
TRAIN_DIR  = "final_dataset/train"
TEST_DIR   = "final_dataset/test"

# ── Check dataset exists ──────────────────────────────────────────────────────
if not os.path.exists(TRAIN_DIR):
    print(f"❌  ERROR: Training folder not found at  '{TRAIN_DIR}'")
    print("    Create the folder structure:")
    print("    final_dataset/")
    print("      train/  plastic/  paper/  glass/  metal/  organic/  general/")
    print("      test/   plastic/  paper/  glass/  metal/  organic/  general/")
    exit(1)

# ── Data augmentation (training) ──────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    rotation_range   = 30,
    zoom_range       = 0.3,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    shear_range      = 0.2,
    brightness_range = [0.7, 1.3],
    horizontal_flip  = True,
    fill_mode        = "nearest"
)

# Only rescale for test / validation data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = "categorical"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = "categorical",
    shuffle     = False          # keep order for confusion matrix
)

# ── Save class indices ────────────────────────────────────────────────────────
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=2)

print("\n✅  Class labels saved to class_indices.json")
print("    Classes found:", train_data.class_indices)

# ── Handle class imbalance ────────────────────────────────────────────────────
counter   = Counter(train_data.classes)
max_count = max(counter.values())

class_weights = {cls: max_count / count for cls, count in counter.items()}
print("\n⚖️   Class weights (for imbalance):", class_weights)

# ── CNN Model ─────────────────────────────────────────────────────────────────
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 4  ← added vs original for better feature depth
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer = "adam",
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    # Save the best checkpoint automatically
    tf.keras.callbacks.ModelCheckpoint(
        "waste_classifier.h5",
        monitor   = "val_accuracy",
        save_best_only = True,
        verbose   = 1
    ),
    # Reduce learning rate when stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 3,
        verbose  = 1
    ),
    # Stop early if no improvement for 5 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor  = "val_accuracy",
        patience = 5,
        restore_best_weights = True,
        verbose  = 1
    )
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n🚀  Starting training ...\n")

history = model.fit(
    train_data,
    validation_data = test_data,
    epochs          = EPOCHS,
    class_weight    = class_weights,
    callbacks       = callbacks
)

# ── Save training history ─────────────────────────────────────────────────────
with open("training_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals]
               for k, vals in history.history.items()}, f, indent=2)

print("\n✅  Training complete!")
print("    Model saved   →  waste_classifier.h5")
print("    History saved →  training_history.json")