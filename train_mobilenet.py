# =============================================================================
# train_mobilenet.py  ─  MobileNetV2 waste classifier  (metal-aware)
# Run:  python train_mobilenet.py
# =============================================================================

import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from collections import Counter

IMG_SIZE        = 224
BATCH_SIZE      = 16     # reduced — more stable on CPU
EPOCHS_FROZEN   = 15
EPOCHS_FINETUNE = 10
TRAIN_DIR       = "final_dataset/train"
TEST_DIR        = "final_dataset/test"

# ── Dataset check ─────────────────────────────────────────────────────────────
for d in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(d):
        print(f"❌  Not found: {d}"); exit(1)

# Count metal images
metal_path = os.path.join(TRAIN_DIR, "metal")
if os.path.exists(metal_path):
    metal_imgs = [f for f in os.listdir(metal_path)
                  if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
    print(f"✅  Metal images in train: {len(metal_imgs)}")
    if len(metal_imgs) < 100:
        print("⚠️  WARNING: Need at least 100 metal images for reliable detection.")
        print("   Download from: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification")
else:
    print("❌  final_dataset/train/metal/ folder missing — creating it.")
    print("   Add metal images before retraining.")
    os.makedirs(metal_path, exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, "metal"), exist_ok=True)

# ── Data generators ───────────────────────────────────────────────────────────
# Extra aggressive augmentation for metal/glass (shiny surfaces)
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range         = 30,
    zoom_range             = 0.3,
    horizontal_flip        = True,
    vertical_flip          = True,          # metal/glass look same upside down
    width_shift_range      = 0.2,
    height_shift_range     = 0.2,
    shear_range            = 0.15,
    brightness_range       = [0.6, 1.4],    # simulate shiny/dull surfaces
    channel_shift_range    = 30.0,          # helps differentiate metal vs glass
    fill_mode              = "nearest"
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
    shuffle     = False
)

# ── Save class indices ────────────────────────────────────────────────────────
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=2)

print(f"\n✅  Classes: {train_data.class_indices}")
print(f"    Train: {train_data.samples}  |  Test: {test_data.samples}")

# ── Class weights — give metal and glass double weight ────────────────────────
counter   = Counter(train_data.classes)
max_count = max(counter.values())
class_weights = {}
for cls, count in counter.items():
    name = list(train_data.class_indices.keys())[cls]
    # Boost metal and glass extra since they're hard to distinguish
    boost = 2.5 if name in ("metal", "glass") else 1.0
    class_weights[cls] = round((max_count / count) * boost, 3)

print(f"⚖️   Class weights: {class_weights}")

# ── Model ─────────────────────────────────────────────────────────────────────
base_model = MobileNetV2(
    weights="imagenet", include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.45),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.35),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-4),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)
model.summary()

# ── Phase 1 ───────────────────────────────────────────────────────────────────
print("\n🚀  Phase 1 — Training top layers ...\n")

cb1 = [
    tf.keras.callbacks.ModelCheckpoint(
        "waste_classifier_mobilenet.h5",
        monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1)
]

h1 = model.fit(train_data, validation_data=test_data,
               epochs=EPOCHS_FROZEN, class_weight=class_weights,
               callbacks=cb1)

# ── Phase 2 — unfreeze top 50 layers ─────────────────────────────────────────
print("\n🔓  Phase 2 — Fine-tuning top 50 layers ...\n")

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(5e-6),  # very low LR for fine-tuning
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)

cb2 = [
    tf.keras.callbacks.ModelCheckpoint(
        "waste_classifier_mobilenet.h5",
        monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6,
        restore_best_weights=True, verbose=1)
]

h2 = model.fit(train_data, validation_data=test_data,
               epochs=EPOCHS_FINETUNE, class_weight=class_weights,
               callbacks=cb2)

# ── Save history ──────────────────────────────────────────────────────────────
combined = {k: h1.history[k] + h2.history.get(k,[]) for k in h1.history}
with open("mobilenet_history.json","w") as f:
    json.dump({k:[float(v) for v in vals] for k,vals in combined.items()},f,indent=2)

best = max(combined.get("val_accuracy",[0]))
print(f"\n✅  Done! Best val_accuracy: {best*100:.1f}%")
print(f"    Model → waste_classifier_mobilenet.h5")