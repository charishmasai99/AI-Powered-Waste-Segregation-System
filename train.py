import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json
from collections import Counter

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = "final_dataset/train"
TEST_DIR = "final_dataset/test"

# -------------------------------
# DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -------------------------------
# SAVE CLASS INDICES
# -------------------------------
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("Class labels:", train_data.class_indices)

# -------------------------------
# CLASS WEIGHTS (IMBALANCE FIX)
# -------------------------------
counter = Counter(train_data.classes)
max_count = max(counter.values())

class_weights = {
    cls: max_count / count
    for cls, count in counter.items()
}

print("Class weights:", class_weights)

# -------------------------------
# CNN MODEL
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu",
                  input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# -------------------------------
# SAVE MODEL + HISTORY
# -------------------------------
model.save("waste_classifier.h5")

with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("✅ Model and training history saved")
