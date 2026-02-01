import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
import json

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10   # fewer epochs needed for transfer learning

TRAIN_DIR = "final_dataset/train"
TEST_DIR = "final_dataset/test"

# -------------------------------
# DATA GENERATORS
# -------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

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
# SAVE CLASS LABELS
# -------------------------------
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("Class labels:", train_data.class_indices)

# -------------------------------
# LOAD PRETRAINED MOBILENET
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # freeze pretrained layers

# -------------------------------
# BUILD MODEL
# -------------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation="softmax")
])

# -------------------------------
# COMPILE
# -------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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
    epochs=EPOCHS
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("waste_classifier_mobilenet.h5")
print("✅ MobileNet model saved")
