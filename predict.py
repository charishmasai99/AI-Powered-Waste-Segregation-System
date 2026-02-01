import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os

CONFIDENCE_THRESHOLD = 40  # %

# Load model
model = tf.keras.models.load_model("waste_classifier.h5")

# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# Input image
img_path = input("Enter image path: ")

if not os.path.exists(img_path):
    print("❌ Image not found")
    exit()

# Preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)[0]

top_index = int(np.argmax(predictions))
top_confidence = predictions[top_index] * 100

print("\n==============================")
print("🗑️ Waste Type Prediction")
print("==============================")

if top_confidence < CONFIDENCE_THRESHOLD:
    print("Prediction : ❓ Uncertain")
    print("Top predictions:")
    top_indices = predictions.argsort()[-3:][::-1]
    for idx in top_indices:
        print(f"{class_names[idx]} : {predictions[idx]*100:.2f}%")
else:
    print(f"Predicted Category : {class_names[top_index]}")
    print(f"Confidence         : {top_confidence:.2f}%")

print("==============================\n")
