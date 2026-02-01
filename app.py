import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CONFIDENCE_THRESHOLD = 40  # %

# --------------------------------------------------
# WASTE ACTION MAPPING
# --------------------------------------------------
WASTE_ACTIONS = {
    "plastic": [
        "Reduce single-use plastics",
        "Reuse containers if possible",
        "Recycle through plastic collection"
    ],
    "paper": [
        "Reduce unnecessary printing",
        "Reuse as scrap paper",
        "Recycle in paper bins"
    ],
    "glass": [
        "Reuse glass jars and bottles",
        "Recycle at glass collection centers"
    ],
    "metal": [
        "Reuse metal containers",
        "Recycle through scrap collection"
    ],
    "organic": [
        "Compost food waste",
        "Use for biogas or manure"
    ],
    "general": [
        "Segregate waste before disposal",
        "Avoid mixing recyclable waste"
    ]
}

# --------------------------------------------------
# LOAD MODEL & LABELS
# --------------------------------------------------
model = tf.keras.models.load_model("waste_classifier_mobilenet.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("🗑️ AI-Powered Waste Segregation System")
st.write("Upload a waste image to classify it and get disposal recommendations.")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # ----------------------------
    # DISPLAY IMAGE
    # ----------------------------
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    # ----------------------------
    # PREPROCESS IMAGE (MobileNet)
    # ----------------------------
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # ----------------------------
    # PREDICTION
    # ----------------------------
    predictions = model.predict(img_array)[0]
    top_index = int(np.argmax(predictions))
    top_confidence = predictions[top_index] * 100
    top_class = class_names[top_index]

    st.subheader("🔍 Prediction Result")

    # ----------------------------
    # DECISION LOGIC
    # ----------------------------
    if top_confidence < CONFIDENCE_THRESHOLD or top_class == "general":
        st.warning("⚠️ Mixed or Uncertain Waste Detected")
        st.write("This image appears to contain multiple or ambiguous waste types.")
        display_class = "general"

        st.write("Top possible categories:")
        top_indices = predictions.argsort()[-3:][::-1]
        for idx in top_indices:
            st.write(f"{class_names[idx]} : {predictions[idx]*100:.2f}%")
    else:
        st.success(f"Predicted Category: {top_class}")
        st.write(f"Confidence: {top_confidence:.2f}%")
        display_class = top_class

    # ----------------------------
    # RECOMMENDED ACTIONS
    # ----------------------------
    if display_class in WASTE_ACTIONS:
        st.subheader("♻️ Recommended Actions")
        for action in WASTE_ACTIONS[display_class]:
            st.write(f"• {action}")
