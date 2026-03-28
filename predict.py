# =============================================================================
# predict.py  ─  Command-line single-image waste predictor
# Run:  python predict.py
# Classes: general, glass, metal, organic, paper, plastic
# =============================================================================

import os, sys, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH           = "waste_classifier_mobilenet.h5"
CONFIDENCE_THRESHOLD = 40    # % — show uncertain below this

# ── Full segregation guide per class ─────────────────────────────────────────
WASTE_INFO = {
    "plastic": {
        "bin":         "🔵 Blue Bin  (Dry Recyclable)",
        "bin_color":   "BLUE",
        "segregation": "Dry Recyclable — keep clean and dry",
        "actions": [
            "Rinse containers before placing in bin",
            "Remove caps and lids — recycle separately",
            "Flatten bottles to save space",
            "Separate by plastic type if required (PET, HDPE, PP etc.)",
            "Never mix with wet or organic waste"
        ],
        "do_not": [
            "Do not put dirty or greasy plastic in recycling",
            "Do not include plastic bags in recycling bin",
            "Do not mix with organic or hazardous waste"
        ],
        "tip": "Plastic takes 400+ years to decompose. Recycling 1 ton saves 7.4 cubic yards of landfill."
    },
    "paper": {
        "bin":         "🔵 Blue Bin  (Dry Recyclable)",
        "bin_color":   "BLUE",
        "segregation": "Dry Recyclable — must be kept dry",
        "actions": [
            "Keep paper dry and clean at all times",
            "Remove staples, paper clips and plastic windows",
            "Flatten cardboard boxes before placing in bin",
            "Newspapers, magazines, office paper all accepted",
            "Shred sensitive documents before recycling"
        ],
        "do_not": [
            "Do not recycle wet, oily or food-stained paper",
            "Do not include tissue paper or paper towels",
            "Do not mix with organic waste"
        ],
        "tip": "Recycling 1 ton of paper saves 17 trees and 7,000 gallons of water."
    },
    "metal": {
        "bin":         "🔵 Blue Bin  (Dry Recyclable)",
        "bin_color":   "BLUE",
        "segregation": "Dry Recyclable — separated by magnetic or eddy current methods",
        "actions": [
            "Rinse food cans and containers thoroughly",
            "Crush aluminium cans to save space",
            "Separate ferrous (iron/steel) from non-ferrous (aluminium)",
            "Remove paper labels if possible",
            "Aluminium foil can be recycled — scrunch into a ball first"
        ],
        "do_not": [
            "Do not include aerosol cans that are not fully empty",
            "Do not put sharp metal objects loose in the bin",
            "Do not mix with organic or hazardous waste"
        ],
        "tip": "Recycling aluminium uses 95% less energy than producing it from raw ore."
    },
    "glass": {
        "bin":         "⚫ Grey Bin / Specialised Glass Bin",
        "bin_color":   "GREY",
        "segregation": "Separated by colour — clear, green, brown",
        "actions": [
            "Rinse bottles and jars thoroughly",
            "Remove metal lids and caps — recycle separately",
            "Separate by colour where required (clear, green, brown)",
            "Keep whole — do not break glass intentionally"
        ],
        "do_not": [
            "Do not put broken glass in the recycling bin — wrap in newspaper",
            "Do not include ceramics, mirrors, or window glass",
            "Do not include drinking glasses or Pyrex"
        ],
        "tip": "Glass can be recycled endlessly without any loss in quality or purity."
    },
    "organic": {
        "bin":         "🟢 Green Bin  (Wet/Biodegradable Waste)",
        "bin_color":   "GREEN",
        "segregation": "Wet/Biodegradable — composted or bio-methanation",
        "actions": [
            "Compost food scraps, vegetable and fruit peels",
            "Include garden waste — grass, leaves, small branches",
            "Use a sealed bin to control odour",
            "Layer dry material (leaves) with wet scraps in compost"
        ],
        "do_not": [
            "Do not include meat, fish or dairy in home compost",
            "Do not mix with dry recyclables — moisture ruins paper/cardboard",
            "Do not include plastic bags even if marked biodegradable"
        ],
        "tip": "Composting organic waste reduces methane emissions and creates rich fertiliser for soil."
    },
    "general": {
        "bin":         "🔴 Red Bin / Grey General Waste Bin",
        "bin_color":   "RED",
        "segregation": "Residual/Non-Recyclable — sent to landfill or waste-to-energy plant",
        "actions": [
            "Segregate all recyclables out before using this bin",
            "Wrap sharp or messy items before disposing",
            "Reduce general waste by choosing recyclable packaging",
            "Check local council guidelines for accepted items"
        ],
        "do_not": [
            "Do not put hazardous waste (batteries, chemicals) in general bin",
            "Do not put e-waste in general bin",
            "Do not put medical waste in general bin"
        ],
        "tip": "Reducing general waste starts with buying less — choose products with recyclable packaging."
    },
}

# ── Guards ────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"❌  Model not found: {MODEL_PATH}")
    print("    Run  python train_mobilenet.py  first.")
    sys.exit(1)

# ── Load model & classes ──────────────────────────────────────────────────────
print("⏳  Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}
print(f"✅  Model ready  |  Classes: {[class_names[i] for i in sorted(class_names)]}\n")

# ── Get image path ────────────────────────────────────────────────────────────
img_path = input("Enter image path (drag & drop file into terminal): ").strip().strip('"')

if not os.path.exists(img_path):
    print(f"❌  File not found: {img_path}")
    sys.exit(1)

# ── Preprocess ────────────────────────────────────────────────────────────────
img       = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, 0)

# ── Predict ───────────────────────────────────────────────────────────────────
preds    = model.predict(img_array, verbose=0)[0]
top_idx  = int(np.argmax(preds))
top_conf = preds[top_idx] * 100

# ── Display ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("♻️   AI WASTE SEGREGATION SYSTEM  —  Prediction")
print("═" * 55)

if top_conf < CONFIDENCE_THRESHOLD:
    print("⚠️   Result     :  UNCERTAIN  (confidence too low)")
    print("\n    Top 3 predictions:")
    for idx in preds.argsort()[-3:][::-1]:
        bar = "█" * int(preds[idx] * 30)
        print(f"    • {class_names[idx]:<12}  {preds[idx]*100:5.1f}%  {bar}")
    predicted = "general"
    print("\n    Defaulting to General Waste advice.\n")
else:
    predicted = class_names[top_idx]
    bar       = "█" * int(top_conf / 3)
    print(f"  Waste Type   :  {predicted.upper()}")
    print(f"  Confidence   :  {top_conf:.1f}%  {bar}\n")
    print("  All class scores:")
    for idx in preds.argsort()[::-1]:
        b    = "█" * int(preds[idx] * 30)
        mark = " ◀ predicted" if idx == top_idx else ""
        print(f"    {class_names[idx]:<12}  {preds[idx]*100:5.1f}%  {b}{mark}")

info = WASTE_INFO.get(predicted, WASTE_INFO["general"])

print(f"\n{'─'*55}")
print(f"  Bin          :  {info['bin']}")
print(f"  Category     :  {info['segregation']}")
print(f"\n  ✅  What TO DO:")
for a in info["actions"]:
    print(f"      •  {a}")
print(f"\n  ❌  What NOT to do:")
for d in info["do_not"]:
    print(f"      •  {d}")
print(f"\n  💡  Did you know?")
print(f"      {info['tip']}")
print("═" * 55 + "\n")