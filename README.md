# ♻️ EcoSort AI – Smart Waste Segregation System

An AI-powered web application that classifies waste using deep learning and provides actionable recycling guidance.

---

## 🚀 Features

* 🧠 **AI Waste Classification** (MobileNetV2-based model)
* 🔍 Real-time image prediction
* 📊 Confidence-aware predictions
* ♻️ Smart bin recommendations (Blue, Green, Red, Grey)
* 💡 Eco-friendly tips & upcycling ideas
* 🔐 Google Authentication login
* 📈 Dataset analysis & model evaluation tools

---

## 🖥️ Demo

> Upload an image → Get waste category → See bin + eco suggestions

---

## 📂 Project Structure

```
AI_Waste_Segregation/
│── app.py                  # Main Streamlit app
│── auth_module.py         # Authentication logic
│── predict.py             # Model prediction
│── dataset_analysis.py    # Dataset insights
│── confusion_matrix.py    # Model evaluation
│── class_indices.json     # Class mapping
│── assets/                # Images & UI assets
│── requirements.txt       # Dependencies
```

---

## ⚙️ Installation

```bash
git clone https://github.com/charishmasai99/AI-Powered-Waste-Segregation-System.git
cd AI-Powered-Waste-Segregation-System

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📊 Model Details

* Architecture: **MobileNetV2**
* Task: Multi-class image classification
* Output: Waste category + confidence score

---

## 📈 Evaluation Tools

Run:

```bash
python confusion_matrix.py
python dataset_analysis.py
```

---

## 🔐 Authentication

* Google OAuth login supported
* Secrets stored securely via Streamlit

---

## 🌱 Use Case

* Smart waste segregation
* Recycling awareness
* Environmental sustainability

---

## 👩‍💻 Author

**Sai Charishma**

GitHub: https://github.com/charishmasai99

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
