# 🩻 Pneumonia Detection using Chest X-Ray

This web app uses a deep learning model to classify chest X-ray images as **Normal** or **Abnormal** (potential pneumonia) using **TensorFlow** and **Gradio**.

---

## 🚀 How it Works

1. Upload a chest X-ray image (JPG or PNG).
2. Optionally enter the patient's name.
3. Click **"Analyze X-ray"**.
4. View prediction result and download a patient report.

---

## 🧠 Model Details

- ✅ Based on **MobileNetV2** architecture.
- 🧠 Trained to classify: **Normal** vs **Abnormal** X-rays.
- 📦 Saved model path: `model/pneumonia_detection_Vision_Model.keras`.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Gradio**
- **NumPy / Pillow**

---

## 📄 Report Feature

Each prediction generates a downloadable `.txt` report containing:
- Patient name
- Date/time
- Predicted result
- Confidence %
- Disclaimer notice

---

## ⚠️ Disclaimer

> This tool is intended for **educational and research purposes only**.
> It should **not** be used as a replacement for professional medical diagnosis.

---

## 👨‍💻 Developed By

Youssef Taha Badawi – AI & Deep Learning Enthusiast  
[LinkedIn](https://www.linkedin.com/in/yousseftaha1902/) | [GitHub](https://github.com/Youssef-Ai1001)