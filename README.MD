## Pneumonia Detection from Chest X-Ray using CNN and Gradio

This repository contains an end-to-end pipeline for detecting pneumonia from chest X-ray images using a convolutional neural network (CNN) based on MobileNetV2, trained on a public medical imaging dataset and deployed as an interactive Gradio web application.

The project demonstrates the full workflow from dataset selection and baseline model choice, through data preprocessing and model training, to model export and web deployment.

---

## 1. Project Links

- **GitHub repository**: [`Xray-Analysis-using-CNN-and-gradio-deployment`](https://github.com/Youssef-Ai1001/Xray-Analysis-using-CNN-and-gradio-deployment)
- **Training notebook (Kaggle)**: [`Chest-Xray-Vision-98`](https://www.kaggle.com/code/yousseftaha98/chest-xray-vision-98)
- **Dataset (Kaggle)**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 2. Problem Statement

Pneumonia is a serious lung infection that can be identified in chest X-ray images.  
The objective of this project is to build a computer vision model that:

- **Classifies** a chest X-ray as either **Normal** or **Pneumonia (Abnormal)**.
- **Provides** an easy-to-use web interface for uploading an image and obtaining a prediction.
- **Supports** reproducible experimentation through a structured notebook and codebase.

This tool is designed for educational and research purposes and is **not** intended to replace clinical diagnosis.

---

## 3. Dataset and Preprocessing

- **Dataset source**: [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**:
  - `NORMAL`
  - `PNEUMONIA`
- **Data structure** (as provided on Kaggle):
  - `train/` – training images (Normal and Pneumonia)
  - `val/` – validation images
  - `test/` – held-out test images

### 3.1 Preprocessing Steps

The main preprocessing and input pipeline decisions are:

- **Image resizing**: All images are resized to the MobileNetV2 input resolution (e.g., `224 x 224 x 3`).
- **Color channels**: Images are converted to 3-channel RGB (even if originally grayscale).
- **Normalization**: Pixel values are scaled to the range required by MobileNetV2 (e.g., `[0, 1]` or using the built-in preprocessing of `tf.keras.applications.mobilenet_v2.preprocess_input`).
- **Train/Validation/Test splits**: The original dataset splits are preserved to ensure fair evaluation.
- **Data augmentation** (applied on the training set):
  - Random horizontal flips
  - Small random rotations / shifts
  - Slight zoom / rescale

These steps improve generalization and make the model more robust to variations in real-world chest X-rays.

For full details of the preprocessing and experimentation pipeline, refer to the Kaggle notebook:  
[`Chest-Xray-Vision-98`](https://www.kaggle.com/code/yousseftaha98/chest-xray-vision-98).

---

## 4. Baseline Model and Architecture

### 4.1 Baseline Choice

The baseline model was chosen to balance **accuracy**, **parameter efficiency**, and **deployment speed**:

- **Backbone**: `MobileNetV2` (pretrained on ImageNet).
- **Reasoning**:
  - Lightweight and efficient, suitable for deployment on CPUs.
  - Strong performance on transfer learning tasks with limited medical data.
  - Faster training and inference compared to heavier architectures (e.g., ResNet50, Inception).

### 4.2 Model Architecture

The final classifier is built as follows:

- Load **MobileNetV2** without the top classification head (`include_top=False`).
- Use **global average pooling** to compress spatial feature maps.
- Add a small **fully connected head**:
  - Dense layer(s) with ReLU activation (optional dropout for regularization).
  - Final Dense layer with a **sigmoid** or **softmax** activation for binary classification (`Normal` vs `Pneumonia`).
- Optionally **freeze** the lower (earlier) layers of MobileNetV2 initially and then **fine-tune** some of the deeper layers.

Training and evaluation details, including experiments, metrics, and plots, are provided in the Kaggle notebook:  
[`Chest-Xray-Vision-98`](https://www.kaggle.com/code/yousseftaha98/chest-xray-vision-98).

---

## 5. Model Training and Export

### 5.1 Training Strategy

- **Loss function**: Binary cross-entropy.
- **Optimizer**: Adam (with tuned learning rate).
- **Metrics**: Accuracy, along with monitoring validation loss and accuracy.
- **Callbacks** (typical configuration):
  - Early stopping on validation loss.
  - Model checkpointing to save the best weights.

Training is performed in the Kaggle environment, using the dataset and preprocessing described above.

### 5.2 Saved Model

After training, the best-performing model is exported and saved in Keras format:

- **Path**: `models/pneumonia_detection_Vision_Model.keras`

This model is then loaded by the Gradio application for inference.

---

## 6. Gradio Web Application

The `app/` directory contains the Gradio-based deployment of the trained model.

### 6.1 Functionality

The web app provides:

- Image upload (JPG/PNG chest X-ray).
- Optional patient name field.
- Button to run the **"Analyze X-ray"** prediction.
- Display of:
  - Predicted class: **Normal** or **Pneumonia (Abnormal)**.
  - Model confidence score.
- Automatic generation of a **downloadable text report**.

### 6.2 Report Generation

Each prediction generates a `.txt` report that includes:

- Patient name (if provided).
- Date and time of analysis.
- Predicted label.
- Model confidence (percentage).
- Medical disclaimer.

---

## 7. Project Structure

High-level structure of the repository:

- `app/` – Gradio web application and inference logic.
  - `app.py` – main application entry point (recommended for running the app).
  - `app_v1.py`, `app_v2.py` – earlier experimental versions kept for reference.
- `models/` – Saved Keras model (`pneumonia_detection_Vision_Model.keras`).
- `notebooks/` – Supporting notebooks (in addition to the main Kaggle notebook).
- `docs/` – Documentation-related assets.
- `reports/` – Generated or example reports and analysis outputs.
- `Images/` – Project-related images (e.g., screenshots).
- `requirements.txt` – Python dependencies for running the app.

---

## 8. Installation and Usage

### 8.1 Setup

```bash
git clone https://github.com/Youssef-Ai1001/Xray-Analysis-using-CNN-and-gradio-deployment.git
cd Xray-Analysis-using-CNN-and-gradio-deployment
pip install -r requirements.txt
```

Ensure that the trained model file `pneumonia_detection_Vision_Model.keras` is available under the `models/` directory (you can download or export it from the Kaggle notebook if needed).

### 8.2 Run the Gradio App

From the project root:

```bash
python app/app.py
```

Then open the local URL printed by Gradio in your browser and use the interface to upload chest X-ray images.

---

## 9. Limitations and Disclaimer

- This model is trained on a specific public dataset and may not generalize perfectly to all clinical settings or imaging devices.
- Predictions can be affected by image quality, preprocessing, and dataset biases.
- The tool is intended **only for educational and research purposes** and must **not** be used as a clinical decision-making tool.

> Always consult qualified medical professionals for diagnosis and treatment decisions.

---

## 10. Author

**Youssef Taha Badawi** – AI & Deep Learning Enthusiast  
[GitHub](https://github.com/Youssef-Ai1001) | [LinkedIn](https://www.linkedin.com/in/yousseftahaai/)