# 👁️‍🗨️ Face Emotion Detection

This project is a real-time face emotion detection app built using **TensorFlow**, **OpenCV**, and **Gradio**, and deployed on **Hugging Face Spaces**.

It allows users to:
- Upload an image or capture a picture using their device camera
- Automatically detect human faces
- Predict the facial emotion using a trained deep learning model

For real-time live emotion detection, users can run the `app_web_cam.py` file locally.

---

## 🚀 Demo

👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/GiriAman/emotion-detector)  

---

## 📦 Contents

- `app.py` – Gradio interface code for Hugging Face deployment
- `app_web_cam.py` – Code for real-time webcam-based emotion detection (Run locally)
- `face_emotion_detection.h5` – Pre-trained Keras model
- `requirements.txt` – Python dependencies
- `README.md` – Project description

---

## 🧠 Emotions Detected

This model predicts one of the following 7 emotions:

- 😠 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😀 Happy  
- 😐 Neutral  
- 😢 Sad  
- 😲 Surprise  

---

## 🧪 Model Details

- Input: 48x48 grayscale face image
- Architecture: Custom CNN trained on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset
- Framework: TensorFlow / Keras
- Face detection: OpenCV Haar cascades

---

## 🛠️ How to Run Locally

```bash
git clone https://github.com/amngiri/emotion-detector.git
cd emotion-detector
pip install -r requirements.txt
```


### Running the Real-Time Webcam Version
```bash
python app_web_cam.py
```
Ensure that the `face_emotion_detection.h5` model file is in the same directory as `app_web_cam.py`.

