# emotion-detector
# 👁️‍🗨️ Face Emotion Detection

This project is a real-time face emotion detection app built using **TensorFlow**, **OpenCV**, and **Gradio**, and deployed on **Hugging Face Spaces**.

It allows users to:
- Upload an image (or optionally use their webcam)
- Automatically detect human faces
- Predict the facial emotion using a trained deep learning model

---

## 🚀 Demo

👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)  

---

## 📦 Contents

- `app.py` – Gradio interface code
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
git clone https://github.com/yourusername/emotion-detector.git
cd emotion-detector

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
python app.py
