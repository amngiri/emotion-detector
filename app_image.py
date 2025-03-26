
import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("face_emotion_detection.h5")
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_emotion(image):
    if image is None:
        return "No image received", image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected", image

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = img_to_array(roi_gray) / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        preds = model.predict(roi, verbose=0)[0]
        label = class_names[np.argmax(preds)]
        confidence = round(np.max(preds) * 100, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {confidence}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        return f"{label} ({confidence}%)", image

# Gradio interface (upload-only version)
gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(type="numpy", label="Upload a face image"),
    outputs=[gr.Label(label="Predicted Emotion"), gr.Image(type="numpy", label="Annotated Image")],
    title="Emotion Detector",
    description="Upload an image of a face and get emotion prediction."
).launch()
