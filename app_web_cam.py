
import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

cv2.setNumThreads(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error: Could not load face cascade.")
    exit()

model_path = "G:/aman office/practise/face_emotion_detection/face_emotion_detection.h5"  
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
if not cap.isOpened():
    print("Error: Could not access the webcam. Try restarting or using a different device ID.")
    exit()

cv2.namedWindow("Real-time Emotion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-time Emotion Detection", 800, 600)

def process_face(face_gray, output_list, idx):
    try:
        face = cv2.resize(face_gray, (48, 48))
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        predictions = model.predict(face, verbose=0)[0]
        output_list[idx] = predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        output_list[idx] = np.zeros(len(class_names))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        threads = []
        results = [None] * len(faces)

        for i, (x, y, w, h) in enumerate(faces):
            face = gray[y:y + h, x:x + w]
            thread = threading.Thread(target=process_face, args=(face, results, i))
            thread.start()
            threads.append((thread, x, y, w, h))

        for i, (thread, x, y, w, h) in enumerate(threads):
            thread.join()
            predictions = results[i]
            if predictions is None:
                continue

            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]
            confidence = round(predictions[predicted_index] * 100, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_class}: {confidence}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Real-time Emotion Detection", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Real-time Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Exit triggered.")
            break

finally:
    print("Releasing camera and closing windows.")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
