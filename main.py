import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

clf = joblib.load("fingernail_ml_model_AI.joblib")
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

IMG_SIZE = 224
NAIL_CROP_HEIGHT = 120
NAIL_CROP_WIDTH = 160

prediction_history = deque(maxlen=30)
full_history = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

root = tk.Tk()
root.title("Fingernail Condition Detector")
root.geometry("1200x700")
root.configure(bg="#ffffff")

# LEFT PANEL
left_frame = tk.Frame(root, bg="#ffffff")
left_frame.pack(side=tk.LEFT, padx=20, pady=20)

tk.Label(left_frame, text="Detection History", font=("Arial", 16), bg="#ffffff").pack()

# Scrollable Listbox
history_container = tk.Frame(left_frame)
history_container.pack()

scrollbar = tk.Scrollbar(history_container)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

history_listbox = tk.Listbox(
    history_container, width=40, height=20,
    font=("Arial", 12),
    yscrollcommand=scrollbar.set
)
history_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
scrollbar.config(command=history_listbox.yview)

# Prediction label
prediction_var = tk.StringVar()
prediction_label = tk.Label(left_frame, textvariable=prediction_var,
                            font=("Arial", 20), bg="#ffffff")
prediction_label.pack(pady=20)

# Track unique predictions
unique_predictions = set()

# RIGHT PANEL (Camera + ROI)
right_frame = tk.Frame(root, bg="#ffffff")
right_frame.pack(side=tk.RIGHT, padx=20, pady=20)

tk.Label(right_frame, text="Camera Feed", font=("Arial", 16), bg="#ffffff").pack()
camera_label = tk.Label(right_frame, bg="#dddddd")
camera_label.pack(pady=10)

tk.Label(right_frame, text="Nail ROI", font=("Arial", 16), bg="#ffffff").pack()
roi_label = tk.Label(right_frame, bg="#dddddd")
roi_label.pack(pady=10)

cap = cv2.VideoCapture(0)

best_roi_frame = None
latest_prediction = None
anemia_level = None

def estimate_anemia_level(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = np.mean(hsv[:, :, 2])  # Brightness
    # Example thresholds (you may need to calibrate)
    if v > 170:
        return "Normal"
    elif v > 150:
        return "Mild"
    elif v > 130:
        return "Moderate"
    else:
        return "Severe"

def camera_processing():
    global best_roi_frame, latest_prediction, anemia_level, processed_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        best_roi = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            for tip in [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]:
                cx = int(lm.landmark[tip].x * w)
                cy = int(lm.landmark[tip].y * h)

                y1 = max(0, cy - NAIL_CROP_HEIGHT)
                y2 = min(h, cy)
                x1 = max(0, cx - (NAIL_CROP_WIDTH // 2))
                x2 = min(w, cx + (NAIL_CROP_WIDTH // 2))

                roi = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                if roi.size > 0:
                    best_roi = roi

        if best_roi is not None:
            best_roi_frame = best_roi.copy()

            nail = cv2.resize(best_roi, (IMG_SIZE, IMG_SIZE))
            nail = cv2.cvtColor(nail, cv2.COLOR_BGR2RGB)
            nail = np.expand_dims(nail, 0)
            nail = preprocess_input(nail)

            features = feature_extractor.predict(nail)
            pred = clf.predict(features)[0]

            prediction_history.append(pred)
            full_history.append(pred)

            latest_prediction = max(set(prediction_history), key=prediction_history.count)
            print(latest_prediction)

            # If anemia detected → calculate level
            if 'Anaemia' == latest_prediction:
                anemia_level = estimate_anemia_level(best_roi)
            else:
                anemia_level = None

        processed_frame = frame.copy()
        time.sleep(0.01)

def update_gui():
    if "processed_frame" in globals():
        img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(img)
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)

    if best_roi_frame is not None:
        roi_img = Image.fromarray(cv2.cvtColor(best_roi_frame, cv2.COLOR_BGR2RGB))
        roi_img = roi_img.resize((200, 150))
        roi_imgtk = ImageTk.PhotoImage(roi_img)
        roi_label.imgtk = roi_imgtk
        roi_label.config(image=roi_imgtk)

    # Update prediction and anemia level
    if latest_prediction is not None:
        if anemia_level:
            prediction_var.set(f"Condition: {latest_prediction} ({anemia_level})")
        else:
            prediction_var.set(f"Condition: {latest_prediction}")

        if latest_prediction not in unique_predictions:
            unique_predictions.add(latest_prediction)
            history_listbox.insert(tk.END, latest_prediction)

    root.after(30, update_gui)


# Start thread
thread = threading.Thread(target=camera_processing, daemon=True)
thread.start()

update_gui()
root.mainloop()

cap.release()
hands.close()
cv2.destroyAllWindows()
