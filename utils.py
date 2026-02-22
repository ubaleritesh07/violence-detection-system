import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("violence_model.h5")

IMG_SIZE = 128
MAX_FRAMES = 16

def extract_frames_for_display(video_path, save_folder):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_norm = frame / 255.0

        save_path = f"{save_folder}/frame_{count}.jpg"
        cv2.imwrite(save_path, frame)

        frames.append(frame_norm)
        count += 1

    cap.release()

    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames)

def predict_violence(video_path, frame_folder):
    frames = extract_frames_for_display(video_path, frame_folder)
    batch = np.expand_dims(frames, axis=0)

    prob = model.predict(batch, verbose=0)[0][0]
    label = "Violent" if prob >= 0.5 else "Non-Violent"

    return label, float(prob)

def predict_violence_frames(frames):
    frames = np.expand_dims(frames, axis=0)
    prob = model.predict(frames, verbose=0)[0][0]
    label = "Violent" if prob >= 0.5 else "Non-Violent"
    return label, float(prob)
