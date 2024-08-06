import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from transformers import pipeline
import sounddevice as sd
import librosa

# Load the trained models and labels
try:
    model = load_model("model.h5")
    label = np.load("labels.npy", allow_pickle=True)
    audio_model = load_model("audio_emotion_model.h5")
    audio_labels = np.load("audio_labels.npy", allow_pickle=True)
except Exception as e:
    print(f"Error loading model or labels: {e}")
    exit()

# Initialize Mediapipe holistic model
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
text_model = pipeline('text-classification', model='bhadresh-savani/bert-base-uncased-emotion')

# Function to predict emotion from text
def predict_text_emotion(text):
    try:
        predictions = text_model(text)
        emotion_label = predictions[0]['label']
        confidence = predictions[0]['score']
        return emotion_label, confidence
    except Exception as e:
        print(f"Error during text emotion prediction: {e}")
        return "Error", 0.0

# Function to record audio
def record_audio(duration=5, fs=44100):
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio.flatten(), fs

# Function to extract audio features
def extract_audio_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Function to predict emotion from audio
def predict_audio_emotion(audio_features):
    audio_features = np.expand_dims(audio_features, axis=0)
    predictions = audio_model.predict(audio_features)
    emotion_index = np.argmax(predictions)
    return audio_labels[emotion_index]

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    lst = []
    _, frm = cap.read()
    if not _:
        break

    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        # Predict the gesture
        pred = label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    # Draw landmarks on frame
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:  # Exit on ESC key
        cv2.destroyAllWindows()
        cap.release()
        break

# Text input emotion detection
text = input("Enter text for emotion detection: ")
text_emotion, text_confidence = predict_text_emotion(text)
print(f"Text Emotion: {text_emotion} ({text_confidence:.2f})")

# Audio input emotion detection
audio, sr = record_audio(duration=10)
audio_features = extract_audio_features(audio, sr)
audio_emotion = predict_audio_emotion(audio_features)
print(f"Audio Emotion: {audio_emotion}")
