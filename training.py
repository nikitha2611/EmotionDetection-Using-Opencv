import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Constants
DATASET_PATH = r"C:\Users\Harsha\Downloads\liveEmoji-main\TESS Toronto emotional speech set data"
EMOTIONS = ["OAF_happy", "OAF_sad", "YAF_fear", "OAF_angry", "YAF_neutral"]
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40

# Function to extract audio features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Load dataset and extract features
features = []
labels = []

for emotion in EMOTIONS:
    emotion_path = os.path.join(DATASET_PATH, emotion)
    if not os.path.exists(emotion_path):
        print(f"Directory {emotion_path} does not exist.")
        continue
    
    for file_name in os.listdir(emotion_path):
        file_path = os.path.join(emotion_path, file_name)
        if os.path.isfile(file_path):
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)
            else:
                print(f"Feature extraction failed for {file_path}")

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, num_classes=len(EMOTIONS))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(256, input_shape=(N_MFCC,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(EMOTIONS), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save the model and labels
model.save("audio_emotion_model.h5")
np.save("audio_labels.npy", le.classes_)
