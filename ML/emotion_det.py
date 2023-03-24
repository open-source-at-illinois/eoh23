import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the FER2013 dataset
data = pd.read_csv('fer2013.csv')

# Extract the training data
x_train = []
y_train = []
for i in range(len(data)):
    pixels = np.array(list(data['pixels'][i].split(' ')), dtype=np.uint8)
    image = pixels.reshape((48, 48))
    x_train.append(image)
    y_train.append(data['emotion'][i])

# Convert the training data to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create a Mediapipe Face Detection object
mp_face_detection = mp.solutions.face_detection.FaceDetection()

# This function extracts facial landmarks
def get_landmarks(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #RGB Conversion

    # Mediapipe's Face Detection function
    results = mp_face_detection.process(frame)

    if results.detections:
        # Get the landmarks of the face
        landmarks = results.detections[0].location_data.relative_keypoints

        # Convert the landmarks to a numpy array
        landmarks = np.array([(landmark.x, landmark.y) for landmark in landmarks])

        return landmarks

    return None

def preprocess_landmarks(landmarks):
    landmarks[:, 0] = (landmarks[:, 0] - 0.5) * 2
    landmarks[:, 1] = (landmarks[:, 1] - 0.5) * 2

    return landmarks

def predict_emotion(landmarks):
    # Preprocess the landmarks
    landmarks = preprocess_landmarks(landmarks)

    landmarks = landmarks.flatten()

    # SVM classifier
    return clf.predict([landmarks])[0]

clf = SVC(kernel='linear', random_state=42)
clf.fit(x_train.reshape(len(x_train), -1), y_train)

# Evaluate the classifier on the validation data
accuracy = clf.score(X_val.reshape(len(X_val), -1), y_val)
print("Validation accuracy:", accuracy)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        landmarks = get_landmarks(frame)

        if landmarks is not None:
            emotion = predict_emotion(landmarks)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, emotion, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
