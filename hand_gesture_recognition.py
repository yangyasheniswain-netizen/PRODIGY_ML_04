import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = []
labels = []

dataset_path = r"C:\Users\YANGYASHENI SWAIN\Downloads\archive (2)\leapGestRecog"

for subject in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject)

    if os.path.isdir(subject_path):
        for gesture in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture)

            if os.path.isdir(gesture_path):
                for img_name in os.listdir(gesture_path)[:100]:
                    img_path = os.path.join(gesture_path, img_name)

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        data.append(img.flatten())
                        labels.append(gesture)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Hand Gesture Recognition Accuracy:", accuracy)