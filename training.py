import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

def normalize(sample):
    """
    Transform the array into a 21:3 matrix and normalize it using the wrist as the origin. This is with the objective
    of maintaining consistency in the data while training the model and to avoid that the position of the hand
    affects the prediction
    :param
    sample: np.ndarray
    Array of size 63 with the (x,y,z) coordinates of the 21 mediapipe landmarks in the hand
    :return:
    np.ndarray
    Array of size 63 with the normalized coordinates of the 21 mediapipe landmarks in the hand
    """
    pts = sample.reshape(21, 3)
    pts = pts - pts[0]

    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val

    return pts.flatten()

DATASET_DIR = "dataset_landmarks"

X = [] #Normalized coordinates of each sample in the training set
y = [] #Letter correspond to each sample in X

#Iterates in the training set, normalize each sample using the function from
#the utils model and append it to X and Y (Normalized Coordinates, Letter) to
#later train the model
for label in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        if file.endswith(".npy"):
            path = os.path.join(class_dir, file)

            data = np.load(path)

            if len(data.shape) == 1:
                data = [data]

            for sample in data:
                X.append(normalize(sample))
                y.append(label)

X = np.array(X)
y = np.array(y)

print("Data uploaded", X.shape)

#Random Forest with 500 estimators
model = RandomForestClassifier(n_estimators=500 , oob_score = True)
model.fit(X, y)

print("OOB accuracy = ", model.oob_score_)

#Saved the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/modelo_senas.pkl")

print("Model Saved")