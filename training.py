import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

from features import extract_features

DATASET_DIR = "dataset_landmarks"
MODEL_PATH = "model/modelo_senas.pkl"

X = [] #Feature vector of each sample in the training set (normalized coordinates + geometric features)
y = [] #Letter correspond to each sample in X

#Iterates in the training set, extracts the features of each sample using the shared function from
#features.py and append it to X and Y (Features, Letter) to later train the model. Using the same
#function as main.py guarantees that the model receives the same features in training and in real time
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
                X.append(extract_features(sample))
                y.append(label)

X = np.array(X)
y = np.array(y)

print("Data uploaded", X.shape)

#Random Forest with 500 estimators, trained on the FULL dataset. The real performance of this
#configuration is measured with evaluate.py; here we use all the data to get the best final model
model = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)

#The OOB accuracy is optimistic: consecutive frames are near-duplicates, so it leaks (see evaluate.py)
print("OOB accuracy (optimistic, see evaluate.py) = ", model.oob_score_)

#Saved the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Model Saved:", MODEL_PATH, "| features per sample:", model.n_features_in_)