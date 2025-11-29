import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

DATASET_PATH = "datasets"

# Load MobileNet for feature extraction
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

data = []
labels = []

# Loop through dataset
for split in ["train", "validation"]:
    split_path = os.path.join(DATASET_PATH, split)

    for class_name in os.listdir(split_path):
        class_folder = os.path.join(split_path, class_name)

        if not os.path.isdir(class_folder):
            continue

        print(f"Processing → {split}/{class_name}")

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)

            img = cv2.imread(img_path)
            if img is None:
                print("Skipped:", img_path)
                continue

            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, 0)
            img = preprocess_input(img)

            features = feature_extractor.predict(img)[0]

            data.append(features)
            labels.append(class_name)

# Convert
data = np.array(data)
labels = np.array(labels)

print("\nTotal samples:", len(data))
print("Features shape:", data.shape)
print("Classes:", set(labels))

# Train classifier
clf = RandomForestClassifier(n_estimators=400, max_depth=40)
clf.fit(data, labels)

# Save model
joblib.dump(clf, "fingernail_ml_model_AI.joblib")
print("\n✔ Training completed!")
print("✔ Saved as fingernail_ml_model_AI.joblib")
