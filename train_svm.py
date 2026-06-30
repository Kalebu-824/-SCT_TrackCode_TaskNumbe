import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Dataset path
dataset_path = "."
print("Current folder:", os.getcwd())
print("Files/Folders:", os.listdir(dataset_path))

# Two gesture categories
categories = ["thumbs_up", "peace"]

images = []
labels = []

print("Loading dataset...")

# Check if dataset folder exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset folder '{dataset_path}' not found.")
    exit()

# Load images
for category in categories:
    path = os.path.join(dataset_path, category)

    if not os.path.exists(path):
        print(f"Warning: Folder '{category}' not found. Skipping...")
        continue

    label = categories.index(category)

    for img_name in os.listdir(path):

        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))
        img = img.flatten()

        images.append(img)
        labels.append(label)

X = np.array(images, dtype=np.float32)
y = np.array(labels)

print(f"Total images loaded: {len(X)}")

if len(X) == 0:
    print("No images found!")
    exit()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, random_state=42))
])

print("\nTraining SVM model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n===== Model Evaluation =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=categories
))

# Save model
joblib.dump(model, "gesture_model.pkl")

print("\nModel saved successfully as 'gesture_model.pkl'")