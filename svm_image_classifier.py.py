import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset path
DATASET_PATH = "train"

# Image size
IMG_SIZE = 64

def main():
    # 1. Check if the directory exists to prevent FileNotFoundError
    if not os.path.exists(DATASET_PATH):
        print(f"Error: The directory '{DATASET_PATH}' does not exist.")
        sys.exit(1)

    images = []
    labels = []

    print("Loading images...")
    
    files = os.listdir(DATASET_PATH)
    
    # 2. Check if directory is empty
    if not files:
        print(f"Error: No files found in '{DATASET_PATH}'.")
        sys.exit(1)

    # Use first 4000 images for faster training
    for file in tqdm(files[:4000]):
        path = os.path.join(DATASET_PATH, file)

        # Read image
        img = cv2.imread(path)

        # Skip if the image is unreadable, corrupted, or a non-image file
        if img is None:
            continue

        try:
            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Flatten image for SVM
            img = img.flatten()

            images.append(img)

            # Label generation (0 for cat, 1 for dog/other)
            if "cat" in file.lower():
                labels.append(0)
            else:
                labels.append(1)
                
        except Exception as e:
            # Prevent single corrupted files from crashing the whole loop
            print(f"\nWarning: Could not process {file}. Reason: {e}")
            continue

    # 3. Check if any valid images were actually loaded before proceeding
    if len(images) == 0:
        print("Error: No valid images could be processed. Exiting.")
        sys.exit(1)

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    print("Dataset Shape:", X.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training SVM...")
    # Initialize and train the model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    print("Predicting...")
    # Make predictions
    predictions = model.predict(X_test)

    # Calculate and display metrics
    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()