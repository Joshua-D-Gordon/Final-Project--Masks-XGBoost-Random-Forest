import os
import cv2
import numpy as np
import pandas as pd
import json  # Add this line for importing json module
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Define paths
masks_path = 'masks'
meta_path = 'meta'


# Function to extract features from masks
def extract_features(mask):
    features = {}

    # Area
    area = cv2.countNonZero(mask)
    features['Area'] = area

    # Perimeter
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True)
    features['Perimeter'] = perimeter

    # Major and Minor Axis Lengths
    if len(contours[0]) >= 5:
        ellipse = cv2.fitEllipse(contours[0])
        (center, axes, orientation) = ellipse
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
    else:
        # no 5 points to fit an ellipse
        return None

    # Avoid divide by zeros error and remove instance from data
    if minor_axis_length == 0:
        return None

    features['Major Axis Length'] = major_axis_length
    features['Minor Axis Length'] = minor_axis_length

    # Aspect Ratio
    aspect_ratio = major_axis_length / minor_axis_length
    features['Aspect Ratio'] = aspect_ratio

    # Circularity
    if perimeter != 0:
        circularity = 4 * np.pi * (area / (perimeter ** 2))
    else:
        # Skip instance
        return None
    features['Circularity'] = circularity

    # Compactness
    if area != 0:
        compactness = (perimeter ** 2) / (4 * np.pi * area)
    else:
        # Skip image to avoid divide by zero
        return None
    features['Compactness'] = compactness

    # Eccentricity
    eccentricity = major_axis_length / minor_axis_length
    features['Eccentricity'] = eccentricity

    # Jaggedness (Boundary Roughness)
    smoothed_contour = cv2.approxPolyDP(contours[0], epsilon=0.02 * perimeter, closed=True)
    smoothed_perimeter = cv2.arcLength(smoothed_contour, True)
    if smoothed_perimeter != 0:
        jaggedness = perimeter / smoothed_perimeter
    else:
        # Skip image to avoid divide by zero
        return None
    features['Jaggedness'] = jaggedness

    # Convex Hull Area
    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull)
    if hull_area != 0:
        solidity = area / hull_area
    else:
        return None  # Skip this image to avoid divide-by-zero

    features['Convex Hull Area'] = hull_area
    features['Solidity'] = solidity

    return features


# Function to extract features from all masks in a directory
def extract_features_from_all_masks(masks_path):
    features_list = []
    filenames = []

    for mask_file in os.listdir(masks_path):
        if mask_file.endswith('.png'):
            mask_image = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
            features = extract_features(mask_image)
            if features is not None:
                features_list.append(features)
                filenames.append(mask_file)

    return features_list, filenames


# Function to load metadata
def load_metadata(meta_path):
    labels = {}

    for meta_file in os.listdir(meta_path):
        if meta_file.endswith('.json'):
            with open(os.path.join(meta_path, meta_file), 'r') as f:
                meta_data = json.load(f)
                labels[meta_file.replace('.json', '.png')] = meta_data.get('Class', 'Unknown')

    return labels


# Function to create dataset
def create_dataset(features_list, filenames, labels):
    dataset = []
    for features, filename in zip(features_list, filenames):
        label = labels.get(filename, None)
        if label is not None:
            features['Label'] = label
            dataset.append(features)
    return dataset


# Extract features from all masks
features_list, filenames = extract_features_from_all_masks(masks_path)

# Load metadata
labels = load_metadata(meta_path)

# Create dataset
dataset = create_dataset(features_list, filenames, labels)

# Convert to DataFrame for better visualization and manipulation
df = pd.DataFrame(dataset)

# Encode labels as numerical values
label_mapping = {'benign': 0, 'malignant': 1}
df['Label'] = df['Label'].map(label_mapping)

# Prepare data for training
if 'Label' in df.columns:
    X = df.drop('Label', axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Train XGBoost model
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)

    # Evaluate RandomForest model
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print("RandomForest Accuracy:", rf_accuracy)

    # Evaluate XGBoost model
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    print("XGBoost Accuracy:", xgb_accuracy)

    # Identify misclassified masks
    misclassified_rf = []
    misclassified_xgb = []

    for idx, (true_label, rf_pred, xgb_pred) in enumerate(zip(y_test, y_pred_rf, y_pred_xgb)):
        if rf_pred != true_label:
            misclassified_rf.append(filenames[idx])
        if xgb_pred != true_label:
            misclassified_xgb.append(filenames[idx])

    print("\nRandomForest misclassified masks:")
    print(misclassified_rf)

    print("\nXGBoost misclassified masks:")
    print(misclassified_xgb)

else:
    print("Error: 'Label' column not found in the DataFrame.")
