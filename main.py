#imports
import os
import cv2
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define paths
masks_path = 'masks'
meta_path = 'meta'

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
        major_axis_length = minor_axis_length = 0
    
    features['Major Axis Length'] = major_axis_length
    features['Minor Axis Length'] = minor_axis_length
    
    # Aspect Ratio
    aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0
    features['Aspect Ratio'] = aspect_ratio
    
    # Circularity
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
    features['Circularity'] = circularity
    
    # Compactness
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area != 0 else 0
    features['Compactness'] = compactness
    
    # Eccentricity
    eccentricity = major_axis_length / minor_axis_length if minor_axis_length != 0 else 0
    features['Eccentricity'] = eccentricity
    
    # Jaggedness (Boundary Roughness)
    smoothed_contour = cv2.approxPolyDP(contours[0], epsilon=0.02*perimeter, closed=True)
    smoothed_perimeter = cv2.arcLength(smoothed_contour, True)
    jaggedness = perimeter / smoothed_perimeter if smoothed_perimeter != 0 else 0
    features['Jaggedness'] = jaggedness
    
    # Convex Hull Area
    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull)
    features['Convex Hull Area'] = hull_area
    
    # Solidity
    solidity = area / hull_area if hull_area != 0 else 0
    features['Solidity'] = solidity
    
    return features

def extract_features_from_all_masks(masks_path):
    features_list = []
    filenames = []
    
    for mask_file in os.listdir(masks_path):
        if mask_file.endswith('.png'):
            mask_image = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
            features = extract_features(mask_image)
            features_list.append(features)
            filenames.append(mask_file)
    
    return features_list, filenames

def load_metadata(meta_path):
    labels = {}
    
    for meta_file in os.listdir(meta_path):
        if meta_file.endswith('.json'):
            with open(os.path.join(meta_path, meta_file), 'r') as f:
                meta_data = json.load(f)
                labels[meta_file.replace('.json', '.png')] = meta_data.get('Class', 'Unknown')
    
    return labels

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

# Debug print statements to verify mappings
print("Features List Length:", len(features_list))
print("Filenames Length:", len(filenames))
# Create dataset
dataset = create_dataset(features_list, filenames, labels)

# Debug print statements to verify the final dataset
print("Dataset Length:", len(dataset))
print("\n")
print("Sample from Dataset:", dataset[:5])  # Print first 5 entries

# Convert to DataFrame for better visualization and manipulation
df = pd.DataFrame(dataset)
print("\n")
print(df.head())

# Encode labels as numerical values
label_mapping = {'benign': 0, 'malignant': 1}
df['Label'] = df['Label'].map(label_mapping)

# Prepare data for training
if 'Label' in df.columns:
    X = df.drop('Label', axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Evaluate XGBoost model
    y_pred_xgb = xgb_model.predict(X_test)
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

    # Train RandomForest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Evaluate RandomForest model
    y_pred_rf = rf_model.predict(X_test)
    print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("RandomForest Classification Report:\n", classification_report(y_test, y_pred_rf))
else:
    print("Error: 'Label' column not found in the DataFrame.")
