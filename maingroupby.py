import os
import cv2
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths and datasets
masks_path = 'masks'
meta_path = 'meta'
datasets = ['BrEaST', 'BUSBRA', 'BUSI', 'OASBUD', 'UDIAT']

# Store features and labels for each dataset separately
dataset_features = {dataset: [] for dataset in datasets}
dataset_labels = {dataset: [] for dataset in datasets}

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
        # no 5 points to fit an eclipse
        return None
    
    #aviod divide by zeros error and remove instance from data
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
        #skip instance
        return None
    features['Circularity'] = circularity
    
    # Compactness
    if area != 0:
        compactness = (perimeter ** 2) / (4 * np.pi * area)
    else:
        #skip image to avoid divide by zero
        return None
    features['Compactness'] = compactness
    
    # Eccentricity
    eccentricity = major_axis_length / minor_axis_length
    features['Eccentricity'] = eccentricity
    
    # Jaggedness (Boundary Roughness)
    smoothed_contour = cv2.approxPolyDP(contours[0], epsilon=0.02*perimeter, closed=True)
    smoothed_perimeter = cv2.arcLength(smoothed_contour, True)
    if smoothed_perimeter != 0:
        jaggedness = perimeter / smoothed_perimeter
    else:
        #skip image to avoid divide by zero
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

def extract_features_from_dataset(dataset_name, masks_path, meta_path):
    features_list = []
    filenames = []
    labels = []

    for mask_file in os.listdir(masks_path):
        if mask_file.startswith(dataset_name):
            if mask_file.endswith('.png'):
                mask_image = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
                features = extract_features(mask_image)
                if features is not None:
                    features_list.append(features)
                    filenames.append(mask_file)
                    json_file = mask_file.replace('.png', '.json')
                    with open(os.path.join(meta_path, json_file), 'r') as f:
                        meta_data = json.load(f)
                        label = meta_data.get('Class', 'Unknown')
                        labels.append(label)

    return features_list, filenames, labels
    
    
# Extract features and labels from each dataset
for dataset in datasets:
    features_list, filenames, labels = extract_features_from_dataset(dataset, masks_path, meta_path)
    dataset_features[dataset] = features_list
    dataset_labels[dataset] = labels

# Train and evaluate separate models for each dataset
for dataset in datasets:
    features = dataset_features[dataset]
    labels = dataset_labels[dataset]

    # Convert to DataFrame
    df = pd.DataFrame(features)
    df['Label'] = labels

    # Encode labels as numerical values
    label_mapping = {'benign': 0, 'malignant': 1}
    df['Label'] = df['Label'].map(label_mapping)

    # Remove rows with NaN labels
    df = df.dropna(subset=['Label'])

    X = df.drop('Label', axis=1)
    y = df['Label']
    
    print(f"Dataset {dataset} size: {len(df)}") 
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train_scaled, y_train)

    # Evaluate XGBoost model
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    print(f"{dataset} - XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print(f"{dataset} - XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

    # Train RandomForest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate RandomForest model
    y_pred_rf = rf_model.predict(X_test_scaled)
    print(f"{dataset} - RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(f"{dataset} - RandomForest Classification Report:\n", classification_report(y_test, y_pred_rf))
