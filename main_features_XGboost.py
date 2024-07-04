import os
import cv2
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
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
    if len(contours) == 0:
        return None
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
    try:
        xgb_model = XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)

        # Evaluate XGBoost model
        y_pred_xgb = xgb_model.predict(X_test)
        print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
        print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
else:
    print("Error: 'Label' column not found in the DataFrame.")

# Choose a specific mask file for feature visualization
specific_mask_file = 'BUSBRA-000001-r.png'  # Replace with your specific mask filename

# Extract features from the specific mask
if not os.path.exists(os.path.join(masks_path, specific_mask_file)):
    print(f"Error: The file {specific_mask_file} does not exist.")
else:
    mask_image = cv2.imread(os.path.join(masks_path, specific_mask_file), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        print(f"Error: Could not read the file {specific_mask_file}.")
    else:
        features = extract_features(mask_image)
        if features is None:
            print(f"Error: No valid features extracted from the file {specific_mask_file}.")
        else:
            # Display the mask
            plt.figure(figsize=(6, 6))
            plt.imshow(mask_image, cmap='gray')
            plt.title('Mask Image\n(Mask filename: {})'.format(specific_mask_file))
            plt.axis('off')
            plt.show()

            # Convert features to DataFrame
            features_df = pd.DataFrame([features])

            # Scale features using the same scaler fitted on the training data
            scaler = StandardScaler()
            scaler.fit(X)
            features_scaled = scaler.transform(features_df)

            # Get feature importances from the trained XGBoost model
            try:
                feature_importances = xgb_model.feature_importances_

                # Create a DataFrame for feature importances
                importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

                # Sort by importance
                importances_df = importances_df.sort_values(by='Importance', ascending=False)

                # Plot XGBoost feature importances as a bar chart
                plt.figure(figsize=(12, 6))
                plt.barh(importances_df['Feature'], importances_df['Importance'])
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title(f'XGBoost Feature Importances - {specific_mask_file}')
                plt.xticks(rotation=45)
                plt.show()

                # Plot the specific mask's feature values
                features_df_scaled = pd.DataFrame(features_scaled, columns=features_df.columns)
                plt.figure(figsize=(12, 6))
                plt.barh(features_df_scaled.columns, features_df_scaled.iloc[0])
                plt.xlabel('Scaled Feature Value')
                plt.ylabel('Feature')
                plt.title(f'Scaled Feature Values for {specific_mask_file}')
                plt.xticks(rotation=45)
                plt.show()
            except Exception as e:
                print(f"Error extracting feature importances from XGBoost model: {e}")
