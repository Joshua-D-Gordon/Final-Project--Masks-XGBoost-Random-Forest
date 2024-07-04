#imports
import os
import cv2
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from treeinterpreter import treeinterpreter as ti


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

# Dataset statistics
class_counts = df['Label'].value_counts()
class_percentages = df['Label'].value_counts(normalize=True) * 100
class_balance = pd.DataFrame({'Count': class_counts, 'Percentage': class_percentages})
print("Class balance:\n", class_balance)

plt.figure(figsize=(12, 8))
sns.countplot(x='Label', data=df)
plt.title('Class Balance')
plt.xlabel('Class')
plt.ylabel('Count')
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()} ({p.get_height()/len(df)*100:.2f}%)', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), 
                       textcoords='offset points')
plt.savefig('class_balance.png')
plt.close()

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
    print("XGBoost AUC Score:", roc_auc_score(y_test, y_pred_xgb))

    # Confusion Matrix for XGBoost
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('xgb_confusion_matrix.png')
    plt.close()

    # Feature importance from XGBoost
    importances_xgb = xgb_model.feature_importances_
    indices_xgb = np.argsort(importances_xgb)[::-1]

    # Plot feature importance for XGBoost
    plt.figure(figsize=(12, 8))
    plt.title("XGBoost Feature Importance")
    plt.bar(range(X.shape[1]), importances_xgb[indices_xgb], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices_xgb], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.savefig('xgboost_feature_importance.png')
    plt.close()

    # ROC Curve for XGBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb)
    plt.figure()
    plt.plot(fpr_xgb, tpr_xgb, color='blue', lw=2, label='XGBoost ROC curve (area = %0.2f)' % auc(fpr_xgb, tpr_xgb))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('xgb_roc_curve.png')
    plt.close()

    # Train RandomForest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Evaluate RandomForest model
    y_pred_rf = rf_model.predict(X_test)
    print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("RandomForest Classification Report:\n", classification_report(y_test, y_pred_rf))
    print("RandomForest AUC Score:", roc_auc_score(y_test, y_pred_rf))

    # Confusion Matrix for RandomForest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
    plt.title('RandomForest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('rf_confusion_matrix.png')
    plt.close()

    # ROC Curve for RandomForest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    plt.figure()
    plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='RandomForest ROC curve (area = %0.2f)' % auc(fpr_rf, tpr_rf))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RandomForest ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('rf_roc_curve.png')
    plt.close()

    # Feature importance from RandomForest
    importances_rf = rf_model.feature_importances_
    indices_rf = np.argsort(importances_rf)[::-1]

    # Plot feature importance for RandomForest
    plt.figure(figsize=(12, 8))
    plt.title("RandomForest Feature Importance")
    plt.bar(range(X.shape[1]), importances_rf[indices_rf], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices_rf], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.savefig('randomforest_feature_importance.png')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Identify misclassified & correctly classified masks
    misclassified_rf = []
    misclassified_xgb = []
    correct_classified_rf = []
    correct_classified_xgb = []

    for idx, (true_label, rf_pred, xgb_pred) in enumerate(zip(y_test, y_pred_rf, y_pred_xgb)):
        if rf_pred != true_label:
            misclassified_rf.append(filenames[idx])
        else:
            correct_classified_rf.append(filenames[idx])

        if xgb_pred != true_label:
            misclassified_xgb.append(filenames[idx])
        else:
            correct_classified_xgb.append(filenames[idx])

    def extract_features_for_list(filenames, masks_path, scaler=None):
        features_list = []
        for filename in filenames:
            mask_image = cv2.imread(os.path.join(masks_path, filename), cv2.IMREAD_GRAYSCALE)
            features = extract_features(mask_image)
            if features is not None:
                features_list.append(features)
        features_df = pd.DataFrame(features_list)
        if scaler:
            features_df = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns)
        return features_df

    def plot_image_features_with_predictions_and_contributions(filename, features, xgb_pred, rf_pred, rf_contributions):
        # Load the image
        img = cv2.imread(os.path.join(masks_path, filename))
        
        # Create a figure
        plt.figure(figsize=(16, 8))
        
        # Plot the image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image - {filename}')
        plt.axis('off')
        
        # Plot the features
        plt.subplot(1, 3, 2)
        features.plot(kind='barh', legend=False)
        plt.title('Normalized Extracted Features')
        plt.xlabel('Feature Value')
        
        # Plot the contributions for each class
        plt.subplot(1, 3, 3)
        rf_contrib_summary_class0 = rf_contributions[:, 0]
        rf_contrib_summary_class1 = rf_contributions[:, 1]
        
        pd.Series(rf_contrib_summary_class0, index=features.index).plot(kind='barh', color='blue', alpha=0.6, label='Class 0')
        pd.Series(rf_contrib_summary_class1, index=features.index).plot(kind='barh', color='red', alpha=0.6, label='Class 1')
        
        plt.title('Random Forest Contributions')
        plt.xlabel('Contribution Value')
        plt.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f'{filename}_features_predictions.png')
        plt.close()

        # Print the model predictions
        print(f"Predictions for {filename}:")
        print(f"  XGBoost: {'benign' if xgb_pred == 0 else 'malignant'}")
        print(f"  RandomForest: {'benign' if rf_pred == 0 else 'malignant'}")

    # Prepare the scaler using the training set features
    all_filenames = correct_classified_rf + misclassified_rf + correct_classified_xgb + misclassified_xgb
    train_features = extract_features_for_list(all_filenames, masks_path)
    scaler = StandardScaler().fit(train_features)

    # Analyze 3 correctly classified and 3 misclassified examples for RandomForest and XGBoost
    def analyze_examples(classified_list, model, model_name):
        for i in range(min(3, len(classified_list))):
            filename = classified_list[i]
            features = extract_features_for_list([filename], masks_path, scaler)
            xgb_pred = xgb_model.predict(features)
            rf_pred = rf_model.predict(features)
            rf_predictions, rf_bias, rf_contributions = ti.predict(rf_model, features)

            plot_image_features_with_predictions_and_contributions(
                filename,
                features.iloc[0],
                xgb_pred[0],
                rf_pred[0],
                rf_contributions[0]
            )

    analyze_examples(correct_classified_rf, rf_model, "RandomForest Correctly Classified")
    analyze_examples(misclassified_rf, rf_model, "RandomForest Misclassified")
    analyze_examples(correct_classified_xgb, xgb_model, "XGBoost Correctly Classified")
    analyze_examples(misclassified_xgb, xgb_model, "XGBoost Misclassified")

else:
    print("Error: 'Label' column not found in the DataFrame.")

