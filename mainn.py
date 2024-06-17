import os
import cv2
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import shap

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

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model with Grid Search
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    }
    xgb_model = XGBClassifier(random_state=42)
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, n_jobs=-1)
    xgb_grid_search.fit(X_train_scaled, y_train)
    xgb_best_model = xgb_grid_search.best_estimator_

    # Evaluate XGBoost model
    y_pred_xgb = xgb_best_model.predict(X_test_scaled)
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

    # Train RandomForest model with Grid Search
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, n_jobs=-1)
    rf_grid_search.fit(X_train_scaled, y_train)
    rf_best_model = rf_grid_search.best_estimator_

    # Evaluate RandomForest model
    y_pred_rf = rf_best_model.predict(X_test_scaled)
    print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("RandomForest Classification Report:\n", classification_report(y_test, y_pred_rf))

    # Train Neural Network model with Grid Search
    nn_param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    nn_model = MLPClassifier(max_iter=500, random_state=42)
    nn_grid_search = GridSearchCV(nn_model, nn_param_grid, cv=5, n_jobs=-1)
    nn_grid_search.fit(X_train_scaled, y_train)
    nn_best_model = nn_grid_search.best_estimator_

    # Evaluate Neural Network model
    y_pred_nn = nn_best_model.predict(X_test_scaled)
    print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
    print("Neural Network Classification Report:\n", classification_report(y_test, y_pred_nn))

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Feature importance
    def plot_feature_importance(model, model_name):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(X.shape[1]), importance[indices], align='center')
        plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
        plt.savefig(f'feature_importance_{model_name}.png')
        plt.close()

    plot_feature_importance(rf_best_model, 'Random_Forest')
    plot_feature_importance(xgb_best_model, 'XGBoost')

    # For Neural Network, feature importance can be derived using SHAP
    explainer = shap.KernelExplainer(nn_best_model.predict, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
    plt.savefig('shap_summary_plot_nn.png')
    plt.close()

    # ROC Curves
    def plot_roc_curve(y_test, y_pred_prob, model_name):
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{model_name}.png')
        plt.close()

    # Predict probabilities for ROC curve
    y_pred_prob_rf = rf_best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_prob_xgb = xgb_best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_prob_nn = nn_best_model.predict_proba(X_test_scaled)[:, 1]

    plot_roc_curve(y_test, y_pred_prob_rf, 'Random Forest')
    plot_roc_curve(y_test, y_pred_prob_xgb, 'XGBoost')
    plot_roc_curve(y_test, y_pred_prob_nn, 'Neural Network')

    # Model Stacking
    estimators = [
        ('rf', rf_best_model),
        ('xgb', xgb_best_model),
        ('nn', nn_best_model)
    ]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(random_state=42))
    stacking_model.fit(X_train_scaled, y_train)

    # Evaluate Stacking model
    y_pred_stack = stacking_model.predict(X_test_scaled)
    y_pred_prob_stack = stacking_model.predict_proba(X_test_scaled)[:, 1]
    print("Stacking Model Accuracy:", accuracy_score(y_test, y_pred_stack))
    print("Stacking Model Classification Report:\n", classification_report(y_test, y_pred_stack))

    # ROC Curve for Stacking model
    plot_roc_curve(y_test, y_pred_prob_stack, 'Stacking Model')

else:
    print("Error: 'Label' column not found in the DataFrame.")


