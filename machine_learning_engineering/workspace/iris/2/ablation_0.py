
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Define file paths
TRAIN_FILE = './input/train.csv'

# Load datasets
# As per instructions, data is guaranteed to be in './input' and no error handling for FileNotFoundError is needed.
train_df = pd.read_csv(TRAIN_FILE)

# Prepare data
# The target variable is 'class'. All other columns are features, excluding 'id' if present.
feature_columns = [col for col in train_df.columns if col not in ['id', 'class']]

X = train_df[feature_columns]
y = train_df['class']

# Determine the number of unique classes for XGBoost
num_classes = y.nunique()

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest Classifier (from base solution)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize the XGBoost Classifier (from reference solution)
model_xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# --- Baseline: Original Ensembled Model Performance ---
print("--- Baseline: Original Ensembled Model ---")
# Train both models on the training data
model_rf.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)

# Make probability predictions on the validation set for both models
y_val_proba_rf = model_rf.predict_proba(X_val)
y_val_proba_xgb = model_xgb.predict_proba(X_val)

# Ensemble the predictions by averaging probabilities
# Then, convert averaged probabilities back to class labels
y_val_proba_ensemble = (y_val_proba_rf + y_val_proba_xgb) / 2
y_val_pred_ensemble = np.argmax(y_val_proba_ensemble, axis=1)

# Evaluate the ensembled model using the accuracy metric
baseline_accuracy = accuracy_score(y_val, y_val_pred_ensemble)
print(f"Validation Accuracy (Baseline - Ensembled): {baseline_accuracy:.4f}")

# --- Ablation 1: Only Random Forest Classifier ---
print("\n--- Ablation 1: Only Random Forest Classifier ---")
# The Random Forest model is already trained from the baseline step
y_val_pred_rf_only = model_rf.predict(X_val)
rf_only_accuracy = accuracy_score(y_val, y_val_pred_rf_only)
print(f"Validation Accuracy (Ablation - RF Only): {rf_only_accuracy:.4f}")

# --- Ablation 2: Only XGBoost Classifier ---
print("\n--- Ablation 2: Only XGBoost Classifier ---")
# The XGBoost model is already trained from the baseline step
y_val_pred_xgb_only = model_xgb.predict(X_val)
xgb_only_accuracy = accuracy_score(y_val, y_val_pred_xgb_only)
print(f"Validation Accuracy (Ablation - XGBoost Only): {xgb_only_accuracy:.4f}")

# --- Determine the most contributing part ---
results = {
    "Baseline (Ensembled RF + XGBoost)": baseline_accuracy,
    "RandomForest Classifier Only": rf_only_accuracy,
    "XGBoost Classifier Only": xgb_only_accuracy
}

# Find the configuration with the highest accuracy
best_performance_label = max(results, key=results.get)
max_accuracy = results[best_performance_label]

if best_performance_label == "Baseline (Ensembled RF + XGBoost)":
    print(f"\nThe part that contributes most to the overall performance is the ensembling of RandomForest and XGBoost models, achieving {max_accuracy:.4f} accuracy.")
elif best_performance_label == "RandomForest Classifier Only":
    print(f"\nThe part that contributes most to the overall performance is the RandomForest Classifier alone, achieving {max_accuracy:.4f} accuracy.")
else: # XGBoost Classifier Only
    print(f"\nThe part that contributes most to the overall performance is the XGBoost Classifier alone, achieving {max_accuracy:.4f} accuracy.")
