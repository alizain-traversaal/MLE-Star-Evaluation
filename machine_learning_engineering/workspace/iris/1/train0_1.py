

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate features (X) and target (y)
X = train_df.drop('class', axis=1)
y = train_df['class']

# Split the training data into training and validation sets
# Using a 80/20 split for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Determine the number of classes
num_classes = len(y.unique())

# --- Model Training ---

# 1. Initialize and Train LightGBM Classifier (from base solution)
# 'objective': 'multiclass' for multi-class classification
# 'num_class': specifies the number of target classes
# 'random_state': for reproducibility
lgb_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, random_state=42)
lgb_model.fit(X_train, y_train)

# 2. Initialize and Train Random Forest Classifier (from reference solution)
# n_estimators=100 (number of trees) and random_state=42 for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Ensemble Predictions on Validation Set ---

# Get predicted probabilities from LightGBM model
lgb_val_proba = lgb_model.predict_proba(X_val)

# Get predicted probabilities from Random Forest model
rf_val_proba = rf_model.predict_proba(X_val)

# Average the probabilities for ensembling
# This simple ensemble method averages the probability distributions
# and then selects the class with the highest average probability.
ensemble_val_proba = (lgb_val_proba + rf_val_proba) / 2

# Get the class with the highest average probability (index)
y_pred_ensemble_val_indices = np.argmax(ensemble_val_proba, axis=1)

# Map the predicted indices back to original class labels
# Both LightGBM and RandomForestClassifier store the unique sorted class labels
# in their 'classes_' attribute. They should be identical if trained on the same data.
predicted_classes_map = lgb_model.classes_
y_pred_ensemble_val = [predicted_classes_map[idx] for idx in y_pred_ensemble_val_indices]


# --- Evaluate Ensemble Accuracy on Validation Set ---
validation_accuracy = accuracy_score(y_val, y_pred_ensemble_val)

# Print the final validation performance
print(f'Final Validation Performance: {validation_accuracy}')

# --- Make Predictions on the Test Set (Ensembled) ---

# Load the test data (features only)
test_df = pd.read_csv('./input/test.csv')

# Get predicted probabilities for the test set from LightGBM
lgb_test_proba = lgb_model.predict_proba(test_df)

# Get predicted probabilities for the test set from Random Forest
rf_test_proba = rf_model.predict_proba(test_df)

# Average the probabilities for ensembling on the test set
ensemble_test_proba = (lgb_test_proba + rf_test_proba) / 2

# Get the class with the highest average probability (index)
test_predictions_indices = np.argmax(ensemble_test_proba, axis=1)

# Map the predicted indices back to original class labels
test_predictions = [predicted_classes_map[idx] for idx in test_predictions_indices]

# Create a submission DataFrame (though not explicitly asked to submit, creating it as part of a complete solution)
# submission_df = pd.DataFrame({'class': test_predictions})
# submission_df.to_csv('submission.csv', index=False)

