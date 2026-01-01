
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Define file paths
TRAIN_FILE = './input/train.csv'
TEST_FILE = './input/test.csv'
SUBMISSION_FILE = 'submission.csv'

# Load datasets
# As per instructions, data is guaranteed to be in './input' and no error handling for FileNotFoundError is needed.
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Prepare data
# The target variable is 'class'. All other columns are features, excluding 'id' if present.
feature_columns = [col for col in train_df.columns if col not in ['id', 'class']]

X = train_df[feature_columns]
y = train_df['class']

# Determine the number of unique classes for XGBoost
num_classes = y.nunique()

# Split the training data into training and validation sets
# A test_size of 0.2 (20% for validation) is a common choice.
# stratify=y ensures that the proportion of target classes is the same in both
# the training and validation sets, which is good practice for classification.
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
validation_accuracy = accuracy_score(y_val, y_val_pred_ensemble)

# Print the final validation performance
print(f"Final Validation Performance: {validation_accuracy}")

# Prepare test data for prediction
# Ensure test_df contains only the feature columns used during training
test_features_df = test_df[feature_columns]

# Make probability predictions on the actual test dataset for both models
test_proba_rf = model_rf.predict_proba(test_features_df)
test_proba_xgb = model_xgb.predict_proba(test_features_df)

# Ensemble the test predictions by averaging probabilities
test_proba_ensemble = (test_proba_rf + test_proba_xgb) / 2
test_predictions_ensemble = np.argmax(test_proba_ensemble, axis=1)

# Create the submission file in the specified format
submission_df = pd.DataFrame({'class': test_predictions_ensemble})
submission_df.to_csv(SUBMISSION_FILE, index=False)
