
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

# --- Python Solution 1 Modified ---

# Load the training and test data
train_df_s1 = pd.read_csv('./input/train.csv')
test_df_s1 = pd.read_csv('./input/test.csv')

# Separate features (X) and target (y) from the training data
X_s1 = train_df_s1.drop('class', axis=1)
y_s1 = train_df_s1['class']

# Split the training data into training and validation sets
# Using a 20% validation set with stratification to maintain class distribution
# For consistent ensemble validation, this split will be used by both models.
X_train, X_val, y_train, y_val = train_test_split(X_s1, y_s1, test_size=0.2, random_state=42)

# Determine the number of unique classes for the LightGBM model
num_classes_s1 = len(np.unique(y_train))

# Initialize LightGBM Classifier
model_s1 = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes_s1, random_state=42)

# Train the model on the training set
model_s1.fit(X_train, y_train)

# Prepare the test data for submission
X_test_submission_s1 = test_df_s1.copy()

# Get predicted probabilities for validation set (for ensemble validation)
model1_val_probabilities = model_s1.predict_proba(X_val)

# Save predicted probabilities for the test set as per ensemble plan
model1_test_probabilities = model_s1.predict_proba(X_test_submission_s1)
np.save('model1_test_probabilities.npy', model1_test_probabilities)

# Original submission logic (output will be overwritten by ensemble submission)
test_predictions_s1 = model_s1.predict(X_test_submission_s1)
submission_df_s1 = pd.DataFrame({'class': test_predictions_s1})
submission_df_s1.to_csv('submission_s1_temp.csv', index=False) # Temporarily save, will be cleaned up


# --- Python Solution 2 Modified ---

# Load the training data
train_df_s2 = pd.read_csv('./input/train.csv')
# MODIFICATION 2.1: Add test_df loading as per ensemble plan
test_df_s2 = pd.read_csv('./input/test.csv')

# Separate features (X) and target (y)
X_s2 = train_df_s2.drop('class', axis=1)
y_s2 = train_df_s2['class']

# The training and validation split (X_train, X_val, y_train, y_val) is reused from Solution 1.

# Initialize the LightGBM Classifier
model_s2 = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)

# Train the model on the training set
model_s2.fit(X_train, y_train)

# Get predicted probabilities for validation set (for ensemble validation)
model2_val_probabilities = model_s2.predict_proba(X_val)

# Prepare the test data for submission
# MODIFICATION 2.2: Use loaded test_df
X_test_submission_s2 = test_df_s2.copy()

# MODIFICATION 2.3: Make predictions on the actual test set and save probabilities
model2_test_probabilities = model_s2.predict_proba(X_test_submission_s2)
np.save('model2_test_probabilities.npy', model2_test_probabilities)

# MODIFICATION 2.4: Remove or comment out the print statement for validation accuracy
# (The final ensemble validation performance will be printed instead)
# print(f"Final Validation Performance: {accuracy}")


# --- Ensembling Script ---

# Load the saved test probabilities from disk as per ensemble plan
loaded_model1_test_probabilities = np.load('model1_test_probabilities.npy')
loaded_model2_test_probabilities = np.load('model2_test_probabilities.npy')

# Calculate the element-wise average of these two probability arrays for the test set
averaged_test_probabilities = (loaded_model1_test_probabilities + loaded_model2_test_probabilities) / 2

# Determine the final ensemble predictions for the test set
ensemble_test_predictions = np.argmax(averaged_test_probabilities, axis=1)

# Create a pandas DataFrame for the submission
submission_df_ensemble = pd.DataFrame({'class': ensemble_test_predictions})

# Save the ensemble submission file
submission_df_ensemble.to_csv('submission.csv', index=False)

# Clean up temporary files created during the process
os.remove('model1_test_probabilities.npy')
os.remove('model2_test_probabilities.npy')
os.remove('submission_s1_temp.csv')


# --- Calculate and print Ensemble Validation Performance ---
# Calculate the element-wise average of validation probabilities
averaged_val_probabilities = (model1_val_probabilities + model2_val_probabilities) / 2

# Determine the final ensemble predictions for the validation set
ensemble_val_predictions = np.argmax(averaged_val_probabilities, axis=1)

# Evaluate the ensemble model's accuracy on the validation set
final_validation_score = accuracy_score(y_val, ensemble_val_predictions)

# Print the final ensemble validation performance
print(f"Final Validation Performance: {final_validation_score}")
