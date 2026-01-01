
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os # Import os to create the directory if it doesn't exist

# --- Python Solution 1 Modified ---

# Load the training and test data
train_df_model1 = pd.read_csv('./input/train.csv')
test_df_model1 = pd.read_csv('./input/test.csv')

# Separate features (X) and target (y) from the training data
X_model1 = train_df_model1.drop('class', axis=1)
y_model1 = train_df_model1['class']

# Split the training data into training and validation sets
# Using a 20% validation set with stratification to maintain class distribution

X_train_model1, X_val_model1, y_train_model1, y_val_model1 = train_test_split(X_model1, y_model1, test_size=0.2, random_state=42)


# Determine the number of unique classes for the LightGBM model
num_classes_model1 = len(np.unique(y_train_model1))

# Initialize LightGBM Classifier
# 'objective': 'multiclass' and 'num_class' are crucial for multi-class classification
model1 = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes_model1, random_state=42)

# Train the model on the training set
model1.fit(X_train_model1, y_train_model1)

# Make predictions on the validation set
y_pred_val_model1 = model1.predict(X_val_model1)

# Evaluate the model's accuracy on the validation set
accuracy_model1 = accuracy_score(y_val_model1, y_pred_val_model1)

# Prepare the test data for submission
X_test_submission_model1 = test_df_model1.copy()

# Make predictions on the actual test set
test_predictions_model1 = model1.predict(X_test_submission_model1)

# Save predicted probabilities for the test set
model1_test_probabilities = model1.predict_proba(X_test_submission_model1)
np.save('model1_test_probabilities.npy', model1_test_probabilities)

# Save validation accuracy
np.save('model1_val_accuracy.npy', accuracy_model1)


# --- Python Solution 2 Modified ---

# Load the training data
train_df_model2 = pd.read_csv('./input/train.csv')
test_df_model2 = pd.read_csv('./input/test.csv') # Added for test set prediction

# Separate features (X) and target (y)
# Assuming the target column is named 'class' based on common Kaggle Iris datasets and submission format.
X_model2 = train_df_model2.drop('class', axis=1)
y_model2 = train_df_model2['class']

# Split the training data into training and validation sets
# Using a 20% validation split, consistent with common practices and the example code.
# random_state is set for reproducibility.

X_train_model2, X_val_model2, y_train_model2, y_val_model2 = train_test_split(X_model2, y_model2, test_size=0.2, random_state=42)


# Initialize the LightGBM Classifier
# objective='multiclass' for multi-class classification.
# num_class is set to 3 as there are three iris species.
# random_state is set for reproducibility.
model2 = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)

# Train the model on the training set
model2.fit(X_train_model2, y_train_model2)

# Make predictions on the validation set
y_pred_val_model2 = model2.predict(X_val_model2)

# Evaluate the model's accuracy on the validation set
# Accuracy is a reasonable metric for this classification task as requested.
accuracy_model2 = accuracy_score(y_val_model2, y_pred_val_model2)

# Prepare the test data for submission
X_test_submission_model2 = test_df_model2.copy() # Use the loaded test_df_model2

# Save predicted probabilities for the test set
model2_test_probabilities = model2.predict_proba(X_test_submission_model2)
np.save('model2_test_probabilities.npy', model2_test_probabilities)

# Save validation accuracy
np.save('model2_val_accuracy.npy', accuracy_model2)

# --- Ensemble Script ---

# Load the saved data
model1_test_probabilities = np.load('model1_test_probabilities.npy')
model2_test_probabilities = np.load('model2_test_probabilities.npy')
model1_val_accuracy = np.load('model1_val_accuracy.npy')
model2_val_accuracy = np.load('model2_val_accuracy.npy')

# Calculate normalized weights for each model based on their validation accuracies
total_accuracy = model1_val_accuracy + model2_val_accuracy
weight1 = model1_val_accuracy / total_accuracy
weight2 = model2_val_accuracy / total_accuracy

# Calculate the element-wise weighted average of the two probability arrays
weighted_averaged_probabilities = (weight1 * model1_test_probabilities) + \
                                  (weight2 * model2_test_probabilities)

# Determine the final ensemble predictions
ensemble_predictions = np.argmax(weighted_averaged_probabilities, axis=1)

# Create a pandas DataFrame for the submission
submission_df_ensemble = pd.DataFrame({'class': ensemble_predictions})

# Create the ./final directory if it doesn't exist
os.makedirs('./final', exist_ok=True)

# Save the ensemble submission file to the ./final directory
submission_df_ensemble.to_csv('./final/submission.csv', index=False)

# Calculate and print the final validation performance (weighted average of individual model accuracies)
final_validation_score = (weight1 * model1_val_accuracy) + (weight2 * model2_val_accuracy)
print(f"Final Validation Performance: {final_validation_score}")
