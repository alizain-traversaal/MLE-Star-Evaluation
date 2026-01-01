
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate features (X) and target (y)
# The 'class' column is the target variable
X = train_df.drop('class', axis=1)
y = train_df['class']

# Split the training data into training and validation sets
# Using a 80/20 split (test_size=0.2) for training and validation
# random_state for reproducibility
# stratify=y ensures that the proportion of target classes is the same in both train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize Random Forest Classifier as per the model description
# n_estimators=100 (number of trees) and random_state=42 for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model's accuracy on the validation set
# Accuracy is chosen as the evaluation metric as specified in the task description.
validation_accuracy = accuracy_score(y_val, y_pred_val)

# Print the final validation performance
print(f'Final Validation Performance: {validation_accuracy}')

# Load the test data for final predictions (features only)
test_df = pd.read_csv('./input/test.csv')

# Make predictions on the test set
test_predictions = model.predict(test_df)

# The task did not explicitly ask for a submission file, but typically this would be saved
# to a CSV in the specified submission format.
# For example:
# submission_df = pd.DataFrame({'class': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
