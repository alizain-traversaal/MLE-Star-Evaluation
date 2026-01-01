
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
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
# Identify feature columns by excluding 'class' and 'id' (if it exists in train_df).
feature_columns = [col for col in train_df.columns if col not in ['id', 'class']]

X = train_df[feature_columns]
y = train_df['class']

# Determine the number of unique classes for XGBoost
num_classes = y.nunique()

# Split the training data into training and validation sets
# A test_size of 0.2 (20% for validation) is a common choice.
# stratify=y ensures that the proportion of target classes is maintained in both
# the training and validation sets, which is good practice for classification tasks.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create an XGBoost Classifier as described in the model description
# For multi-class classification, objective='multi:softmax' and num_class should be specified.
# eval_metric='mlogloss' is a common metric for multi-class and use_label_encoder=False suppresses a warning.
model = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model using accuracy, as it's a suitable metric for this multi-class classification task.
validation_accuracy = accuracy_score(y_val, y_val_pred)

# Print the final validation performance in the required format
print(f"Final Validation Performance: {validation_accuracy}")

# Prepare test data for prediction
# Ensure test_df contains only the feature columns used during training.
test_features_df = test_df[feature_columns]

# Make predictions on the actual test dataset
test_predictions = model.predict(test_features_df)

# Create the submission file in the specified format
submission_df = pd.DataFrame({'class': test_predictions})
submission_df.to_csv(SUBMISSION_FILE, index=False)
