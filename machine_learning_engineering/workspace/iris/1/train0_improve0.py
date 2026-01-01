
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate features (X) and target (y)
X = train_df.drop('class', axis=1)
y = train_df['class']

# Split the training data into training and validation sets
# Using a 80/20 split for training and validation

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Determine the number of classes
num_classes = len(y.unique())

# Initialize LightGBM Classifier
# 'objective': 'multiclass' for multi-class classification
# 'num_class': specifies the number of target classes
# 'random_state': for reproducibility
model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate accuracy on the validation set
validation_accuracy = accuracy_score(y_val, y_pred_val)

# Print the final validation performance
print(f'Final Validation Performance: {validation_accuracy}')

# Load the test data (features only)
test_df = pd.read_csv('./input/test.csv')

# Make predictions on the test set
test_predictions = model.predict(test_df)

# Create a submission DataFrame (though not explicitly asked to submit, creating it as part of a complete solution)
# submission_df = pd.DataFrame({'class': test_predictions})
# submission_df.to_csv('submission.csv', index=False)
