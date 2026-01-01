
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Solution 1
# Load the data
data = pd.read_csv('./input/train.csv')
X = data.drop(['quality'], axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize XGBoost regressor
xgbr = xgb.XGBRegressor(objective='reg:squarederror', # Specify squared error for regression
                            n_estimators=500,  # Number of boosting rounds
                            learning_rate=0.05, # Step size shrinkage 
                            max_depth=5,        # Maximum depth of a tree
                            min_child_weight=1, # Minimum sum of instance weight needed in a child
                            gamma=0,            # Minimum loss reduction required to make a further partition
                            subsample=0.8,      # Subsample ratio of the training instance
                            colsample_bytree=0.8, # Subsample ratio of columns when constructing each tree
                            reg_alpha=0.005,      # L1 regularization term on weights
                            random_state=42,    # Random seed
                            n_jobs=-1)          # Use all available cores


# Train the model
xgbr.fit(X_train, y_train)

# Make predictions
predictions_1_val = xgbr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions_1_val)
rmse_1 = np.sqrt(mse)
print(f'Solution 1 Root Mean Squared Error: {rmse_1}')

# Make predictions on the test data and save
test_data = pd.read_csv('./input/test.csv')
predictions_1_test = xgbr.predict(test_data)
pd.DataFrame(predictions_1_test).to_csv('solution1_predictions.csv', index=False)


# Solution 2
# Load the dataset
data = pd.read_csv('./input/train.csv')

# Split into features (X) and target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the XGBoost Regressor
xgb_2 = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
xgb_2.fit(X_train, y_train)


# Make predictions
predictions_2_val = xgb_2.predict(X_test)

# Evaluate the model
rmse_2 = mean_squared_error(y_test, predictions_2_val)
rmse_2 = rmse_2**0.5
print(f'Solution 2 RMSE: {rmse_2}')

# Make predictions on the test set and save
test_data = pd.read_csv('./input/test.csv')
predictions_2_test = xgb_2.predict(test_data)
pd.DataFrame(predictions_2_test).to_csv('solution2_predictions.csv', index=False)

# Ensemble
# Load test predictions
predictions_1_test = pd.read_csv('solution1_predictions.csv').values.flatten()
predictions_2_test = pd.read_csv('solution2_predictions.csv').values.flatten()

# Simple Averaging
simple_average_predictions = (predictions_1_test + predictions_2_test) / 2

# Evaluate simple averaging on validation data
simple_average_val_predictions = (predictions_1_val + predictions_2_val) / 2
rmse_simple_average = np.sqrt(mean_squared_error(y_test, simple_average_val_predictions))
print(f'Simple Average RMSE: {rmse_simple_average}')


# Weighted Averaging
best_w = None
best_rmse = float('inf')

for w in np.arange(0, 1.01, 0.01):
    weighted_average_val_predictions = w * predictions_1_val + (1 - w) * predictions_2_val
    rmse_weighted_average = np.sqrt(mean_squared_error(y_test, weighted_average_val_predictions))
    if rmse_weighted_average < best_rmse:
        best_rmse = rmse_weighted_average
        best_w = w

print(f'Best weight: {best_w}')

# Create weighted average predictions on test data using best weight
weighted_average_predictions = best_w * predictions_1_test + (1 - best_w) * predictions_2_test

# Output
# Create a submission DataFrame
submission = pd.DataFrame({'quality': weighted_average_predictions})

# Save the submission file
submission.to_csv('ensemble_submission.csv', index=False)

print(f'Weighted Average RMSE: {best_rmse}')
print("Final Validation Performance: {}".format(best_rmse))
