
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import os

# Load the data
data = pd.read_csv('./input/train.csv')
X = data.drop(['quality'], axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize XGBoost regressor 1
xgbr1 = xgb.XGBRegressor(objective='reg:squarederror', # Specify squared error for regression
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


# Train the model 1
xgbr1.fit(X_train, y_train)

# Make predictions 1
predictions1 = xgbr1.predict(X_test)


# Initialize XGBoost regressor 2
xgb2 = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
xgb2.fit(X_train, y_train)

# Make predictions 2
predictions2 = xgb2.predict(X_test)


# Convert predictions to ranks
ranks1 = rankdata(predictions1)
ranks2 = rankdata(predictions2)

# Average the ranks
average_ranks = (ranks1 + ranks2) / 2

# Convert back to predictions - validation set
# Sort the target variable
sorted_y_test = sorted(y_test)

# Assign the sorted target values based on the averaged ranks
rank_based_predictions_val = [sorted_y_test[int(rank) - 1] for rank in average_ranks]

# Evaluate the model
mse = mean_squared_error(y_test, rank_based_predictions_val)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')


# Load test data
test_data = pd.read_csv('./input/test.csv')

# Make predictions on the test set using model 1
test_predictions1 = xgbr1.predict(test_data)

# Make predictions on the test set using model 2
test_predictions2 = xgb2.predict(test_data)

# Convert test predictions to ranks
test_ranks1 = rankdata(test_predictions1)
test_ranks2 = rankdata(test_predictions2)

# Average the ranks for test predictions
average_test_ranks = (test_ranks1 + test_ranks2) / 2

# Convert back to predictions - test set
# Sort the training target variable
sorted_y_train = sorted(y_train)

# Since we don't have the true target for test, use training data to map back
rank_based_predictions_test = [sorted_y_train[int(rank * (len(y_train) - 1) / len(test_data))] if rank <= len(test_data) else sorted_y_train[-1] for rank in average_test_ranks]

# Create a submission DataFrame
submission = pd.DataFrame({'quality': rank_based_predictions_test})

# Save the submission file
os.makedirs('./final', exist_ok=True)
submission.to_csv('./final/submission.csv', index=False)

print("Final Validation Performance: {}".format(rmse))
