
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('./input/train.csv')
X = data.drop(['quality'], axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost regressor
xgbr = xgb.XGBRegressor(objective='reg:squarederror', # Specify squared error for regression
                            n_estimators=1000,  # Number of boosting rounds
                            learning_rate=0.05, # Step size shrinkage 
                            max_depth=5,        # Maximum depth of a tree
                            min_child_weight=1, # Minimum sum of instance weight needed in a child
                            gamma=0,            # Minimum loss reduction required to make a further partition
                            subsample=0.8,      # Subsample ratio of the training instance
                            colsample_bytree=0.8, # Subsample ratio of columns when constructing each tree
                            reg_alpha=0.005,      # L1 regularization term on weights
                            random_state=42,    # Random seed
                            n_jobs=-1)          # Use all available cores

# Initialize LightGBM regressor
lgbm = lgb.LGBMRegressor(objective='regression', # Specify regression task
                            n_estimators=1000,  # Number of boosting rounds
                            learning_rate=0.05, # Step size shrinkage
                            max_depth=5,        # Maximum depth of a tree
                            num_leaves=31,      # Maximum number of leaves in one tree
                            min_child_samples=20, # Minimum number of data needed in a child
                            subsample=0.8,      # Subsample ratio of the training instance
                            colsample_bytree=0.8, # Subsample ratio of columns when constructing each tree
                            reg_alpha=0.001,      # L1 regularization term on weights
                            reg_lambda=0.001,     # L2 regularization term on weights
                            random_state=42,    # Random seed
                            n_jobs=-1)          # Use all available cores

# Train the models
xgbr.fit(X_train, y_train)
lgbm.fit(X_train, y_train)

# Make predictions
xgbr_predictions = xgbr.predict(X_test)
lgbm_predictions = lgbm.predict(X_test)

# Ensemble the predictions (averaging)
predictions = (xgbr_predictions + lgbm_predictions) / 2

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

print("Final Validation Performance: {}".format(rmse))
