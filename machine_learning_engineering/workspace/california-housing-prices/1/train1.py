
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the training data from the provided CSV file
train_data = pd.read_csv("./input/train.csv")

# Handle missing values (if any) by filling them with the mean
train_data = train_data.fillna(train_data.mean())

# Extract features (X) and target variable (y)
X = train_data.drop("median_house_value", axis=1)
y = train_data["median_house_value"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM regressor model
lgbm = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the LightGBM model on the training data
lgbm.fit(X_train, y_train)

# Predict on the validation set using LightGBM
y_pred_val_lgbm = lgbm.predict(X_val)


import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Create XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=3, verbose=0)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters:", best_params)

# Create XGBoost regressor with the best hyperparameters
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=best_params['n_estimators'],
                             learning_rate=best_params['learning_rate'],
                             max_depth=best_params['max_depth'],
                             random_state=42)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set using XGBoost
y_pred_val_xgb = xgb_model.predict(X_val)


# Ensemble the predictions (simple average)
y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2

# Evaluate the ensemble model using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))

print(f'Final Validation Performance: {rmse}')
