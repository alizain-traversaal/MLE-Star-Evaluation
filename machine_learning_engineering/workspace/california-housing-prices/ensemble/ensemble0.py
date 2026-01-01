
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler

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


# Load data for Solution 2
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Prepare data
X_sol2 = train.drop("median_house_value", axis=1)
y_sol2 = train["median_house_value"]
X_test = test.copy()

# Feature Engineering (example - adding a combined feature)
X_sol2['rooms_per_household'] = X_sol2['total_rooms'] / X_sol2['households']
X_test['rooms_per_household'] = X_test['total_rooms'] / X_test['households']

# Numerical features for scaling
numerical_features = X_sol2.select_dtypes(include=np.number).columns.tolist()

# Data scaling
scaler = StandardScaler()
X_sol2[numerical_features] = scaler.fit_transform(X_sol2[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Drop 'rooms_per_household'
X_sol2 = X_sol2.drop('rooms_per_household', axis=1)
X_test = X_test.drop('rooms_per_household', axis=1)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9]
}

# Initialize XGBoost regressor
xgb_sol2 = XGBRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_sol2, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=0)

# Perform grid search
grid_search.fit(X_sol2, y_sol2)

# Get best parameters
best_params = grid_search.best_params_

# Train model with best parameters on the whole training dataset
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(X_sol2, y_sol2)

# Make predictions on the test set
test_predictions = final_model.predict(X_test)

# Data scaling again for solution 2 validation data
X_val_sol2 = X_val.copy()
X_val_sol2['rooms_per_household'] = X_val_sol2['total_rooms'] / X_val_sol2['households']
X_val_sol2[numerical_features] = scaler.transform(X_val_sol2[numerical_features])
X_val_sol2 = X_val_sol2.drop('rooms_per_household', axis=1)

# Predict on the validation set using Solution 2
oof_predictions_from_solution_2 = final_model.predict(X_val_sol2)

# Optimize weights
weights = np.arange(0, 1.01, 0.01)
best_weight = None
best_rmse = float('inf')

for w in weights:
    y_pred_val_ensemble = w * ((y_pred_val_lgbm + y_pred_val_xgb) / 2) + (1 - w) * oof_predictions_from_solution_2
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w

print(f'Best Weight: {best_weight}')

# Final Prediction
final_predictions = best_weight * ((lgbm.predict(X_test) + xgb_model.predict(X_test)) / 2) + (1 - best_weight) * test_predictions

# Create submission file
submission = pd.DataFrame({'median_house_value': final_predictions})
submission.to_csv('submission.csv', index=False)

final_validation_score = best_rmse

print(f'Final Validation Performance: {final_validation_score}')
