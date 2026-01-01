
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('./input/train.csv')

# Split into features (X) and target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost Regressor
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

# Initialize and train the LightGBM Regressor
lgbm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
lgbm.fit(X_train, y_train)

# Make predictions using both models
xgb_pred = xgb.predict(X_test)
lgbm_pred = lgbm.predict(X_test)

# Ensemble the predictions (simple average)
y_pred = (xgb_pred + lgbm_pred) / 2

# Evaluate the ensemble model
rmse = mean_squared_error(y_test, y_pred)**0.5
print(f'RMSE: {rmse}')

# Make predictions on the test set using both models
test_data = pd.read_csv('./input/test.csv')
xgb_test_predictions = xgb.predict(test_data)
lgbm_test_predictions = lgbm.predict(test_data)

# Ensemble the test set predictions
test_predictions = (xgb_test_predictions + lgbm_test_predictions) / 2

# Create a submission DataFrame
submission = pd.DataFrame({'quality': test_predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {rmse}')
