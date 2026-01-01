
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('./input/train.csv')

# Split into features (X) and target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the XGBoost Regressor
xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)


# Make predictions
y_pred = xgb.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred)
rmse = rmse**0.5
print(f'RMSE: {rmse}')

# Make predictions on the test set
test_data = pd.read_csv('./input/test.csv')
test_predictions = xgb.predict(test_data)

# Create a submission DataFrame
submission = pd.DataFrame({'quality': test_predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print(f'RMSE: {rmse}')
print(f'Final Validation Performance: {rmse}')
