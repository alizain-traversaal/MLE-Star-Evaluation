
import pandas as pd
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

# Create XGBoost regressor with specified hyperparameters
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=100,  # Number of boosting rounds
                             learning_rate=0.1, # Step size shrinkage
                             max_depth=5,       # Maximum depth of a tree
                             random_state=42)    # Random seed for reproducibility

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = xgb_model.predict(X_val)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f'Final Validation Performance: {rmse}')
