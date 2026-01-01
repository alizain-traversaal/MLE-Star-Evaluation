
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the training and testing data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Handle missing values (impute with the median)
for col in train_df.columns:
    if train_df[col].isnull().any():
        train_df[col] = train_df[col].fillna(train_df[col].median())

for col in test_df.columns:
    if test_df[col].isnull().any():
        test_df[col] = test_df[col].fillna(test_df[col].median())

# Define features and target
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
target = 'median_house_value'

# Split the training data into training and validation sets
X = train_df[features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical features
numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

# Scale the numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])

# Train the model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {rmse_val}")

# Prepare the test data
X_test = test_df[features]

# Scale the numerical features of the test data
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Make predictions on the test data
y_pred_test = model.predict(X_test)

# Create a Pandas Series with the predictions
predictions = pd.Series(y_pred_test)

# Save the predictions to a CSV file
predictions.to_csv("median_house_value.csv", index=False, header=True)

final_validation_score = rmse_val
print(f"Final Validation Performance: {final_validation_score}")
