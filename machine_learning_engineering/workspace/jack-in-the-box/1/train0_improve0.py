
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Load the training data
train_data = pd.read_csv("./input/train.csv")

# Convert 'BUSINESS_DATE' to datetime objects
train_data['BUSINESS_DATE'] = pd.to_datetime(train_data['BUSINESS_DATE'])

# Extract features: year, month, day of week, day of year
train_data['year'] = train_data['BUSINESS_DATE'].dt.year
train_data['month'] = train_data['BUSINESS_DATE'].dt.month
train_data['dayofweek'] = train_data['BUSINESS_DATE'].dt.dayofweek
train_data['dayofyear'] = train_data['BUSINESS_DATE'].dt.dayofyear

# Define features and target
features = ['year', 'month', 'dayofweek', 'dayofyear']
target = 'TOTAL_ADJUSTED_QUANTITY'

# Handle missing values with mean imputation
train_data = train_data.fillna(train_data.mean())

# Split the training data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(train_data[features], train_data[target], test_size=0.2, random_state=42)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_absolute_percentage_error', verbose=1, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Make predictions on the validation set using the best model
best_model = grid_search.best_estimator_
y_pred_val = best_model.predict(X_val)

# Evaluate the model on the validation set using MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_val, y_pred_val)
print(f'Validation MAPE: {mape}')


# Load the test data
test_data = pd.read_csv("./input/test.csv")

# Convert 'BUSINESS_DATE' to datetime objects
test_data['BUSINESS_DATE'] = pd.to_datetime(test_data['BUSINESS_DATE'])

# Extract features: year, month, day of week, day of year
test_data['year'] = test_data['BUSINESS_DATE'].dt.year
test_data['month'] = test_data['BUSINESS_DATE'].dt.month
test_data['dayofweek'] = test_data['BUSINESS_DATE'].dt.dayofweek
test_data['dayofyear'] = test_data['BUSINESS_DATE'].dt.dayofyear

# Handle missing values with mean imputation
test_data = test_data.fillna(train_data.mean())

# Make predictions on the test set
predictions = best_model.predict(test_data[features])

# Create a submission DataFrame
submission = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)

final_validation_score = mape
print(f'Final Validation Performance: {final_validation_score}')
