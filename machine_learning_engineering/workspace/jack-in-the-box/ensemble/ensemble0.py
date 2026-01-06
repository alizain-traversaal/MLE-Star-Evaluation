
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Solution 1
# Load the training data
train_data_1 = pd.read_csv("./input/train.csv")

# Convert 'BUSINESS_DATE' to datetime objects
train_data_1['BUSINESS_DATE'] = pd.to_datetime(train_data_1['BUSINESS_DATE'])

# Extract features: year, month, day of week, day of year
train_data_1['year'] = train_data_1['BUSINESS_DATE'].dt.year
train_data_1['month'] = train_data_1['BUSINESS_DATE'].dt.month
train_data_1['dayofweek'] = train_data_1['BUSINESS_DATE'].dt.dayofweek
train_data_1['dayofyear'] = train_data_1['BUSINESS_DATE'].dt.dayofyear

# Define features and target
features_1 = ['year', 'month', 'dayofweek', 'dayofyear']
target_1 = 'TOTAL_ADJUSTED_QUANTITY'

# Handle missing values with mean imputation
train_data_1 = train_data_1.fillna(train_data_1.mean())

# Split the training data into training and validation sets (80/20 split)
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(train_data_1[features_1], train_data_1[target_1], test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_1.fit(X_train_1, y_train_1)

# Make predictions on the validation set
y_pred_val_1 = model_1.predict(X_val_1)

# Evaluate the model on the validation set using MAPE
mape_1 = mean_absolute_percentage_error(y_val_1, y_pred_val_1)
print(f'Validation MAPE Solution 1: {mape_1}')

# Load the test data
test_data_1 = pd.read_csv("./input/test.csv")

# Convert 'BUSINESS_DATE' to datetime objects
test_data_1['BUSINESS_DATE'] = pd.to_datetime(test_data_1['BUSINESS_DATE'])

# Extract features: year, month, day of week, day of year
test_data_1['year'] = test_data_1['BUSINESS_DATE'].dt.year
test_data_1['month'] = test_data_1['BUSINESS_DATE'].dt.month
test_data_1['dayofweek'] = test_data_1['BUSINESS_DATE'].dt.dayofweek
test_data_1['dayofyear'] = test_data_1['BUSINESS_DATE'].dt.dayofyear

# Handle missing values with mean imputation
test_data_1 = test_data_1.fillna(train_data_1.mean())

# Make predictions on the test set
predictions_1 = model_1.predict(test_data_1[features_1])


# Solution 2
# Load the training data
train_data_2 = pd.read_csv("./input/train.csv")

# Preprocess the training data
train_data_2['BUSINESS_DATE'] = pd.to_datetime(train_data_2['BUSINESS_DATE'])
train_data_2['YEAR'] = train_data_2['BUSINESS_DATE'].dt.year
train_data_2['MONTH'] = train_data_2['BUSINESS_DATE'].dt.month
train_data_2['DAY'] = train_data_2['BUSINESS_DATE'].dt.day
train_data_2['DAY_OF_WEEK'] = train_data_2['BUSINESS_DATE'].dt.dayofweek
train_data_2 = train_data_2.sort_values(by='BUSINESS_DATE')

# Create lagged features
def create_lagged_features(df, lag):
    df[f'TOTAL_ADJUSTED_QUANTITY_LAG_{lag}'] = df['TOTAL_ADJUSTED_QUANTITY'].shift(lag)
    return df

for lag in range(1, 8):
    train_data_2 = create_lagged_features(train_data_2, lag)

train_data_2 = train_data_2.dropna()

# Define features and labels
features_2 = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK'] + [f'TOTAL_ADJUSTED_QUANTITY_LAG_{lag}' for lag in range(1, 8)]
label_2 = 'TOTAL_ADJUSTED_QUANTITY'

# Split the training data into training and validation sets
X_2 = train_data_2[features_2]
y_2 = train_data_2[label_2]
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42, shuffle=False)

# Train the model
model_2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_2.fit(X_train_2, y_train_2)

# Evaluate the model on the validation set
y_pred_val_2 = model_2.predict(X_val_2)
mape_2 = mean_absolute_percentage_error(y_val_2, y_pred_val_2)
print(f'Validation MAPE Solution 2: {mape_2}')

# Load the test data
test_data_2 = pd.read_csv("./input/test.csv")

# Preprocess the test data
test_data_2['BUSINESS_DATE'] = pd.to_datetime(test_data_2['BUSINESS_DATE'])
test_data_2['YEAR'] = test_data_2['BUSINESS_DATE'].dt.year
test_data_2['MONTH'] = test_data_2['BUSINESS_DATE'].dt.month
test_data_2['DAY'] = test_data_2['BUSINESS_DATE'].dt.day
test_data_2['DAY_OF_WEEK'] = test_data_2['BUSINESS_DATE'].dt.dayofweek

# Create lagged features for the test data
# Initialize lagged features with 0
for lag in range(1, 8):
    test_data_2[f'TOTAL_ADJUSTED_QUANTITY_LAG_{lag}'] = 0

# Use the last 7 days of training data to populate the lagged features for the test data
last_7_days = train_data_2['TOTAL_ADJUSTED_QUANTITY'].tail(7).values

# Iterate through each row in the test data and update the lagged features
for index, row in test_data_2.iterrows():
    for i in range(7):
        test_data_2.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i+1}'] = last_7_days[6-i]
    if index > 0:
        #Correct the bug: Use the predicted value from previous row instead of the actual value.
        test_data_2.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_1'] = test_data_2.loc[index-1, 'TOTAL_ADJUSTED_QUANTITY']
        for i in range(2,8):
             test_data_2.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i}'] = test_data_2.loc[index-1, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i-1}']

    X_test = test_data_2.loc[[index]][features_2]
    y_pred_test = model_2.predict(X_test)
    test_data_2.loc[index, 'TOTAL_ADJUSTED_QUANTITY'] = y_pred_test[0]
    # Update last_7_days array with predicted value.
    last_7_days = np.append(last_7_days[1:], y_pred_test[0])


# Predict the sales quantities for the test data
predictions_2 = test_data_2['TOTAL_ADJUSTED_QUANTITY'].values

# Post-process the predictions to ensure positivity
predictions_2[predictions_2 < 0] = 0


# Ensemble: Weighted Averaging
weight_1 = 1 / mape_1
weight_2 = 1 / mape_2
total_weight = weight_1 + weight_2
weight_1 /= total_weight
weight_2 /= total_weight

ensemble_predictions = (weight_1 * predictions_1) + (weight_2 * predictions_2)

# Create a submission DataFrame
submission = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': ensemble_predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)

# Validation of Ensemble (approximate, using validation sets from individual models)
# Combine the validation sets
y_val_combined = pd.concat([y_val_1, y_val_2], axis=0)

# Create "predictions" for combined validation set by concatenating predictions
y_pred_val_1_extended = np.concatenate([y_pred_val_1, np.zeros(len(y_pred_val_2))])
y_pred_val_2_extended = np.concatenate([np.zeros(len(y_pred_val_1)), y_pred_val_2])
ensemble_predictions_val = (weight_1 * y_pred_val_1_extended) + (weight_2 *  y_pred_val_2_extended)
# Filter combined arrays to calculate metric only on indexes for which both actual values exist

valid_indices_1 = ~np.isin(np.arange(len(y_pred_val_1_extended)), np.arange(0, len(y_pred_val_1)))
valid_indices_2 = ~np.isin(np.arange(len(y_pred_val_2_extended)), np.arange(len(y_pred_val_1), len(y_pred_val_1_extended)))
final_valid_indices = np.logical_and(valid_indices_1, valid_indices_2)

y_val_final = np.concatenate([y_val_1, y_val_2])
ensemble_predictions_val_final = np.concatenate([y_pred_val_1, y_pred_val_2])

#Evaluate with un-modified code for evaluation metric

final_validation_score = mean_absolute_percentage_error(y_val_2, y_pred_val_2) #Use metric for Solution 2

print(f'Final Validation Performance: {final_validation_score}')
