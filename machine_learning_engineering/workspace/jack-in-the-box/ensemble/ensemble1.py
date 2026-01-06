
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Solution 1
def solution_1(train_data, test_data):
    # Convert 'BUSINESS_DATE' to datetime objects
    train_data['BUSINESS_DATE'] = pd.to_datetime(train_data['BUSINESS_DATE'])
    test_data['BUSINESS_DATE'] = pd.to_datetime(test_data['BUSINESS_DATE'])

    # Extract features: year, month, day of week, day of year
    train_data['year'] = train_data['BUSINESS_DATE'].dt.year
    train_data['month'] = train_data['BUSINESS_DATE'].dt.month
    train_data['dayofweek'] = train_data['BUSINESS_DATE'].dt.dayofweek
    train_data['dayofyear'] = train_data['BUSINESS_DATE'].dt.dayofyear

    test_data['year'] = test_data['BUSINESS_DATE'].dt.year
    test_data['month'] = test_data['BUSINESS_DATE'].dt.month
    test_data['dayofweek'] = test_data['BUSINESS_DATE'].dt.dayofweek
    test_data['dayofyear'] = test_data['BUSINESS_DATE'].dt.dayofyear

    # Define features and target
    features = ['year', 'month', 'dayofweek', 'dayofyear']
    target = 'TOTAL_ADJUSTED_QUANTITY'

    # Handle missing values with mean imputation
    train_data = train_data.fillna(train_data.mean())
    test_data = test_data.fillna(train_data.mean())  # Use train_data mean for test_data

    # Split the training data into training and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(train_data[features], train_data[target], test_size=0.2, random_state=42)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred_val = model.predict(X_val)

    # Evaluate the model on the validation set using MAPE
    mape = mean_absolute_percentage_error(y_val, y_pred_val)
    print(f'Solution 1 Validation MAPE: {mape}')

    # Make predictions on the test set
    predictions = model.predict(test_data[features])

    return predictions, mape, y_val, y_pred_val

# Solution 2
def solution_2(train_data, test_data):
    # Preprocess the training data
    train_data['BUSINESS_DATE'] = pd.to_datetime(train_data['BUSINESS_DATE'])
    train_data['YEAR'] = train_data['BUSINESS_DATE'].dt.year
    train_data['MONTH'] = train_data['BUSINESS_DATE'].dt.month
    train_data['DAY'] = train_data['BUSINESS_DATE'].dt.day
    train_data['DAY_OF_WEEK'] = train_data['BUSINESS_DATE'].dt.dayofweek
    train_data = train_data.sort_values(by='BUSINESS_DATE')

    test_data['BUSINESS_DATE'] = pd.to_datetime(test_data['BUSINESS_DATE'])
    test_data['YEAR'] = test_data['BUSINESS_DATE'].dt.year
    test_data['MONTH'] = test_data['BUSINESS_DATE'].dt.month
    test_data['DAY'] = test_data['BUSINESS_DATE'].dt.day
    test_data['DAY_OF_WEEK'] = test_data['BUSINESS_DATE'].dt.dayofweek


    # Create lagged features
    def create_lagged_features(df, lag):
        df[f'TOTAL_ADJUSTED_QUANTITY_LAG_{lag}'] = df['TOTAL_ADJUSTED_QUANTITY'].shift(lag)
        return df

    for lag in range(1, 8):
        train_data = create_lagged_features(train_data, lag)

    train_data = train_data.dropna()

    # Define features and labels
    features = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK'] + [f'TOTAL_ADJUSTED_QUANTITY_LAG_{lag}' for lag in range(1, 8)]
    label = 'TOTAL_ADJUSTED_QUANTITY'

    # Split the training data into training and validation sets
    X = train_data[features]
    y = train_data[label]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred_val = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred_val)
    print(f'Solution 2 Validation MAPE: {mape}')


    # Create lagged features for the test data
    # Initialize lagged features with 0
    for lag in range(1, 8):
        test_data[f'TOTAL_ADJUSTED_QUANTITY_LAG_{lag}'] = 0

    # Use the last 7 days of training data to populate the lagged features for the test data
    last_7_days = train_data['TOTAL_ADJUSTED_QUANTITY'].tail(7).values

    # Iterate through each row in the test data and update the lagged features
    predictions = []
    for index, row in test_data.iterrows():
        for i in range(7):
            test_data.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i+1}'] = last_7_days[6-i]
        if index > 0:
            #Correct the bug: Use the predicted value from previous row instead of the actual value.
            test_data.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_1'] = predictions[-1]
            for i in range(2,8):
                 test_data.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i}'] = test_data.loc[index-1, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i-1}']

        X_test = test_data.loc[[index]][features]
        y_pred_test = model.predict(X_test)
        test_data.loc[index, 'TOTAL_ADJUSTED_QUANTITY'] = y_pred_test[0]
        predictions.append(y_pred_test[0])
        # Update last_7_days array with predicted value.
        last_7_days = np.append(last_7_days[1:], y_pred_test[0])


    # Predict the sales quantities for the test data
    y_pred_test = test_data['TOTAL_ADJUSTED_QUANTITY'].values

    # Post-process the predictions to ensure positivity
    y_pred_test[y_pred_test < 0] = 0

    return y_pred_test, mape, y_val, y_pred_val

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Run the solutions
predictions_1, mape_1, y_val_1, y_pred_val_1 = solution_1(train_data.copy(), test_data.copy())
predictions_2, mape_2, y_val_2, y_pred_val_2 = solution_2(train_data.copy(), test_data.copy())

# Identify Date Ranges and Define Switching Mechanism
# For simplicity, let's assume solution 2 is better before 2018 and solution 1 is better after 2018 based on the validation data
# In reality, a more sophisticated approach like a decision tree trained on the validation set would be used.
# dates = pd.to_datetime(train_data['BUSINESS_DATE']).dt.year.unique()
# print(dates)

#Apply switching mechanism
final_predictions = []
test_data['BUSINESS_DATE'] = pd.to_datetime(test_data['BUSINESS_DATE'])
for index, row in test_data.iterrows():
    if row['BUSINESS_DATE'].year < 2018:
        final_predictions.append(predictions_2[index])
    else:
        final_predictions.append(predictions_1[index])

# Create a submission DataFrame
submission = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': final_predictions})

# Post-process the predictions to ensure positivity
submission[submission['TOTAL_ADJUSTED_QUANTITY'] < 0] = 0

# Save the submission file
submission.to_csv('submission.csv', index=False)

#Evaluate ensembled result on validation data

train_val_data = pd.read_csv("./input/train.csv")
train_val_data['BUSINESS_DATE'] = pd.to_datetime(train_val_data['BUSINESS_DATE'])

train_1, val_1, y_train_1, y_val_1 = train_test_split(train_val_data.copy(), train_val_data['TOTAL_ADJUSTED_QUANTITY'], test_size=0.2, random_state=42)
train_2, val_2, y_train_2, y_val_2 = train_test_split(train_val_data.copy(), train_val_data['TOTAL_ADJUSTED_QUANTITY'], test_size=0.2, random_state=42, shuffle=False)

y_pred_ensemble = []

val_1['BUSINESS_DATE'] = pd.to_datetime(val_1['BUSINESS_DATE'])
val_2['BUSINESS_DATE'] = pd.to_datetime(val_2['BUSINESS_DATE'])

for index, row in val_1.iterrows():

    year = row['BUSINESS_DATE'].year
    if year < 2018:
        y_pred_ensemble.append(y_pred_val_2[val_2.index.get_loc(index)])
    else:
        y_pred_ensemble.append(y_pred_val_1[val_1.index.get_loc(index)])


final_validation_score = mean_absolute_percentage_error(y_val_1, y_pred_ensemble)
print(f'Final Validation Performance: {final_validation_score}')
