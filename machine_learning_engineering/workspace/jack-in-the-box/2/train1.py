
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# Load the training data
train_data = pd.read_csv("./input/train.csv")

# Preprocess the training data
train_data['BUSINESS_DATE'] = pd.to_datetime(train_data['BUSINESS_DATE'])
train_data['YEAR'] = train_data['BUSINESS_DATE'].dt.year
train_data['MONTH'] = train_data['BUSINESS_DATE'].dt.month
train_data['DAY'] = train_data['BUSINESS_DATE'].dt.day
train_data['DAY_OF_WEEK'] = train_data['BUSINESS_DATE'].dt.dayofweek
train_data = train_data.sort_values(by='BUSINESS_DATE')

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
print(f'Final Validation Performance: {mape}')

# Load the test data
test_data = pd.read_csv("./input/test.csv")

# Preprocess the test data
test_data['BUSINESS_DATE'] = pd.to_datetime(test_data['BUSINESS_DATE'])
test_data['YEAR'] = test_data['BUSINESS_DATE'].dt.year
test_data['MONTH'] = test_data['BUSINESS_DATE'].dt.month
test_data['DAY'] = test_data['BUSINESS_DATE'].dt.day
test_data['DAY_OF_WEEK'] = test_data['BUSINESS_DATE'].dt.dayofweek

# Create lagged features for the test data
# Initialize lagged features with 0
for lag in range(1, 8):
    test_data[f'TOTAL_ADJUSTED_QUANTITY_LAG_{lag}'] = 0

# Use the last 7 days of training data to populate the lagged features for the test data
last_7_days = train_data['TOTAL_ADJUSTED_QUANTITY'].tail(7).values

# Iterate through each row in the test data and update the lagged features
for index, row in test_data.iterrows():
    for i in range(7):
        test_data.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i+1}'] = last_7_days[6-i]
    if index > 0:
        #Correct the bug: Use the predicted value from previous row instead of the actual value.
        test_data.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_1'] = test_data.loc[index-1, 'TOTAL_ADJUSTED_QUANTITY']
        for i in range(2,8):
             test_data.loc[index, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i}'] = test_data.loc[index-1, f'TOTAL_ADJUSTED_QUANTITY_LAG_{i-1}']

    X_test = test_data.loc[[index]][features]
    y_pred_test = model.predict(X_test)
    test_data.loc[index, 'TOTAL_ADJUSTED_QUANTITY'] = y_pred_test[0]
    # Update last_7_days array with predicted value.
    last_7_days = np.append(last_7_days[1:], y_pred_test[0])


# Predict the sales quantities for the test data
y_pred_test = test_data['TOTAL_ADJUSTED_QUANTITY'].values

# Post-process the predictions to ensure positivity
y_pred_test[y_pred_test < 0] = 0

# Create a submission DataFrame
submission = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': y_pred_test})

# Print the submission DataFrame
print(submission)
