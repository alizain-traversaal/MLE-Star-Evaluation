
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


import numpy as np

# Initialize a matrix to hold the lagged features for the test data
num_test_rows = len(test_data)
lagged_features_matrix = np.zeros((num_test_rows, 7))

# Use the last 7 days of training data to initialize the first row of the lagged features matrix
last_7_days = train_data['TOTAL_ADJUSTED_QUANTITY'].tail(7).values
lagged_features_matrix[0, :] = last_7_days[::-1]  # Reverse to match lag order

# Iterate through the test data to populate the lagged features matrix
predicted_values = []
for i in range(num_test_rows):
    if i > 0:
        # Use the predicted value from the previous row to update the lagged features
        lagged_features_matrix[i, 0] = predicted_values[-1]
        lagged_features_matrix[i, 1:] = lagged_features_matrix[i-1, :-1]

    X_test = lagged_features_matrix[[i], :]
    lag_features = [f'TOTAL_ADJUSTED_QUANTITY_LAG_{j}' for j in range(1,8)]
    other_features = [f for f in features if f not in lag_features]
    X_test = np.concatenate([X_test, test_data.iloc[[i]][other_features].values], axis=1)
    y_pred_test = model.predict(X_test)
    predicted_values.append(y_pred_test[0])
    test_data.loc[test_data.index[i], 'TOTAL_ADJUSTED_QUANTITY'] = y_pred_test[0]

# Assign the predicted values to the test data
# test_data['TOTAL_ADJUSTED_QUANTITY'] = predicted_values #This line is redundant, as we already updated the test_data in the loop



# Predict the sales quantities for the test data
y_pred_test = test_data['TOTAL_ADJUSTED_QUANTITY'].values

# Post-process the predictions to ensure positivity
y_pred_test[y_pred_test < 0] = 0

# Create a submission DataFrame
submission = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': y_pred_test})

# Print the submission DataFrame
print(submission)
