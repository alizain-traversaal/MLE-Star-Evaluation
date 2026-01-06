
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


# Feature Engineering: Create lagged features
window1 = 7
window2 = 30

# Calculate rolling mean and standard deviation for the target variable
train_data['rolling_mean_7'] = train_data['TOTAL_ADJUSTED_QUANTITY'].rolling(window=window1).mean()
train_data['rolling_std_7'] = train_data['TOTAL_ADJUSTED_QUANTITY'].rolling(window=window1).std()
train_data['rolling_mean_30'] = train_data['TOTAL_ADJUSTED_QUANTITY'].rolling(window=window2).mean()
train_data['rolling_std_30'] = train_data['TOTAL_ADJUSTED_QUANTITY'].rolling(window=window2).std()

# Fill NaN values resulting from rolling calculations (e.g., with 0 or the mean)
train_data.fillna(0, inplace=True)

# Prepare data for the model, including the new features
X = train_data.drop('BUSINESS_DATE', axis=1).drop('TOTAL_ADJUSTED_QUANTITY', axis=1)
y = train_data['TOTAL_ADJUSTED_QUANTITY']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model on the validation set using MAPE
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

# Create a temporary DataFrame to store predictions
temp_data = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': np.zeros(len(test_data))})

# Feature engineering for test data - iterative prediction
predictions = []
for i in range(len(test_data)):
    # Update temp_data with the current prediction
    test_data.loc[i, 'TOTAL_ADJUSTED_QUANTITY'] = temp_data['TOTAL_ADJUSTED_QUANTITY'][i]

    # Calculate rolling features
    test_data['rolling_mean_7'] = test_data['TOTAL_ADJUSTED_QUANTITY'].rolling(window=window1).mean()
    test_data['rolling_std_7'] = test_data['TOTAL_ADJUSTED_QUANTITY'].rolling(window=window1).std()
    test_data['rolling_mean_30'] = test_data['TOTAL_ADJUSTED_QUANTITY'].rolling(window=window2).mean()
    test_data['rolling_std_30'] = test_data['TOTAL_ADJUSTED_QUANTITY'].rolling(window=window2).std()

    test_data.fillna(0, inplace=True)

    # Handle missing values with mean imputation
    test_data = test_data.fillna(train_data.mean())


    # Prepare features for prediction
    features = ['year', 'month', 'dayofweek', 'dayofyear', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30']
    X_test = test_data.drop('BUSINESS_DATE', axis=1).drop('TOTAL_ADJUSTED_QUANTITY', axis=1)

    # Predict for the current day
    prediction = model.predict(X_test.iloc[[i]])[0]
    predictions.append(prediction)

    # Store the prediction in temp_data for use in subsequent rolling calculations
    temp_data['TOTAL_ADJUSTED_QUANTITY'][i] = prediction


# Create a submission DataFrame
submission = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)

final_validation_score = mape
print(f'Final Validation Performance: {final_validation_score}')
