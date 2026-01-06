
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
try:
    from pmdarima import auto_arima
except ImportError:
    print("pmdarima not found. Install with: pip install pmdarima")
    auto_arima = None

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

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val_rf = model.predict(X_val)

# Evaluate the model on the validation set using MAPE
mape_rf = mean_absolute_percentage_error(y_val, y_pred_val_rf)
print(f'Validation MAPE (Random Forest): {mape_rf}')

# ARIMA Model
arima_train_data = train_data[['BUSINESS_DATE', 'TOTAL_ADJUSTED_QUANTITY']].set_index('BUSINESS_DATE').asfreq('D').dropna()
arima_train_size = int(len(arima_train_data) * 0.8)
arima_train, arima_val = arima_train_data[0:arima_train_size], arima_train_data[arima_train_size:]

order = (5,1,0)
if auto_arima:
    try:
        auto_model = auto_arima(arima_train["TOTAL_ADJUSTED_QUANTITY"], seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
        order = auto_model.order
    except:
        order = (5,1,0)

arima_model = ARIMA(arima_train["TOTAL_ADJUSTED_QUANTITY"], order=order)
arima_fit = arima_model.fit()
arima_predictions = arima_fit.forecast(steps=len(arima_val))

mape_arima = mean_absolute_percentage_error(arima_val["TOTAL_ADJUSTED_QUANTITY"], arima_predictions)
print(f"Validation MAPE (ARIMA): {mape_arima}")

# Weighted Averaging Ensemble
ensemble_weights = [0.5, 0.5]  # Adjust weights as needed
y_pred_val_ensemble = (ensemble_weights[0] * y_pred_val_rf) + (ensemble_weights[1] * np.interp(np.arange(len(y_pred_val_rf)), np.linspace(0, len(y_pred_val_rf)-1, len(arima_predictions)), arima_predictions))
mape_ensemble = mean_absolute_percentage_error(y_val, y_pred_val_ensemble)
print(f"Validation MAPE (Ensemble): {mape_ensemble}")

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

# Make predictions on the test set (Random Forest)
predictions_rf = model.predict(test_data[features])

# Make predictions on the test set (ARIMA)
test_data_arima = pd.read_csv("./input/test.csv", parse_dates=["BUSINESS_DATE"], index_col="BUSINESS_DATE")
test_predictions_arima = arima_fit.forecast(steps=len(test_data_arima))

# Ensemble predictions on the test set
predictions_ensemble = (ensemble_weights[0] * predictions_rf) + (ensemble_weights[1] * np.interp(np.arange(len(predictions_rf)), np.linspace(0, len(predictions_rf)-1, len(test_predictions_arima)), test_predictions_arima))

# Create a submission DataFrame
submission = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': predictions_ensemble})

# Save the submission file
submission.to_csv('submission.csv', index=False)

final_validation_score = mape_ensemble
print(f'Final Validation Performance: {final_validation_score}')
