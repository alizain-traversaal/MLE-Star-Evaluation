
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
try:
    from pmdarima import auto_arima
except ImportError:
    print("pmdarima not found. Install with: pip install pmdarima")
    auto_arima = None
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# Load the training data
train_data = pd.read_csv("./input/train.csv", parse_dates=["BUSINESS_DATE"], index_col="BUSINESS_DATE")
train_data = train_data.asfreq('D')
train_data = train_data.dropna()

# Split data into training and validation sets
train_size = int(len(train_data) * 0.8)
train, val = train_data[0:train_size], train_data[train_size:]

# Automatically find the best ARIMA parameters
order = (5,1,0)
if auto_arima:
    try:
        auto_model = auto_arima(train["TOTAL_ADJUSTED_QUANTITY"], seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
        order = auto_model.order
    except:
        order = (5,1,0)

# Fit the ARIMA model
model = ARIMA(train["TOTAL_ADJUSTED_QUANTITY"], order=order)
fit = model.fit()

# Make predictions on the validation set
predictions = fit.forecast(steps=len(val))

# Calculate MAPE
mape = mean_absolute_percentage_error(val["TOTAL_ADJUSTED_QUANTITY"], predictions)

# Print MAPE
print(f"Final Validation Performance: {mape}")

# Load the test data
test_data = pd.read_csv("./input/test.csv", parse_dates=["BUSINESS_DATE"], index_col="BUSINESS_DATE")

# Make predictions on the test set
test_predictions = fit.forecast(steps=len(test_data))

# Prepare the submission file
submission = pd.DataFrame({'TOTAL_ADJUSTED_QUANTITY': test_predictions})
submission.to_csv("submission_test.csv", index=False)
