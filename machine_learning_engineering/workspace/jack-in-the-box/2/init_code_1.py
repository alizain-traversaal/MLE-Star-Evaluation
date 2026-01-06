The syntax error `SyntaxError: invalid syntax` in the original code was due to the presence of text before the actual Python code. I have removed that text. The corrected code is now a valid Python script that should execute without syntax errors. I am resubmitting the corrected code.

import pandas as pd
from prophet import Prophet
import numpy as np
import logging
logging.getLogger('prophet').setLevel(logging.WARNING)

try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError as e:
    print(f"Error: One or more data files not found. Please ensure 'train.csv' and 'test.csv' are in the ./input/ directory.")
    raise e

train_df['BUSINESS_DATE'] = pd.to_datetime(train_df['BUSINESS_DATE'])
test_df['BUSINESS_DATE'] = pd.to_datetime(test_df['BUSINESS_DATE'])

train_df = train_df.sort_values(by='BUSINESS_DATE')

train_df = train_df.rename(columns={'BUSINESS_DATE': 'ds', 'TOTAL_ADJUSTED_QUANTITY': 'y'})
test_df = test_df.rename(columns={'BUSINESS_DATE': 'ds', 'TOTAL_ADJUSTED_QUANTITY': 'y'})

# Validation Split
validation_size = 7
validation_df = train_df[-validation_size:].copy()
train_df = train_df[:-validation_size].copy()

model = Prophet()
model.fit(train_df)

future_validation = model.make_future_dataframe(periods=validation_size)
validation_forecast = model.predict(future_validation)

future_test = model.make_future_dataframe(periods=len(test_df))
test_forecast = model.predict(future_test)

validation_actual = validation_df['y'].values
validation_predicted = validation_forecast['yhat'][-validation_size:].values

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(validation_actual, validation_predicted)

print(f"Final Validation Performance: {mape}")

submission = pd.DataFrame()
submission['TOTAL_ADJUSTED_QUANTITY'] = test_forecast['yhat'][-len(test_df):].round(1)

print(submission)
