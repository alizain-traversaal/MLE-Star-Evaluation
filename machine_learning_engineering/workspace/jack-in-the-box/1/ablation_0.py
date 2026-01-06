
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

# --- Ablation 1: Removing 'dayofyear' feature ---
features_ablation_1 = ['year', 'month', 'dayofweek']
X_train_ablation_1, X_val_ablation_1 = X_train[features_ablation_1], X_val[features_ablation_1]

model_ablation_1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_ablation_1.fit(X_train_ablation_1, y_train)
y_pred_val_ablation_1 = model_ablation_1.predict(X_val_ablation_1)
mape_ablation_1 = mean_absolute_percentage_error(y_val, y_pred_val_ablation_1)
print(f'Ablation 1 (Removing dayofyear) Validation MAPE: {mape_ablation_1}')

# --- Ablation 2: Using only 'year' and 'month' features ---
features_ablation_2 = ['year', 'month']
X_train_ablation_2, X_val_ablation_2 = X_train[features_ablation_2], X_val[features_ablation_2]

model_ablation_2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_ablation_2.fit(X_train_ablation_2, y_train)
y_pred_val_ablation_2 = model_ablation_2.predict(X_val_ablation_2)
mape_ablation_2 = mean_absolute_percentage_error(y_val, y_pred_val_ablation_2)
print(f'Ablation 2 (Using only year and month) Validation MAPE: {mape_ablation_2}')

# --- Original Model ---
model_original = RandomForestRegressor(n_estimators=100, random_state=42)
model_original.fit(X_train, y_train)
y_pred_val_original = model_original.predict(X_val)
mape_original = mean_absolute_percentage_error(y_val, y_pred_val_original)
print(f'Original Model Validation MAPE: {mape_original}')

# --- Conclusion ---
print("\nConclusion:")
if mape_ablation_1 < mape_original and mape_ablation_1 < mape_ablation_2:
    print("Removing 'dayofyear' improved the performance.")
elif mape_ablation_2 < mape_original and mape_ablation_2 < mape_ablation_1:
    print("Using only 'year' and 'month' improved the performance.")
else:
    print("The original feature set provides the best performance.")
