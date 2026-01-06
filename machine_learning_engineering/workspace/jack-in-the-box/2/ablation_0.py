
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

# Baseline Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_pred_val)
print(f'Baseline Validation Performance: {mape}')

# Ablation 1: Removing Day of Week Feature
features_ablation_1 = ['YEAR', 'MONTH', 'DAY'] + [f'TOTAL_ADJUSTED_QUANTITY_LAG_{lag}' for lag in range(1, 8)]
X_train_ablation_1 = X_train[features_ablation_1]
X_val_ablation_1 = X_val[features_ablation_1]

model_ablation_1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_ablation_1.fit(X_train_ablation_1, y_train)
y_pred_val_ablation_1 = model_ablation_1.predict(X_val_ablation_1)
mape_ablation_1 = mean_absolute_percentage_error(y_val, y_pred_val_ablation_1)
print(f'Ablation 1 (Removing Day of Week) Validation Performance: {mape_ablation_1}')

# Ablation 2: Removing Lagged Features
features_ablation_2 = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK']
X_train_ablation_2 = X_train[features_ablation_2]
X_val_ablation_2 = X_val[features_ablation_2]

model_ablation_2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_ablation_2.fit(X_train_ablation_2, y_train)
y_pred_val_ablation_2 = model_ablation_2.predict(X_val_ablation_2)
mape_ablation_2 = mean_absolute_percentage_error(y_val, y_pred_val_ablation_2)
print(f'Ablation 2 (Removing Lagged Features) Validation Performance: {mape_ablation_2}')

if mape_ablation_1 > mape and mape_ablation_2 > mape:
    print("Lagged features and Day of Week contribute the most to the overall performance.")
elif mape_ablation_1 > mape:
    print("Day of Week contributes the most to the overall performance.")
elif mape_ablation_2 > mape:
    print("Lagged features contribute the most to the overall performance.")
else:
    print("Neither Day of Week nor Lagged Features significantly degrade performance.")
