
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load data
train = pd.read_csv("./input/train.csv")

# Prepare data
X = train.drop("median_house_value", axis=1)
y = train["median_house_value"]

# --- Baseline Model ---
print("Baseline Model:")
X_baseline = X.copy()

# Feature Engineering (example - adding a combined feature)
X_baseline['rooms_per_household'] = X_baseline['total_rooms'] / X_baseline['households']

# Numerical features for scaling
numerical_features = X_baseline.select_dtypes(include=np.number).columns.tolist()

# Data scaling
scaler = StandardScaler()
X_baseline[numerical_features] = scaler.fit_transform(X_baseline[numerical_features])

# Model training
model = XGBRegressor(random_state=42, n_estimators=100, objective='reg:squarederror')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rmse = []

for fold, (train_index, val_index) in enumerate(kf.split(X_baseline, y)):
    X_train, X_val = X_baseline.iloc[train_index], X_baseline.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    oof_rmse.append(rmse)

final_validation_score = np.mean(oof_rmse)
print(f'Final Validation Performance (Baseline): {final_validation_score}')
print("-" * 40)

# --- Ablation 1: No Feature Engineering ---
print("Ablation 1: No Feature Engineering")
X_no_fe = X.copy()

# Numerical features for scaling
numerical_features = X_no_fe.select_dtypes(include=np.number).columns.tolist()

# Data scaling
scaler = StandardScaler()
X_no_fe[numerical_features] = scaler.fit_transform(X_no_fe[numerical_features])

# Model training
model = XGBRegressor(random_state=42, n_estimators=100, objective='reg:squarederror')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rmse = []

for fold, (train_index, val_index) in enumerate(kf.split(X_no_fe, y)):
    X_train, X_val = X_no_fe.iloc[train_index], X_no_fe.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    oof_rmse.append(rmse)

final_validation_score = np.mean(oof_rmse)
print(f'Final Validation Performance (No Feature Engineering): {final_validation_score}')
print("-" * 40)

# --- Ablation 2: No Scaling ---
print("Ablation 2: No Scaling")
X_no_scaling = X.copy()

# Feature Engineering (example - adding a combined feature)
X_no_scaling['rooms_per_household'] = X_no_scaling['total_rooms'] / X_no_scaling['households']


# Model training
model = XGBRegressor(random_state=42, n_estimators=100, objective='reg:squarederror')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rmse = []

for fold, (train_index, val_index) in enumerate(kf.split(X_no_scaling, y)):
    X_train, X_val = X_no_scaling.iloc[train_index], X_no_scaling.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    oof_rmse.append(rmse)

final_validation_score = np.mean(oof_rmse)
print(f'Final Validation Performance (No Scaling): {final_validation_score}')
print("-" * 40)

# Determine the most important part
if (final_validation_score > 0) :
  if (final_validation_score > 1):
    print("Both feature engineering and scaling are important.")
  else:
    print("It is difficult to find out which is more important with only two tests. Please conduct more tests.")

