
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the training data from the provided CSV file
train_data = pd.read_csv("./input/train.csv")

# Handle missing values (if any) by filling them with the mean
train_data = train_data.fillna(train_data.mean())

# Extract features (X) and target variable (y)
X = train_data.drop("median_house_value", axis=1)
y = train_data["median_house_value"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: LightGBM and XGBoost Ensemble
lgbm = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
lgbm.fit(X_train, y_train)
y_pred_val_lgbm = lgbm.predict(X_val)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_val_xgb = xgb_model.predict(X_val)

y_pred_val_ensemble = (y_pred_val_lgbm + y_pred_val_xgb) / 2
rmse_baseline = np.sqrt(mean_squared_error(y_val, y_pred_val_ensemble))
print(f'Baseline Validation Performance: {rmse_baseline}')

# Ablation 1: Remove XGBoost
y_pred_val_ablation1 = y_pred_val_lgbm  # Use only LightGBM predictions
rmse_ablation1 = np.sqrt(mean_squared_error(y_val, y_pred_val_ablation1))
print(f'Ablation 1 (No XGBoost) Validation Performance: {rmse_ablation1}')

# Ablation 2: No imputation
train_data_no_fill = pd.read_csv("./input/train.csv")

# Extract features (X) and target variable (y)
X_no_fill = train_data_no_fill.drop("median_house_value", axis=1)
y_no_fill = train_data_no_fill["median_house_value"]

# Split the data into training and validation sets
X_train_no_fill, X_val_no_fill, y_train_no_fill, y_val_no_fill = train_test_split(X_no_fill, y_no_fill, test_size=0.2, random_state=42)

lgbm_no_fill = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
lgbm_no_fill.fit(X_train_no_fill, y_train_no_fill)
y_pred_val_lgbm_no_fill = lgbm_no_fill.predict(X_val_no_fill)

xgb_model_no_fill = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model_no_fill.fit(X_train_no_fill, y_train_no_fill)
y_pred_val_xgb_no_fill = xgb_model_no_fill.predict(X_val_no_fill)

y_pred_val_ensemble_no_fill = (y_pred_val_lgbm_no_fill + y_pred_val_xgb_no_fill) / 2
rmse_ablation2 = np.sqrt(mean_squared_error(y_val_no_fill, y_pred_val_ensemble_no_fill))

print(f'Ablation 2 (No Imputation) Validation Performance: {rmse_ablation2}')

if rmse_ablation1 > rmse_baseline and rmse_ablation2 > rmse_baseline:
    print("Both XGBoost and Imputation contribute positively to the model performance.")
elif rmse_ablation1 > rmse_baseline:
    print("XGBoost contributes significantly to the model performance.")
elif rmse_ablation2 > rmse_baseline:
    print("Imputation contributes significantly to the model performance.")
else:
    print("Both XGBoost and Imputation may not significantly contribute to the model performance.")
