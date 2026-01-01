
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('./input/train.csv')
X = data.drop(['quality'], axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline XGBoost regressor
xgbr_baseline = xgb.XGBRegressor(objective='reg:squarederror',
                                n_estimators=1000,
                                learning_rate=0.05,
                                max_depth=5,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_alpha=0.005,
                                random_state=42,
                                n_jobs=-1)

xgbr_baseline.fit(X_train, y_train)
predictions_baseline = xgbr_baseline.predict(X_test)
mse_baseline = mean_squared_error(y_test, predictions_baseline)
rmse_baseline = np.sqrt(mse_baseline)
print(f'Baseline Root Mean Squared Error: {rmse_baseline}')


# Ablation 1: Remove L1 regularization (reg_alpha)
xgbr_no_reg = xgb.XGBRegressor(objective='reg:squarederror',
                                n_estimators=1000,
                                learning_rate=0.05,
                                max_depth=5,
                                min_child_weight=1,
                                gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_alpha=0,  # Removed L1 regularization
                                random_state=42,
                                n_jobs=-1)

xgbr_no_reg.fit(X_train, y_train)
predictions_no_reg = xgbr_no_reg.predict(X_test)
mse_no_reg = mean_squared_error(y_test, predictions_no_reg)
rmse_no_reg = np.sqrt(mse_no_reg)
print(f'Root Mean Squared Error without L1 regularization: {rmse_no_reg}')


# Ablation 2: Reduce the number of estimators
xgbr_less_estimators = xgb.XGBRegressor(objective='reg:squarederror',
                                        n_estimators=500,  # Reduced estimators
                                        learning_rate=0.05,
                                        max_depth=5,
                                        min_child_weight=1,
                                        gamma=0,
                                        subsample=0.8,
                                        colsample_bytree=0.8,
                                        reg_alpha=0.005,
                                        random_state=42,
                                        n_jobs=-1)

xgbr_less_estimators.fit(X_train, y_train)
predictions_less_estimators = xgbr_less_estimators.predict(X_test)
mse_less_estimators = mean_squared_error(y_test, predictions_less_estimators)
rmse_less_estimators = np.sqrt(mse_less_estimators)
print(f'Root Mean Squared Error with 500 estimators: {rmse_less_estimators}')

if rmse_no_reg < rmse_baseline and rmse_no_reg < rmse_less_estimators:
    print("Removing L1 regularization contributed the most to performance (lowest RMSE).")
elif rmse_less_estimators < rmse_baseline and rmse_less_estimators < rmse_no_reg:
    print("Reducing the number of estimators contributed the most to performance (lowest RMSE).")
else:
    print("The baseline configuration performed the best.")
