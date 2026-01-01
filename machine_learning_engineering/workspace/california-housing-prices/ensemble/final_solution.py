
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the training and test data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")


# Preprocessing (Solution 1) - Modified to handle test data consistently
train_data = train_data.fillna(train_data.mean())
test_data = test_data.fillna(train_data.mean())

X = train_data.drop("median_house_value", axis=1)
y = train_data["median_house_value"]
X_test = test_data.copy()


# Solution 2 Preprocessing - Feature Engineering and Scaling - applied to both train and test
X2 = train_data.drop("median_house_value", axis=1)
X2_test = test_data.copy()
X2['rooms_per_household'] = X2['total_rooms'] / X2['households']
X2_test['rooms_per_household'] = X2_test['total_rooms'] / X2_test['households']
numerical_features = X2.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X2[numerical_features] = scaler.fit_transform(X2[numerical_features])
X2_test[numerical_features] = scaler.transform(X2_test[numerical_features])
X2 = X2.drop('rooms_per_household', axis=1)
X2_test = X2_test.drop('rooms_per_household', axis=1)


# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# OOF predictions placeholders
oof_lgbm = np.zeros(len(train_data))
oof_xgb1 = np.zeros(len(train_data))
oof_xgb2 = np.zeros(len(train_data))

# Test predictions placeholders
test_preds_lgbm = np.zeros(len(test_data))
test_preds_xgb1 = np.zeros(len(test_data))
test_preds_xgb2 = np.zeros(len(test_data))


# Solution 1 Models
lgbm = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42) #Best parameters found from Solution 1


# Solution 2 Model - Using best params from original solution 2 (after removing grid search)
best_params = {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.7}
xgb2 = xgb.XGBRegressor(**best_params, random_state=42)


# K-Fold Loop
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    X2_train, X2_val = X2.iloc[train_index], X2.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Solution 1 - LGBM
    lgbm.fit(X_train, y_train)
    oof_lgbm[val_index] = lgbm.predict(X_val)
    test_preds_lgbm += lgbm.predict(X_test) / kf.n_splits

    # Solution 1 - XGBoost
    xgb_model.fit(X_train, y_train)
    oof_xgb1[val_index] = xgb_model.predict(X_val)
    test_preds_xgb1 += xgb_model.predict(X_test) / kf.n_splits

    # Solution 2 - XGBoost
    xgb2.fit(X2_train, y_train)
    oof_xgb2[val_index] = xgb2.predict(X2_val)
    test_preds_xgb2 += xgb2.predict(X2_test) / kf.n_splits



# Meta-Learner Training Data
meta_X = pd.DataFrame({'lgbm': oof_lgbm, 'xgb1': oof_xgb1, 'xgb2': oof_xgb2})
meta_X_test = pd.DataFrame({'lgbm': test_preds_lgbm, 'xgb1': test_preds_xgb1, 'xgb2': test_preds_xgb2})

# Meta-Learner Model
meta_model = LinearRegression()
meta_model.fit(meta_X, y)

# Meta-Learner Predictions
final_predictions = meta_model.predict(meta_X_test)

# Clip predictions to the specified range
final_predictions = np.clip(final_predictions, 0, 1000000)


# Create Submission File
submission = pd.DataFrame({'median_house_value': final_predictions})
submission.to_csv('./final/submission.csv', index=False)

# Calculate OOF RMSE for validation
oof_predictions = meta_model.predict(meta_X)
oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))


print(f'Final Validation Performance: {oof_rmse}')
