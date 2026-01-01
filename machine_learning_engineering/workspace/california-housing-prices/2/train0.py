
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Prepare data
X = train.drop("median_house_value", axis=1)
y = train["median_house_value"]
X_test = test.copy()

# Feature Engineering (example - adding a combined feature)
X['rooms_per_household'] = X['total_rooms'] / X['households']
X_test['rooms_per_household'] = X_test['total_rooms'] / X_test['households']

# Numerical features for scaling
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Data scaling
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Model training
model = XGBRegressor(random_state=42, n_estimators=100)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rmse = []
test_predictions = []

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    oof_rmse.append(rmse)
    
    test_pred = model.predict(X_test)
    test_predictions.append(test_pred)

# Calculate average OOF RMSE
final_validation_score = np.mean(oof_rmse)

# Average test predictions
final_predictions = np.mean(test_predictions, axis=0)

# Create submission file
submission = pd.DataFrame({'median_house_value': final_predictions})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {final_validation_score}')
