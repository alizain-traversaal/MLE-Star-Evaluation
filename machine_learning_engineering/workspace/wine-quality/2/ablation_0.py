
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('./input/train.csv')

# Split into features (X) and target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)**0.5
print(f'Baseline RMSE: {rmse}')

# Ablation 1: Removing a feature (volatile acidity)
X_train_ablation1 = X_train.drop('volatile acidity', axis=1)
X_test_ablation1 = X_test.drop('volatile acidity', axis=1)
xgb_ablation1 = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_ablation1.fit(X_train_ablation1, y_train)
y_pred_ablation1 = xgb_ablation1.predict(X_test_ablation1)
rmse_ablation1 = mean_squared_error(y_test, y_pred_ablation1)**0.5
print(f'Ablation 1 RMSE (Removing volatile acidity): {rmse_ablation1}')

# Ablation 2: Reducing the number of estimators
xgb_ablation2 = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
xgb_ablation2.fit(X_train, y_train)
y_pred_ablation2 = xgb_ablation2.predict(X_test)
rmse_ablation2 = mean_squared_error(y_test, y_pred_ablation2)**0.5
print(f'Ablation 2 RMSE (Reducing n_estimators to 50): {rmse_ablation2}')

if rmse_ablation1 > rmse and rmse_ablation2 > rmse:
    print("Conclusion: All ablations worsened performance. Baseline model is the best.")
elif rmse_ablation1 < rmse and rmse_ablation2 > rmse:
    print("Conclusion: Removing 'volatile acidity' improved the model. 'volatile acidity' may be a redundant or noisy feature.")
elif rmse_ablation1 > rmse and rmse_ablation2 < rmse:
    print("Conclusion: Reducing 'n_estimators' improved the model. The original model might have been overfitting.")
elif rmse_ablation1 < rmse and rmse_ablation2 < rmse:
    print("Conclusion: Both ablations improved the model. Consider further tuning.")
else:
    print("Conclusion: Mixed results. Further investigation needed.")
