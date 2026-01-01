
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# Load the training and testing data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Combine for preprocessing
all_df = pd.concat([train_df.drop('median_house_value', axis=1), test_df], axis=0)

# Impute missing values using the median
for col in all_df.columns:
    if all_df[col].isnull().any():
        median_val = all_df[col].median()
        all_df[col] = all_df[col].fillna(median_val)

# Scale the numerical features
numerical_cols = all_df.columns
scaler = StandardScaler()
all_df[numerical_cols] = scaler.fit_transform(all_df[numerical_cols])

# Split back into training and testing sets
X = all_df[:len(train_df)]
X_test = all_df[len(train_df):]
y = train_df['median_house_value']

# Define the model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_df))
rmse_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    oof_predictions[val_index] = val_predictions

    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    rmse_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse}")

# Calculate the final validation score
final_validation_score = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"Final Validation Performance: {final_validation_score}")

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Create the submission file
submission = pd.DataFrame({'median_house_value': test_predictions})
submission.to_csv('submission.csv', index=False)
