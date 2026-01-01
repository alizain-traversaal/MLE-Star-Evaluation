
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("./input/train.csv")

# Preprocessing
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM model parameters
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'verbose': -1
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train the LightGBM model
model_lgbm = lgb.train(lgbm_params, train_data, num_boost_round=100, valid_sets=[train_data, test_data],
                  valid_names=['train','valid'],
                  callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])

# Make predictions on the test set (LightGBM)
y_pred_lgbm = model_lgbm.predict(X_test)

# --- RandomForestClassifier ---
# RandomForest model parameters
rf_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': 42,
    'n_jobs': -1
}

# Train the RandomForest model
model_rf = RandomForestClassifier(**rf_params)
model_rf.fit(X_train, y_train)

# Make predictions on the test set (RandomForest)
y_pred_rf = model_rf.predict_proba(X_test)[:, 1]

# --- Weighted Averaging Ensemble ---
# Combine predictions using weighted average
weight_lgbm = 0.7
weight_rf = 0.3
y_pred_ensemble = (weight_lgbm * y_pred_lgbm) + (weight_rf * y_pred_rf)

# Evaluate the ensemble model
roc_auc_ensemble = roc_auc_score(y_test, y_pred_ensemble)
print(f"Ensemble ROC AUC: {roc_auc_ensemble}")


# Prepare submission (assuming test.csv has the same features as train.csv)
test_df = pd.read_csv("./input/test.csv")
X_test_submission = test_df.copy()
X_test_submission = scaler.transform(X_test_submission)

# Make predictions on the test set (LightGBM)
y_pred_lgbm_submission = model_lgbm.predict(X_test_submission)

# Make predictions on the test set (RandomForest)
y_pred_rf_submission = model_rf.predict_proba(X_test_submission)[:, 1]

# Combine predictions using weighted average
y_pred_ensemble_submission = (weight_lgbm * y_pred_lgbm_submission) + (weight_rf * y_pred_rf_submission)

# Convert probabilities to binary predictions (0 or 1)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred_ensemble_submission]

submission_df = pd.DataFrame({'Class': y_pred_binary})
submission_df.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {roc_auc_ensemble}")
