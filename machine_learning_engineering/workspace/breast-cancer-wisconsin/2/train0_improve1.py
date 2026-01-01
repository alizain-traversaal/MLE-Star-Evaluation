
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv("./input/train.csv")

# Preprocessing
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# LightGBM model parameters
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'min_child_samples': 20 # Added min_child_samples
}


# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train the model
model = lgb.train(lgbm_params, train_data, num_boost_round=100, valid_sets=[train_data, test_data],
                  valid_names=['train','valid'],
                  callbacks=[lgb.early_stopping(stopping_rounds=10)])

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC: {roc_auc}")

# Prepare submission (assuming test.csv has the same features as train.csv)
test_df = pd.read_csv("./input/test.csv")
X_test_submission = test_df.copy()
X_test_submission = scaler.transform(X_test_submission)

y_pred_submission = model.predict(X_test_submission)

# Convert probabilities to binary predictions (0 or 1)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred_submission]

submission_df = pd.DataFrame({'Class': y_pred_binary})
submission_df.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {roc_auc}")
