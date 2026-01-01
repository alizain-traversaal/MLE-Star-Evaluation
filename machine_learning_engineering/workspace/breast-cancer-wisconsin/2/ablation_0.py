
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: With Scaling and Original Parameters
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LightGBM model parameters
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train_scaled, label=y_train)
test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

# Train the model
model = lgb.train(lgbm_params, train_data, num_boost_round=100, valid_sets=[train_data, test_data],
                  valid_names=['train','valid'],
                  callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
roc_auc = roc_auc_score(y_test, y_pred)
print(f"Baseline ROC AUC (Scaling + Original Params): {roc_auc}")

# Ablation 1: Without Scaling
X_train_no_scale, X_test_no_scale, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data_no_scale = lgb.Dataset(X_train_no_scale, label=y_train)
test_data_no_scale = lgb.Dataset(X_test_no_scale, label=y_test, reference=train_data_no_scale)

model_no_scale = lgb.train(lgbm_params, train_data_no_scale, num_boost_round=100, valid_sets=[train_data_no_scale, test_data_no_scale],
                  valid_names=['train','valid'],
                  callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])

y_pred_no_scale = model_no_scale.predict(X_test_no_scale)
roc_auc_no_scale = roc_auc_score(y_test, y_pred_no_scale)
print(f"Ablation 1 ROC AUC (No Scaling): {roc_auc_no_scale}")

# Ablation 2: Reduced num_leaves
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)

lgbm_params_reduced_leaves = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 15,  # Reduced num_leaves
    'max_depth': -1,
    'seed': 42
}

train_data_reduced_leaves = lgb.Dataset(X_train_scaled, label=y_train)
test_data_reduced_leaves = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data_reduced_leaves)

model_reduced_leaves = lgb.train(lgbm_params_reduced_leaves, train_data_reduced_leaves, num_boost_round=100, valid_sets=[train_data_reduced_leaves, test_data_reduced_leaves],
                  valid_names=['train','valid'],
                  callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])

y_pred_reduced_leaves = model_reduced_leaves.predict(X_test_scaled)
roc_auc_reduced_leaves = roc_auc_score(y_test, y_pred_reduced_leaves)
print(f"Ablation 2 ROC AUC (Reduced num_leaves): {roc_auc_reduced_leaves}")

if roc_auc > roc_auc_no_scale and roc_auc > roc_auc_reduced_leaves:
    print("Feature scaling and original num_leaves contribute the most to the overall performance.")
elif roc_auc_no_scale > roc_auc and roc_auc_no_scale > roc_auc_reduced_leaves:
    print("Removing feature scaling helps the most to the overall performance.")
else:
    print("Reducing num_leaves helps the most to the overall performance.")
