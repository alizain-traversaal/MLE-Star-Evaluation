
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Solution 1: Modified with RandomizedSearchCV
# Load data
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# Drop 'id' column
if 'id' in train.columns:
    train = train.drop('id', axis=1)
if 'id' in test.columns:
    test = test.drop('id', axis=1)

# Define target and features
TARGET = 'Class'
FEATURES = [col for col in train.columns if col not in ['id', TARGET]]

# Scale data
scaler = MinMaxScaler()
train[FEATURES] = scaler.fit_transform(train[FEATURES])
test[FEATURES] = scaler.transform(test[FEATURES])


# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'hidden_layer_sizes': [(64,), (64, 32), (128, 64), (128, 64, 32)],
    'max_iter': [100, 200, 300],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

# Create the model
mlp = MLPClassifier(activation='relu', solver='adam', random_state=42)

# Instantiate the randomized search model
random_search = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid, n_iter=10, cv=3, scoring='roc_auc', verbose=0, random_state=42)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_1 = np.zeros((len(train),))
test_preds_1 = np.zeros((len(test),))

for fold, (train_idx, val_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    X_train, y_train = train.loc[train_idx, FEATURES], train.loc[train_idx, TARGET]
    X_val, y_val = train.loc[val_idx, FEATURES], train.loc[val_idx, TARGET]

    # Fit the randomized search to the data
    random_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_

    # Create the model with the best parameters
    model_1 = MLPClassifier(**best_params, activation='relu', solver='adam', random_state=42)

    model_1.fit(X_train, y_train)
    
    oof_preds_1[val_idx] = model_1.predict_proba(X_val)[:, 1]
    test_preds_1 += model_1.predict_proba(test[FEATURES])[:, 1] / skf.get_n_splits()

oof_score_1 = roc_auc_score(train[TARGET], oof_preds_1)
print(f"Solution 1 OOF ROC AUC: {oof_score_1}")

# Solution 2: LightGBM
# Load data
df = pd.read_csv("./input/train.csv")

# Preprocessing
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale the features
scaler_lgbm = StandardScaler()
X = scaler_lgbm.fit_transform(X)

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
    'lambda_l1': 0.1,  # L1 regularization
    'lambda_l2': 0.1   # L2 regularization
}


# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train the model
model_2 = lgb.train(lgbm_params, train_data, num_boost_round=100, valid_sets=[train_data, test_data],
                  valid_names=['train','valid'],
                  callbacks=[lgb.early_stopping(stopping_rounds=10)],
                  )

# Make predictions on the test set
y_pred = model_2.predict(X_test)

# Evaluate the model
roc_auc_2 = roc_auc_score(y_test, y_pred)
print(f"Solution 2 ROC AUC: {roc_auc_2}")

# Prepare submission (assuming test.csv has the same features as train.csv)
test_df = pd.read_csv("./input/test.csv")
X_test_submission = test_df.copy()
X_test_submission = scaler_lgbm.transform(X_test_submission)

test_preds_2 = model_2.predict(X_test_submission)


# Ensemble with weighted average
weight_1 = 0.5
weight_2 = 0.5
final_preds = (weight_1 * test_preds_1) + (weight_2 * test_preds_2)

# Convert to binary predictions
final_preds_binary = [1 if p >= 0.5 else 0 for p in final_preds]

# Create submission file
submission = pd.DataFrame({'Class': final_preds_binary})
submission.to_csv('submission.csv', index=False)

# Calculate OOF score for ensembled predictions
ensembled_oof_preds = (weight_1 * oof_preds_1)  # Only using OOF from solution 1
ensembled_oof_binary = [1 if p >= 0.5 else 0 for p in ensembled_oof_preds]
oof_score_ensembled = roc_auc_score(train[TARGET], oof_preds_1)

print(f"Ensembled OOF ROC AUC: {oof_score_ensembled}")

print(f'Final Validation Performance: {oof_score_ensembled}')
