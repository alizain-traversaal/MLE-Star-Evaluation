
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

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


from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, randint

# Define the parameter grid
param_distributions = {
    'hidden_layer_sizes': [(64,), (64, 32), (128, 64), (128, 64, 32)],
    'max_iter': randint(100, 300),
    'alpha': uniform(0.0001, 0.01),
}

# Create the model
mlp = MLPClassifier(activation='relu', solver='adam', random_state=42)

# Instantiate the randomized search model
randomized_search = RandomizedSearchCV(estimator=mlp, param_distributions=param_distributions, 
                                      n_iter=10, cv=3, scoring='roc_auc', verbose=0, random_state=42, n_jobs=-1)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_1 = np.zeros((len(train),))
test_preds_1 = np.zeros((len(test),))

for fold, (train_idx, val_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    X_train, y_train = train.loc[train_idx, FEATURES], train.loc[train_idx, TARGET]
    X_val, y_val = train.loc[val_idx, FEATURES], train.loc[val_idx, TARGET]

    # Fit the randomized search to the data
    randomized_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = randomized_search.best_params_

    # Create the model with the best parameters
    model = MLPClassifier(**best_params, activation='relu', solver='adam', random_state=42)


    model.fit(X_train, y_train)
    
    oof_preds_1[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_1 += model.predict_proba(test[FEATURES])[:, 1] / skf.get_n_splits()

oof_score_1 = roc_auc_score(train[TARGET], oof_preds_1)
print(f"Solution 1 OOF ROC AUC: {oof_score_1}")



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
    'lambda_l1': 0.1,  # L1 regularization
    'lambda_l2': 0.1   # L2 regularization
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
print(f"Solution 2 Validation ROC AUC: {roc_auc}")

# Prepare submission (assuming test.csv has the same features as train.csv)
test_df = pd.read_csv("./input/test.csv")
X_test_submission = test_df.copy()
X_test_submission = scaler.transform(X_test_submission)

test_preds_2 = model.predict(X_test_submission)

# Rank averaging ensemble
def rank_averaging(predictions):
    ranks = np.empty_like(predictions)
    ranks[np.argsort(predictions)] = np.arange(len(predictions))
    return ranks

rank_1 = rank_averaging(test_preds_1)
rank_2 = rank_averaging(test_preds_2)

average_rank = (rank_1 + rank_2) / 2
probability_predictions = (average_rank / len(test_preds_1))

# Refinement step
disagreement = np.abs(test_preds_1 - test_preds_2) > 0.8  #tune threshold

if oof_score_1 > roc_auc:
    final_predictions = np.where(disagreement, test_preds_1, probability_predictions)
    best_score = oof_score_1
else:
    final_predictions = np.where(disagreement, test_preds_2, probability_predictions)
    best_score = roc_auc

# Convert probabilities to binary predictions (0 or 1)
y_pred_binary = [1 if p >= 0.5 else 0 for p in final_predictions]

submission_df = pd.DataFrame({'Class': y_pred_binary})
submission_df.to_csv("./final/submission.csv", index=False)

print(f"Final Validation Performance: {best_score}")
