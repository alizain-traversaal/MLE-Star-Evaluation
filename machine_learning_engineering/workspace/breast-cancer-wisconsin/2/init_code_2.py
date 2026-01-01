Okay, the code has been reviewed and it appears to be functionally correct. It performs data loading, preprocessing, model training, prediction, and submission file generation as expected. The ROC AUC score is also calculated and printed. There are no apparent errors. Therefore, no changes are needed. The final version of the code is the same as the original.

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# Preprocessing
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
train.iloc[:, :-1] = imputer.fit_transform(train.iloc[:, :-1])
test.iloc[:, :] = imputer.transform(test.iloc[:, :])

# Scale data
scaler = MinMaxScaler()
train.iloc[:, :-1] = scaler.fit_transform(train.iloc[:, :-1])
test.iloc[:, :] = scaler.transform(test.iloc[:, :])

X = train.drop('Class', axis=1)
y = train['Class']
X_test = test.copy()

# Subsampling
n_samples = 5000  # Number of samples to keep
if len(X) > n_samples:
    X_sampled = X.sample(n=n_samples, random_state=42)
    y_sampled = y[X_sampled.index]
    X = X_sampled
    y = y_sampled

# Model and cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

oof_score = roc_auc_score(y, oof_preds)
print(f"OOF ROC AUC: {oof_score}")

# Create submission file
submission = pd.DataFrame({'Class': test_preds})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {oof_score}')
