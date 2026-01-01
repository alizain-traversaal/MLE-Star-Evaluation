
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
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

# Model: RandomForestClassifier
model = RandomForestClassifier(n_estimators=200,
                             max_depth=7,
                             random_state=42,
                             n_jobs=-1,
                             class_weight='balanced')

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train),))
test_preds = np.zeros((len(test),))

for fold, (train_idx, val_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    X_train, y_train = train.loc[train_idx, FEATURES], train.loc[train_idx, TARGET]
    X_val, y_val = train.loc[val_idx, FEATURES], train.loc[val_idx, TARGET]

    model.fit(X_train, y_train)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test[FEATURES])[:, 1] / skf.get_n_splits()

oof_score = roc_auc_score(train[TARGET], oof_preds)
print(f"OOF ROC AUC: {oof_score}")

# Create submission file
submission = pd.DataFrame({'Class': test_preds})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {oof_score}')
