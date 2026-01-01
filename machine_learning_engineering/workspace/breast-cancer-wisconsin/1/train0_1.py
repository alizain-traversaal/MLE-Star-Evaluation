
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
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

# Models
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100, random_state=42)
rf_model = RandomForestClassifier(n_estimators=200,
                             max_depth=7,
                             random_state=42,
                             n_jobs=-1,
                             class_weight='balanced')

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_mlp = np.zeros((len(train),))
test_preds_mlp = np.zeros((len(test),))
oof_preds_rf = np.zeros((len(train),))
test_preds_rf = np.zeros((len(test),))


for fold, (train_idx, val_idx) in enumerate(skf.split(train[FEATURES], train[TARGET])):
    X_train, y_train = train.loc[train_idx, FEATURES], train.loc[train_idx, TARGET]
    X_val, y_val = train.loc[val_idx, FEATURES], train.loc[val_idx, TARGET]

    mlp_model.fit(X_train, y_train)
    oof_preds_mlp[val_idx] = mlp_model.predict_proba(X_val)[:, 1]
    test_preds_mlp += mlp_model.predict_proba(test[FEATURES])[:, 1] / skf.get_n_splits()
    
    rf_model.fit(X_train, y_train)
    oof_preds_rf[val_idx] = rf_model.predict_proba(X_val)[:, 1]
    test_preds_rf += rf_model.predict_proba(test[FEATURES])[:, 1] / skf.get_n_splits()

oof_score_mlp = roc_auc_score(train[TARGET], oof_preds_mlp)
oof_score_rf = roc_auc_score(train[TARGET], oof_preds_rf)
print(f"OOF ROC AUC MLP: {oof_score_mlp}")
print(f"OOF ROC AUC RF: {oof_score_rf}")

# Ensemble
oof_preds = 0.5 * oof_preds_mlp + 0.5 * oof_preds_rf
test_preds = 0.5 * test_preds_mlp + 0.5 * test_preds_rf

oof_score = roc_auc_score(train[TARGET], oof_preds)

# Create submission file
submission = pd.DataFrame({'Class': test_preds})
submission.to_csv('submission.csv', index=False)

print(f'Final Validation Performance: {oof_score}')
