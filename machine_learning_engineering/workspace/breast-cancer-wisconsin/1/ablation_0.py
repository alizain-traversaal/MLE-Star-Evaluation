
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_csv('./input/train.csv')

# Drop 'id' column
if 'id' in train.columns:
    train = train.drop('id', axis=1)

# Define target and features
TARGET = 'Class'
FEATURES = [col for col in train.columns if col not in ['id', TARGET]]

# --- Ablation 1: No Scaling ---
print("Ablation 1: No Scaling")
train_no_scaling = train.copy()
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_no_scaling = np.zeros((len(train_no_scaling),))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_no_scaling[FEATURES], train_no_scaling[TARGET])):
    X_train, y_train = train_no_scaling.loc[train_idx, FEATURES], train_no_scaling.loc[train_idx, TARGET]
    X_val, y_val = train_no_scaling.loc[val_idx, FEATURES], train_no_scaling.loc[val_idx, TARGET]

    model.fit(X_train, y_train)
    oof_preds_no_scaling[val_idx] = model.predict_proba(X_val)[:, 1]

oof_score_no_scaling = roc_auc_score(train_no_scaling[TARGET], oof_preds_no_scaling)
print(f"OOF ROC AUC (No Scaling): {oof_score_no_scaling}")

# --- Ablation 2: No hidden layers ---
print("\nAblation 2: No Hidden Layers")
train_no_hidden = train.copy()
scaler = MinMaxScaler()
train_no_hidden[FEATURES] = scaler.fit_transform(train_no_hidden[FEATURES])
model = MLPClassifier(hidden_layer_sizes=(), activation='relu', solver='adam', max_iter=100, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_no_hidden = np.zeros((len(train_no_hidden),))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_no_hidden[FEATURES], train_no_hidden[TARGET])):
    X_train, y_train = train_no_hidden.loc[train_idx, FEATURES], train_no_hidden.loc[train_idx, TARGET]
    X_val, y_val = train_no_hidden.loc[val_idx, FEATURES], train_no_hidden.loc[val_idx, TARGET]

    model.fit(X_train, y_train)
    oof_preds_no_hidden[val_idx] = model.predict_proba(X_val)[:, 1]

oof_score_no_hidden = roc_auc_score(train_no_hidden[TARGET], oof_preds_no_hidden)
print(f"OOF ROC AUC (No Hidden Layers): {oof_score_no_hidden}")

# --- Original Model ---
print("\nOriginal Model")
train_original = train.copy()
scaler = MinMaxScaler()
train_original[FEATURES] = scaler.fit_transform(train_original[FEATURES])
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=100, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_original = np.zeros((len(train_original),))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_original[FEATURES], train_original[TARGET])):
    X_train, y_train = train_original.loc[train_idx, FEATURES], train_original.loc[train_idx, TARGET]
    X_val, y_val = train_original.loc[val_idx, FEATURES], train_original.loc[val_idx, TARGET]

    model.fit(X_train, y_train)
    oof_preds_original[val_idx] = model.predict_proba(X_val)[:, 1]

oof_score_original = roc_auc_score(train_original[TARGET], oof_preds_original)
print(f"OOF ROC AUC (Original): {oof_score_original}")

print("\nConclusion:")
if oof_score_no_scaling < oof_score_original and oof_score_no_hidden < oof_score_original:
    print("Both scaling and hidden layers contribute positively to the model's performance.")
elif oof_score_no_scaling < oof_score_original:
    print("Scaling the data contributes more to the model's performance than hidden layers.")
elif oof_score_no_hidden < oof_score_original:
    print("Adding hidden layers contributes more to the model's performance than scaling.")
else:
    print("Neither scaling nor hidden layers contribute positively, or their impact is negligible with the current configuration.")
