
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Separate features (X) and target (y)
X = train_df.drop('class', axis=1)
y = train_df['class']

# Determine the number of classes
num_classes = len(y.unique())

# --- Function to train and evaluate a model for consistency ---
def train_and_evaluate(X_train, y_train, X_val, y_val, model_params):
    model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, **model_params)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    return accuracy_score(y_val, y_pred_val)

# --- Original Configuration ---
print("--- Original Configuration ---")
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_params_orig = {'random_state': 42}
original_accuracy = train_and_evaluate(X_train_orig, y_train_orig, X_val_orig, y_val_orig, model_params_orig)
print(f'Original Validation Performance: {original_accuracy}')
print("-" * 30)

# --- Ablation 1: Remove stratification from train_test_split ---
# This tests the impact of ensuring class distribution is similar across splits.
print("--- Ablation 1: No Stratification in train_test_split ---")
X_train_no_strat, X_val_no_strat, y_train_no_strat, y_val_no_strat = train_test_split(X, y, test_size=0.2, random_state=42) # Removed stratify=y
ablation1_accuracy = train_and_evaluate(X_train_no_strat, y_train_no_strat, X_val_no_strat, y_val_no_strat, model_params_orig)
print(f'Ablation 1 Performance (No Stratification): {ablation1_accuracy}')
print("-" * 30)

# --- Ablation 2: Remove random_state from LGBMClassifier ---
# This tests the impact of reproducibility on model training, introducing variability.
print("--- Ablation 2: No random_state in LGBMClassifier ---")
# Use original split for this ablation to isolate the change
X_train_ablation2, X_val_ablation2, y_train_ablation2, y_val_ablation2 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_params_no_random_state = {} # Removed random_state
ablation2_accuracy = train_and_evaluate(X_train_ablation2, y_train_ablation2, X_val_ablation2, y_val_ablation2, model_params_no_random_state)
print(f'Ablation 2 Performance (No Model Random State): {ablation2_accuracy}')
print("-" * 30)

# --- Conclusion ---
print("\n--- Ablation Study Summary ---")
print(f"Original Validation Performance: {original_accuracy}")
print(f"Ablation 1 (No Stratification): {ablation1_accuracy} (Change: {ablation1_accuracy - original_accuracy:.4f})")
print(f"Ablation 2 (No Model Random State): {ablation2_accuracy} (Change: {ablation2_accuracy - original_accuracy:.4f})")

# Determine which part contributed the most (or had the most impact when removed/modified)
performance_diffs = {
    'Stratification in train_test_split': abs(ablation1_accuracy - original_accuracy),
    'Random state in LGBMClassifier': abs(ablation2_accuracy - original_accuracy)
}

most_impactful_part = max(performance_diffs, key=performance_diffs.get)
print(f"\nBased on this ablation study, '{most_impactful_part}' had the most impact on the model's performance when modified or disabled.")
