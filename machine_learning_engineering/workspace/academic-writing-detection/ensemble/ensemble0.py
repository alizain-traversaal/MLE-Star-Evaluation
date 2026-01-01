
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Solution 1
train_df = pd.read_csv("./input/train.csv")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_df=0.7)

# Fit the vectorizer to the training data and transform the training and validation data
X_train_vectors = vectorizer.fit_transform(X_train)
X_val_vectors = vectorizer.transform(X_val)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_vectors, y_train)

# Make predictions on the validation set
sol1_val_probs = model.predict_proba(X_val_vectors)

# Load the unlabeled test data
test_unlabeled_df = pd.read_csv("./input/test_unlabeled.csv")

# Transform the unlabeled test data
test_vectors = vectorizer.transform(test_unlabeled_df['text'])

# Make predictions on the unlabeled test data
sol1_test_probs = model.predict_proba(test_vectors)

# Solution 2
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test_targets.csv")

train_text, val_text, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label'], test_size=0.2, random_state=42, stratify=train_df['label']
)

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=None, ngram_range=(1, 2))

train_vectors = tfidf_vectorizer.fit_transform(train_text)
val_vectors = tfidf_vectorizer.transform(val_text)

model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(train_vectors, train_labels)

sol2_val_probs = model.predict_proba(val_vectors)

test_unlabeled_df = pd.read_csv("./input/test_unlabeled.csv")
test_vectors = tfidf_vectorizer.transform(test_unlabeled_df['text'])
sol2_test_probs = model.predict_proba(test_vectors)

# Ensemble
alpha = 0.5
ensemble_val_probs = alpha * sol1_val_probs + (1 - alpha) * sol2_val_probs
ensemble_test_probs = alpha * sol1_test_probs + (1 - alpha) * sol2_test_probs

ensemble_val_preds = np.argmax(ensemble_val_probs, axis=1)
val_labels_numeric = np.where(np.array(val_labels) == 'human', 0, 1)
final_validation_score = accuracy_score(val_labels_numeric, ensemble_val_preds)

ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)
submission_df = pd.DataFrame({'label': ['human' if i == 0 else 'machine' for i in ensemble_test_preds]})
submission_df.to_csv("submission.csv", index=False)

print(f'Final Validation Performance: {final_validation_score}')
