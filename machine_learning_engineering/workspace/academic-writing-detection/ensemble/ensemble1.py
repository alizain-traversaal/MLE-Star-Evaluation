
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import rankdata


# Solution 1
train_df1 = pd.read_csv("./input/train.csv")
X_train1, X_val1, y_train1, y_val1 = train_test_split(train_df1['text'], train_df1['label'], test_size=0.2, random_state=42)
vectorizer1 = TfidfVectorizer(max_df=0.7)
X_train_vectors1 = vectorizer1.fit_transform(X_train1)
X_val_vectors1 = vectorizer1.transform(X_val1)
model1 = LogisticRegression(random_state=42)
model1.fit(X_train_vectors1, y_train1)


# Solution 2
train_df2 = pd.read_csv("./input/train.csv")
test_unlabeled_df2 = pd.read_csv("./input/test_unlabeled.csv")
train_text2, val_text2, train_labels2, val_labels2 = train_test_split(
    train_df2['text'], train_df2['label'], test_size=0.2, random_state=42, stratify=train_df2['label']
)
tfidf_vectorizer2 = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=None, ngram_range=(1, 2))
train_vectors2 = tfidf_vectorizer2.fit_transform(train_text2)
val_vectors2 = tfidf_vectorizer2.transform(val_text2)
model2 = LogisticRegression(solver='liblinear', random_state=42)
model2.fit(train_vectors2, train_labels2)

# -- Validation Set Prediction --
val_predictions1 = model1.predict(X_val_vectors1)
val_predictions2 = model2.predict(val_vectors2)
accuracy1 = accuracy_score(y_val1, val_predictions1)
accuracy2 = accuracy_score(val_labels2, val_predictions2)
print(f"Validation Accuracy Sol1: {accuracy1}")
print(f"Validation Accuracy Sol2: {accuracy2}")


# -- Rank Ensemble on Validation Set --
val_probs1 = model1.predict_proba(X_val_vectors1)[:, 1]  # Probability of 'human'
val_probs2 = model2.predict_proba(val_vectors2)[:, 1]  # Probability of 'human'
val_ranks1 = rankdata(val_probs1, axis=0)
val_ranks2 = rankdata(val_probs2, axis=0)
avg_val_ranks = (val_ranks1 + val_ranks2) / 2.0
threshold = np.median(avg_val_ranks)
val_ensemble_predictions = ['human' if rank > threshold else 'machine' for rank in avg_val_ranks]
final_validation_score = accuracy_score(y_val1, val_ensemble_predictions)

print(f'Final Validation Performance: {final_validation_score}')


# -- Preparing Submission --
# Train full model for sol1
vectorizer1_full = TfidfVectorizer(max_df=0.7)
X_train_vectors1_full = vectorizer1_full.fit_transform(train_df1['text'])
model1_full = LogisticRegression(random_state=42)
model1_full.fit(X_train_vectors1_full, train_df1['label'])

# Train full model for sol2
tfidf_vectorizer2_full = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=None, ngram_range=(1, 2))
full_train_vectors2 = tfidf_vectorizer2_full.fit_transform(train_df2['text'])
model2_full = LogisticRegression(solver='liblinear', random_state=42)
model2_full.fit(full_train_vectors2, train_df2['label'])

# -- Rank Ensemble on Test Set --
test_vectors1 = vectorizer1_full.transform(test_unlabeled_df2['text'])
test_vectors2 = tfidf_vectorizer2_full.transform(test_unlabeled_df2['text'])
test_probs1 = model1_full.predict_proba(test_vectors1)[:, 1]
test_probs2 = model2_full.predict_proba(test_vectors2)[:, 1]
test_ranks1 = rankdata(test_probs1, axis=0)
test_ranks2 = rankdata(test_probs2, axis=0)
avg_test_ranks = (test_ranks1 + test_ranks2) / 2.0
threshold = np.median(avg_test_ranks)  # Use median from test set ranks
test_ensemble_predictions = ['human' if rank > threshold else 'machine' for rank in avg_test_ranks]

submission_df = pd.DataFrame({'label': test_ensemble_predictions})
submission_df.to_csv("submission.csv", index=False)
