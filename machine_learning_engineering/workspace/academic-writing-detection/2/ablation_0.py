
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Split training data for validation
train_text, val_text, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label'], test_size=0.2, random_state=42, stratify=train_df['label']
)

# Baseline: TF-IDF with Logistic Regression
tfidf_vectorizer_baseline = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2))
train_vectors_baseline = tfidf_vectorizer_baseline.fit_transform(train_text)
val_vectors_baseline = tfidf_vectorizer_baseline.transform(val_text)
model_baseline = LogisticRegression(solver='liblinear', random_state=42)
model_baseline.fit(train_vectors_baseline, train_labels)
val_predictions_baseline = model_baseline.predict(val_vectors_baseline)
baseline_accuracy = accuracy_score(val_labels, val_predictions_baseline)
print(f"Baseline Accuracy: {baseline_accuracy}")

# Ablation 1: Removing stop words
tfidf_vectorizer_no_stopwords = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))  # Removed stop_words
train_vectors_no_stopwords = tfidf_vectorizer_no_stopwords.fit_transform(train_text)
val_vectors_no_stopwords = tfidf_vectorizer_no_stopwords.transform(val_text)
model_no_stopwords = LogisticRegression(solver='liblinear', random_state=42)
model_no_stopwords.fit(train_vectors_no_stopwords, train_labels)
val_predictions_no_stopwords = model_no_stopwords.predict(val_vectors_no_stopwords)
no_stopwords_accuracy = accuracy_score(val_labels, val_predictions_no_stopwords)
print(f"Accuracy without Stop Words: {no_stopwords_accuracy}")

# Ablation 2: Using only unigrams
tfidf_vectorizer_unigrams = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 1))  # Only unigrams
train_vectors_unigrams = tfidf_vectorizer_unigrams.fit_transform(train_text)
val_vectors_unigrams = tfidf_vectorizer_unigrams.transform(val_text)
model_unigrams = LogisticRegression(solver='liblinear', random_state=42)
model_unigrams.fit(train_vectors_unigrams, train_labels)
val_predictions_unigrams = model_unigrams.predict(val_vectors_unigrams)
unigrams_accuracy = accuracy_score(val_labels, val_predictions_unigrams)
print(f"Accuracy with only Unigrams: {unigrams_accuracy}")

if baseline_accuracy > no_stopwords_accuracy and baseline_accuracy > unigrams_accuracy:
    print("The original configuration (baseline) performed the best.")
elif no_stopwords_accuracy > baseline_accuracy and no_stopwords_accuracy > unigrams_accuracy:
    print("Removing stop words improved the performance the most.")
else:
    print("Using only unigrams improved the performance the most.")
