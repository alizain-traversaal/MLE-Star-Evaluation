
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    import nltk
    nltk.download('stopwords')

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    import nltk
    nltk.download('punkt')


# Load the data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test_unlabeled.csv")

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_text)

# Apply preprocessing
train_df['processed_text'] = train_df['text'].apply(preprocess_text)
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

# Split data
X_train, X_val, y_train, y_val = train_test_split(train_df['processed_text'], train_df['label'], test_size=0.2, random_state=42)

# Feature extraction
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)  # Limiting features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])

# Model training
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_tfidf, y_train)

# Validation
y_pred_val = model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {val_accuracy}")

# Prediction on test data (test_unlabeled.csv)
y_pred_test = model.predict(X_test_tfidf)

# Create submission DataFrame
submission_df = pd.DataFrame({'label': y_pred_test})
submission_df.to_csv("submission.csv", index=False)

#Evaluate on test_targets.csv
test_targets_df = pd.read_csv("./input/test_targets.csv")
test_targets_processed_text = test_targets_df['text'].apply(preprocess_text)
X_test_targets_tfidf = tfidf_vectorizer.transform(test_targets_processed_text)
y_pred_test_targets = model.predict(X_test_targets_tfidf)
test_accuracy = accuracy_score(test_targets_df['label'], y_pred_test_targets)
print(f"Final Validation Performance: {test_accuracy}")
