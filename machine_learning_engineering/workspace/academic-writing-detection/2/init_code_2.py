
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Load the test data for validation
test_df = pd.read_csv("./input/test_targets.csv")

# Split training data for validation
train_text, val_text, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label'], test_size=0.2, random_state=42, stratify=train_df['label']
)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2))

# Fit and transform the training text
train_vectors = tfidf_vectorizer.fit_transform(train_text)

# Transform the validation text
val_vectors = tfidf_vectorizer.transform(val_text)

# Train a Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(train_vectors, train_labels)

# Make predictions on the validation set
val_predictions = model.predict(val_vectors)

# Evaluate the model
final_validation_score = accuracy_score(val_labels, val_predictions)
print(f"Final Validation Performance: {final_validation_score}")

# Train the model on the entire training dataset
full_train_vectors = tfidf_vectorizer.fit_transform(train_df['text'])
model.fit(full_train_vectors, train_df['label'])

# Load the unlabeled test data
test_unlabeled_df = pd.read_csv("./input/test_unlabeled.csv")

# Transform the unlabeled test data
test_vectors = tfidf_vectorizer.transform(test_unlabeled_df['text'])

# Make predictions on the unlabeled test data
test_predictions = model.predict(test_vectors)

# Create a submission DataFrame
submission_df = pd.DataFrame({'label': test_predictions})

# Save the submission file
submission_df.to_csv("submission.csv", index=False)
