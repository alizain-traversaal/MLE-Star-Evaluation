
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Define the directory where the data is stored
data_dir = "./input"

# Load the training data
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

# Split the training data into training and validation sets
train_text, val_text, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label'], test_size=0.2, random_state=42
)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Fit the vectorizer on the training text and transform both training and validation text
train_vectors = tfidf_vectorizer.fit_transform(train_text)
val_vectors = tfidf_vectorizer.transform(val_text)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(train_vectors, train_labels)

# Make predictions on the validation set
val_predictions = classifier.predict(val_vectors)

# Calculate the accuracy of the model on the validation set
final_validation_score = accuracy_score(val_labels, val_predictions)

# Print the final validation performance
print(f'Final Validation Performance: {final_validation_score}')

# Load the test data
test_df = pd.read_csv(os.path.join(data_dir, "test_unlabeled.csv"))

# Transform the test text using the fitted vectorizer
test_vectors = tfidf_vectorizer.transform(test_df['text'])

# Make predictions on the test set
test_predictions = classifier.predict(test_vectors)

# Create a submission dataframe
submission_df = pd.DataFrame({'id': test_df['id'], 'label': test_predictions})

# Save the submission dataframe to a CSV file
submission_df.to_csv('submission.csv', index=False)
