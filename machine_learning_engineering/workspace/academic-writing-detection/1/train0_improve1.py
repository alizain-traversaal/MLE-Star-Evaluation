
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Subsample the training data
train_df = train_df.sample(frac=0.5, random_state=42)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_df['text'], train_df['label'], test_size=0.2, random_state=42
)

# Define custom stop words
custom_stop_words = [
    'the', 'and', 'a', 'of', 'to', 'in', 'is', 'it', 'that',
    'for', 'on', 'with', 'as', 'are', 'be', 'this', 'by', 'an',
    'have', 'from', 'at', 'was', 'were', 'which', 'but', 'not',
    'can', 'do', 'or', 'will', 'if', 'more', 'than', 'other',
    'has', 'also', 'may', 'its', 'been', 'such', 'these', 'those',
    'would', 'their', 'there', 'some', 'then', 'who', 'what',
    'when', 'where', 'how', 'so', 'up', 'down', 'out', 'in', 'over',
    'under', 'again', 'further', 'now', 'only', 'very', 'just',
    'no', 'yes', 'too', 'all', 'both', 'each', 'few', 'many',
    'most', 'same', 'several', 'any', 'one', 'two', 'three',
    'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
]

# Create a TF-IDF vectorizer with the custom stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stop_words)

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the validation data
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_pred)

# Print the final validation performance
print(f"Final Validation Performance: {accuracy}")
