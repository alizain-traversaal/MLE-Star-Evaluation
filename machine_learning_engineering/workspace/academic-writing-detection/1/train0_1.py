
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import os

# Define the directory where the data is stored
data_dir = "./input"

# Load the training data
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)

# Fit the vectorizer to the training data and transform the training and validation data
X_train_vectors = vectorizer.fit_transform(X_train)
X_val_vectors = vectorizer.transform(X_val)

# Train a logistic regression model
model1 = LogisticRegression(random_state=42)
model1.fit(X_train_vectors, y_train)

# Train a Multinomial Naive Bayes classifier
model2 = MultinomialNB()
model2.fit(X_train_vectors, y_train)

# Create an ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', model1), ('nb', model2)], voting='soft')
ensemble_model.fit(X_train_vectors, y_train)

# Make predictions on the validation set
y_pred = ensemble_model.predict(X_val_vectors)

# Evaluate the model
final_validation_score = accuracy_score(y_val, y_pred)
print(f'Final Validation Performance: {final_validation_score}')
