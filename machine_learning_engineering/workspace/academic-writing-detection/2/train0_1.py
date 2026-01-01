
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

# Load the training data
train_df = pd.read_csv("./input/train.csv")

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

# Train Logistic Regression model
lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(train_vectors, train_labels)

# Train Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(train_vectors, train_labels)

# Make predictions on the validation set
lr_val_predictions = lr_model.predict(val_vectors)
nb_val_predictions = nb_model.predict(val_vectors)

# Evaluate the models
lr_val_score = accuracy_score(val_labels, lr_val_predictions)
nb_val_score = accuracy_score(val_labels, nb_val_predictions)
print(f"LR Validation Accuracy: {lr_val_score}")
print(f"NB Validation Accuracy: {nb_val_score}")

# Ensemble the models (VotingClassifier)
voting_clf = VotingClassifier(estimators=[('lr', lr_model), ('nb', nb_model)], voting='soft')
voting_clf.fit(train_vectors, train_labels)
ensemble_val_predictions = voting_clf.predict(val_vectors)

# Evaluate the ensemble model
final_validation_score = accuracy_score(val_labels, ensemble_val_predictions)
print(f"Final Validation Performance: {final_validation_score}")

# Train the models on the entire training dataset
full_train_vectors = tfidf_vectorizer.fit_transform(train_df['text'])

lr_model.fit(full_train_vectors, train_df['label'])
nb_model.fit(full_train_vectors, train_df['label'])
voting_clf.fit(full_train_vectors, train_df['label'])

# Load the unlabeled test data
test_unlabeled_df = pd.read_csv("./input/test_unlabeled.csv")

# Transform the unlabeled test data
test_vectors = tfidf_vectorizer.transform(test_unlabeled_df['text'])

# Make predictions on the unlabeled test data using the ensemble model
test_predictions = voting_clf.predict(test_vectors)

# Create a submission DataFrame
submission_df = pd.DataFrame({'label': test_predictions})

# Save the submission file
submission_df.to_csv("submission.csv", index=False)
