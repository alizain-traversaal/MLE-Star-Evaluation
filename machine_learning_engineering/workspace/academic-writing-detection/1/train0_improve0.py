
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Subsample data to reduce training time
human_df = train_df[train_df['label'] == 'human'].sample(n=500, random_state=42)
machine_df = train_df[train_df['label'] == 'machine'].sample(n=500, random_state=42)
train_df = pd.concat([human_df, machine_df], axis=0).sample(frac=1, random_state=42)

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
y_pred = model.predict(X_val_vectors)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f'Final Validation Performance: {accuracy}')
