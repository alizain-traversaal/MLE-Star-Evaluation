
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

# Baseline: TF-IDF vectorizer with stop words and max_df
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectors = vectorizer.fit_transform(X_train)
X_val_vectors = vectorizer.transform(X_val)
model = LogisticRegression(random_state=42)
model.fit(X_train_vectors, y_train)
y_pred = model.predict(X_val_vectors)
baseline_accuracy = accuracy_score(y_val, y_pred)
print(f'Baseline Validation Performance: {baseline_accuracy}')

# Ablation 1: Remove stop words
vectorizer_no_stopwords = TfidfVectorizer(stop_words=None, max_df=0.7)
X_train_vectors_no_stopwords = vectorizer_no_stopwords.fit_transform(X_train)
X_val_vectors_no_stopwords = vectorizer_no_stopwords.transform(X_val)
model_no_stopwords = LogisticRegression(random_state=42)
model_no_stopwords.fit(X_train_vectors_no_stopwords, y_train)
y_pred_no_stopwords = model_no_stopwords.predict(X_val_vectors_no_stopwords)
accuracy_no_stopwords = accuracy_score(y_val, y_pred_no_stopwords)
print(f'Validation Performance without Stop Words: {accuracy_no_stopwords}')

# Ablation 2: Remove max_df
vectorizer_no_max_df = TfidfVectorizer(stop_words='english', max_df=1.0)
X_train_vectors_no_max_df = vectorizer_no_max_df.fit_transform(X_train)
X_val_vectors_no_max_df = vectorizer_no_max_df.transform(X_val)
model_no_max_df = LogisticRegression(random_state=42)
model_no_max_df.fit(X_train_vectors_no_max_df, y_train)
y_pred_no_max_df = model_no_max_df.predict(X_val_vectors_no_max_df)
accuracy_no_max_df = accuracy_score(y_val, y_pred_no_max_df)
print(f'Validation Performance without max_df: {accuracy_no_max_df}')

if baseline_accuracy > accuracy_no_stopwords and baseline_accuracy > accuracy_no_max_df:
    print("Stop words and max_df contribute positively to the model.")
elif accuracy_no_stopwords > baseline_accuracy and accuracy_no_stopwords > accuracy_no_max_df:
    print("Removing stop words improves the model.")
elif accuracy_no_max_df > baseline_accuracy and accuracy_no_max_df > accuracy_no_stopwords:
    print("Removing max_df improves the model.")
else:
    print("No clear winner, further ablation studies are needed.")
