import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
import joblib



# Read in the data and clean up column names
pd.set_option('display.max_colwidth', 100)

# Read the training dataset
messages_train = pd.read_csv('train_dataset.csv', encoding='utf-16-le')
messages_train = messages_train.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, errors='ignore')
messages_train.columns = ["label", "text"]

# Read the validation dataset
messages_test = pd.read_csv('validation_dataset.csv', encoding='utf-16-le')
messages_test = messages_test.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, errors='ignore')
messages_test.columns = ["label", "text"]

# Clean data using the built-in cleaner in gensim
messages_train['text_clean'] = messages_train['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
messages_test['text_clean'] = messages_test['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

# Map labels to numerical values
messages_train['label'] = messages_train['label'].map({'white': 1, 'mal': 0})
messages_test['label'] = messages_test['label'].map({'white': 1, 'mal': 0})

# Add a new feature representing the length of each text message in terms of characters
messages_train['text_length_chars'] = messages_train['text'].apply(len)
messages_test['text_length_chars'] = messages_test['text'].apply(len)

# Split data into train and test sets
X_train, y_train = messages_train['text'], messages_train['label']
X_test, y_test = messages_test['text'], messages_test['label']

# Initialize and fit TF-IDF vectorizer on training data
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform test data using fitted TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train the Word2Vec model
w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=100,
                                   window=5,
                                   min_count=2)

words = set(w2v_model.wv.index_to_key)

# Generate word vectors for each word in the sentence
X_train_word2vec = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                             for ls in messages_train['text_clean']],dtype="object")

X_test_word2vec = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                            for ls in messages_test['text_clean']],dtype="object")

# Average the word vectors for each sentence
X_train_word2vec_avg = np.array([v.mean(axis=0) if v.size else np.zeros(100) for v in X_train_word2vec])
X_test_word2vec_avg = np.array([v.mean(axis=0) if v.size else np.zeros(100) for v in X_test_word2vec])

# Concatenate TF-IDF features with Word2Vec features and the new feature (text length in chars)
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_word2vec_avg, messages_train['text_length_chars'].values.reshape(-1, 1)))
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_word2vec_avg, messages_test['text_length_chars'].values.reshape(-1, 1)))

# Instantiate and fit a basic Random Forest model on top of the combined features
rf = RandomForestClassifier()
rf_model = rf.fit(X_train_combined, y_train)

# Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test_combined)

# Define the file path where you want to save the trained model& the w2v model
model_file_path = 'trained_model.joblib'
w2v_model_file_path = 'w2v_trained_model.joblib'
tfidf_vectorizer_path = 'tfidf_vectorizer.joblib'


# Save the trained model to the specified file path
joblib.dump(rf_model, model_file_path)
joblib.dump(w2v_model, w2v_model_file_path)
joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)



print("Trained model saved successfully at:", model_file_path)
print("Trained w2v model saved successfully at:", w2v_model_file_path)
print("tfidf vectorizer model saved successfully at:", tfidf_vectorizer_path)



# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract values from confusion matrix
false_positive = conf_matrix[0, 1]  # Predicted white, but actually mal
false_negative = conf_matrix[1, 0]  # Predicted mal, but actually white

print('False Positive Count:', false_positive)
print('False Negative Count:', false_negative)

# Evaluate the predictions of the model on the holdout test set
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = (y_pred == y_test).sum() / len(y_pred)
print('Precision: {:.3f}, Recall: {:.3f}, Accuracy: {:.3f}'.format(precision, recall, accuracy))