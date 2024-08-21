import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

# Load test data
test_data = pd.read_csv('test_dataset_without_labels.csv', encoding='utf-16-le')

# Load trained model
model = joblib.load('trained_model.joblib')
w2v_model = joblib.load('w2v_trained_model.joblib')

# Preprocess test data
X_test = test_data['vba_code']
X_test_clean = test_data['vba_code'].apply(lambda x: gensim.utils.simple_preprocess(x))

# Load TF-IDF vectorizer and transform test data
vectorizer = joblib.load('tfidf_vectorizer.joblib')
X_test_tfidf = vectorizer.transform(test_data['vba_code'])

# Extract text lengths
X_test_text_length = test_data['vba_code'].apply(len)

# Generate Word2Vec word vectors for the test data
words = set(w2v_model.wv.index_to_key)
X_test_word2vec = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                            for ls in X_test], dtype="object")
X_test_word2vec = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                            for ls in X_test_clean],dtype="object")

# Average the word vectors for each sentence
X_test_word2vec_avg = np.array([v.mean(axis=0) if v.size else np.zeros(100) for v in X_test_word2vec])

# Combine TF-IDF features, Word2Vec features, and text lengths
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_word2vec_avg, X_test_text_length.values.reshape(-1, 1)))

# Make predictions
predictions = model.predict(X_test_combined)

# Replace numerical predictions with labels
label_mapping = {1: 'white', 0: 'mal'}
predictions_labels = [label_mapping[p] for p in predictions]
                      
# Save predictions
submission = pd.DataFrame({'prediction': predictions_labels})
submission.to_csv('test_prediction.csv', index=False)

print("Predictions saved to test_prediction.csv")
