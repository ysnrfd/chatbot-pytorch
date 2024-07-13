# nltk_utils.py

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK resources if needed
nltk.download('punkt')
nltk.download('wordnet')

# Tokenization function
def tokenize_words(sentence):
    return nltk.word_tokenize(sentence)

# Stemming function
def stem(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word.lower())

# Bag of words function
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bow = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bow[idx] = 1
    return bow
