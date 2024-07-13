import nltk
from nltk.stem import SnowballStemmer
import numpy as np

# Download nltk resources if not already downloaded
nltk.download('punkt')

stemmer = SnowballStemmer('english')

def tokenize_words(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
