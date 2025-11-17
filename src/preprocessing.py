from setup import *
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

stopword = stopwords.words('english')
stemming = PorterStemmer()
lemmatize = WordNetLemmatizer()

# -------------------------------
# Cleaning Function (row wise)
# -------------------------------
def cleaning_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]","", text)
    return text

# -------------------------------
# Tokenize
# -------------------------------
def tokenize(text):
    return word_tokenize(text)

# -------------------------------
# Remove Stopwords
# -------------------------------
def remove_stopwords(text):
    # print(text)
    return [word for word in text if word not in stopword]

# -------------------------------
# Stemming
# -------------------------------
def stem_words(text):
    return [stemming.stem(word) for word in text]

# -------------------------------
# Lemmatization
# -------------------------------
def lemmatize_words(text):
    return [lemmatize.lemmatize(word) for word in text]

# -------------------------------
# N-Grams
# -------------------------------
def generate_ngrams(tokens, n=2):
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]