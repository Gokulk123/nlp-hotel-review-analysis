import nltk
import spacy

# Download NLTK resources only once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model only once
nlp = spacy.load('en_core_web_sm')
