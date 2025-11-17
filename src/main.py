import pandas as pd
from preprocessing import *;
from text_tagging import pos_tagging, ner_tagging;
from sentiment_analysis import *;
from vectorize_text import *;

# Load dataset
data = pd.read_csv('data/tripadvisor_hotel_reviews.csv')

############################################################################
# 1. Text Preprocessing
############################################################################

data['cleaned'] = data['Review'].apply(cleaning_text)
data['tokenize'] = data['cleaned'].apply(tokenize)
data['remove_stopwords'] = data['tokenize'].apply(remove_stopwords)
data['stemming'] = data['remove_stopwords'].apply(stem_words)
data['lemmatize'] = data['remove_stopwords'].apply(lemmatize_words)
data['n-grams'] = data['lemmatize'].apply(generate_ngrams)

############################################################################
# 2. Text Tagging
############################################################################

data["pos_tags"] = data['cleaned'].apply(pos_tagging)
data['ner_tags'] = data['cleaned'].apply(ner_tagging)

############################################################################
# 3. Sentiment Analysis
############################################################################

data['sentiment_analysis'] = data['lemmatize'].apply(sentiment_analysis)

############################################################################
# 4. Text vectorization
############################################################################

data['bag_of_word_vector'] = data['lemmatize'].apply(BofW)
data['tf_idf_vector'] = data['lemmatize'].apply(Tfidf)

# Topic Modeling
topic_matrix, lda_model, tfidf_model = generate_topics(data['lemmatize'], n_topics=5)
data['dominant_topic'] = topic_matrix.argmax(axis=1)

# Rating classification
data['label'] = data['Rating'].apply(lambda x: 1 if x >= 4 else 0)


print(data.head())