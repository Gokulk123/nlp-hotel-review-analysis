from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


def BofW(text):
    bofw = CountVectorizer()
    new_text = ' '.join(text)
    bofw_fit = bofw.fit_transform([new_text])
    return (bofw_fit.toarray()[0],bofw.get_feature_names_out())

def Tfidf(text):
    tfidf = TfidfVectorizer()
    new_text = ' '.join(text)
    tfidf_fit = tfidf.fit_transform([new_text])
    return (tfidf_fit.toarray()[0],tfidf.get_feature_names_out())

def generate_topics(text_list, n_topics=5):
    documents = [" ".join(tokens) for tokens in text_list]

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_matrix = lda.fit_transform(tfidf_matrix)

    return topic_matrix, lda, tfidf
