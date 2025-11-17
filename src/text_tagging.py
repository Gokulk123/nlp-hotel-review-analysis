from setup import nlp

def pos_tagging(text):
    text_new = ' '.join(text)
    doc = nlp(text_new)
    return[(t.text, t.pos_) for t in doc]

def ner_tagging(text):
    text_new = ' '.join(text)
    doc = nlp(text_new)
    return [(t.text, t.label_) for t in doc.ents]