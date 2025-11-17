from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def sentiment_analysis(text):
  new_text = ' '.join(text)
  score = analyser.polarity_scores(new_text)
  return score['compound']