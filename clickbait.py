"""
File to fill with fun clickbait analysis!
Love, Lucino
"""

from newsapi import NewsApiClient
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import random

def create_labeled_data():
    news_api = NewsApiClient(api_key='99a7e430ec8f4cc1b2e6a6b77b65a5bc')
    buzzfeed_headlines = []
    reuters_headlines = []
    x = 1
    while x <= 10:
        b_headlines = news_api.get_everything(sources='buzzfeed',
                                              from_param='2018-11-11',
                                              to='2018-12-09',
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100,
                                              page=x)['articles']
        r_headlines = news_api.get_everything(sources='reuters',
                                              from_param='2018-11-11',
                                              to='2018-12-09',
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100,
                                              page=x)['articles']
        b_titles = [article['title'] for article in b_headlines]
        b_titles = list(filter(None.__ne__,b_titles))
        b_titles = [(article, 'bait') for article in b_titles]
        buzzfeed_headlines.extend(b_titles)
        r_titles = [(article['title'], 'not_bait') for article in r_headlines]
        reuters_headlines.extend(r_titles)
        x += 1

    all_headlines = buzzfeed_headlines + reuters_headlines
    random.shuffle(all_headlines)
    return all_headlines

def create_feature_sets(labeled_data):
	featuresets = [(bait_features(headline), label) for (headline, label) in labeled_data]
	train_set, test_set = featuresets[:1900], featuresets[1900:]
	return train_set, test_set
	
def bait_features(headline):
	featureset = {}
	featureset['pcount'] = pcount(headline)
	featureset['punctcount'] = punctcount(headline)
	featureset['averagewordlength'] = averagewordlength(headline)
	return featureset
	
	
def pcount(headline):
	pronouns = ["we","you","i","everyone","us","your","our"]
	for w in word_tokenize(headline):
		if w.lower() in pronouns:
			return True
	return False

def punctcount(headline):
	count = 0
	punct = [".","!","?"]
	for w in word_tokenize(headline):
		if w in punct:
			count += 1
	return count > 0
	
def averagewordlength(headline):
    charactercount = 0;
    wordcount = 0;
    commonshortwords = ["the", "a", "for", "an", "of", "and", "so", "but", "with", ",",".",":",";"]
    for w in word_tokenize(headline):
        if w.lower() not in commonshortwords:
            charactercount += len(w)
            wordcount += 1
    avg = charactercount/wordcount
    return avg < 4
	
def train_classifier(training_set):
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    return classifier

def evaluate_classifier(classifier, test_set):
    print(nltk.classify.accuracy(classifier, test_set))


if __name__ == '__main__':
    labeled_data = create_labeled_data()
    training_set, test_set = create_feature_sets(labeled_data)
    classifier = train_classifier(training_set)
    evaluate_classifier(classifier, test_set)
    classifier.show_most_informative_features()