"""

File to fill with fun clickbait analysis!
Love, Lucino

"""

from newsapi import NewsApiClient
import nltk
from nltk.tag import pos_tag
import random

def create_labeled_data():
    news_api = NewsApiClient(api_key='063f02817dbb49528058d7372964f645')
    buzzfeed_headlines = []
    reuters_headlines = []
    x = 1
    while x <= 10:
        b_headlines = news_api.get_everything(sources='buzzfeed',
                                              from_param='2018-11-10',
                                              to='2018-12-09',
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100,
                                              page=x)['articles']
        r_headlines = news_api.get_everything(sources='reuters',
                                              from_param='2018-11-10',
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
	featureset['adjcount'] = adjcount(headline)
	return featureset
	
	
def pcount(headline):
	pronouns = ["we","you","I","everyone"]
	for w in headline:
		if w.lower() in pronouns:
			return True
	return False

def punctcount(headline):
	count = 0
	punct = [".","!","?"]
	for w in headline.split():
		if w in punct:
			count += 1
	return count
	
def adjcount(headline):
	count = 0
	for tag in pos_tag(headline):
		if tag[1] == 'JJ':
			count += 1
	return count
	
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
