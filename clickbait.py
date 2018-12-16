"""
File to fill with fun clickbait analysis!
Love, Lucino
"""

from newsapi import NewsApiClient
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import random
from collections import Counter
from nltk.corpus import stopwords


##not sure where to put this code or how to condense it, but basically this section is where i grab the most 
#common bigrams from the buzzfeed/mtv news corpus, so we can use them in the bigrams feature later.
news_api = NewsApiClient(api_key='8c032ece9401408182134442e81ebfc9')
buzzfeed_headlines = []
x = 1
while x <= 10:
	b_headlines = news_api.get_everything(sources='buzzfeed, mtv-news',from_param='2018-11-16',to='2018-12-15',language='en',sort_by='relevancy',page_size=100,page=x)['articles']
	b_titles = [article['title'] for article in b_headlines]
	b_titles = list(filter(None.__ne__,b_titles))
	buzzfeed_headlines.extend(b_titles)
	x += 1
raw = " ".join(buzzfeed_headlines)
tokens = raw.split(" ")
bigrams = list(nltk.bigrams(tokens))
fdist = nltk.FreqDist(bigrams)
most_common = fdist.most_common(25)
common_bigrams = [x[0] for x in most_common]

#Pulls two lists of articles from newsAPI, extracting titles only: one consisting of Buzzfeed and MTV News headlines, 
#bulk-labeled as clickbait, and another consisting of Reuters and Associated Press headlines, bulk-labeled as non-clickbait.

def create_labeled_data():
	news_api = NewsApiClient(api_key='8c032ece9401408182134442e81ebfc9')
	buzzfeed_headlines = []
	reuters_headlines = []
	x = 1
	while x <= 10:
		b_headlines = news_api.get_everything(sources='buzzfeed, mtv-news',from_param='2018-11-16',to='2018-12-15',language='en',sort_by='relevancy',page_size=100,page=x)['articles']
		r_headlines = news_api.get_everything(sources='reuters, associated-press',from_param='2018-11-16',to='2018-12-15',language='en',sort_by='relevancy',page_size=100,page=x)['articles']
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
	print(len(featuresets))
	train_set, test_set = featuresets[:1700], featuresets[1700:]
	return train_set, test_set

def bait_features(headline):
	featureset = {}
	featureset['procount'] = procount(headline)
	featureset['punct'] = punct(headline)
	featureset['averagewordlength'] = averagewordlength(headline)
	featureset['mostcommontag'] = mostcommontag(headline)
	featureset['wh'] = wh(headline)
	featureset['startswithnum'] = startswithnum(headline)
	featureset['imperative'] = imperative(headline)
	featureset['bigrams'] = bigrams(headline)
	return featureset
	
#Checks for the use of first and second-person pronouns in article headline; returns true if any found.
def procount(headline):
	pronouns = ["we","you","i","everyone","us","your","our"]
	for w in word_tokenize(headline):
		if w.lower() in pronouns:
			return True
	return False

#Checks end-of-sentence punctuation count in headline; returns true if count is greater than 0.
def punct(headline):
	count = 0
	punct = [".","!","?"]
	for w in word_tokenize(headline):
		if w in punct:
			count += 1
	return count > 0

#Checks whether the headline starts with a digit; returns true if so.
def startswithnum(headline):
	tags = [w[1] for w in pos_tag(word_tokenize(headline))]
	if tags[0] == 'CD':
		return True
	return False

#Calculates average word length within headline; returns true if the average is greater than 4.
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

#Checks for the most common POS tag in the headline; returns true if the most common tag is NN.
def mostcommontag(headline):
	tags = [w[1] for w in pos_tag(word_tokenize(headline))]
	counts = Counter(tags)
	return counts.most_common()[0][0] == "NN"

#Checks for the use of superlative adjectives in the headline; returns true if any found.
def superlative(headline):
	tags = [w[1] for w in pos_tag(word_tokenize(headline))]
	for tag in tags:
		if tag == 'JJS' or tag == 'RBS':
			return True
	return False

#Checks for the use of wh-words in the headline; returns true if any found.
def wh(headline):
	tags = [w[1] for w in pos_tag(word_tokenize(headline))]
	for tag in tags:
		if tag == 'WP':
			return True
	return False

#Checks whether the first word in a headline is tagged as a bare-form verb, indicating an imperative;
#returns true if it is.
def imperative(headline):
	tags = [w[1] for w in pos_tag(word_tokenize(headline))]
	if tags[0] == 'VB':
		return True
	return False

#Checks all bigrams in the headline, and compares them against a list of most common clickbait bigrams. Returns
#true if any match.
def bigrams(headline):
	bigrams = nltk.bigrams(headline.split(" "))
	for x in bigrams:
		if x in common_bigrams:
			return True
	return False

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