"""
File to fill with fun clickbait analysis!
Love, Lucino
"""

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import random
import pickle
from collections import Counter
from nltk.corpus import stopwords
from headline import Headline

common_bigrams = []
splitpoint = 2700

"""
Opens the pickled file and shuffles it up, returning a list of labeled headlines
"""
def create_headlines():
    headline_tuples = pickle.load(open('headlines.p', 'rb'))
    headlines = [Headline(h[0], h[1]) for h in headline_tuples]
    random.shuffle(headlines)
    return headlines

def create_feature_sets(headlines):
    common_bigrams.extend(get_bigrams(headlines))
    feature_sets = [(bait_features(headline), headline.label) for headline in headlines]
    train_set, test_set = feature_sets[:splitpoint], feature_sets[splitpoint:]
    return train_set, test_set


#pulls out most frequent bigrams from the training set each time (they are almost always the same!)
def get_bigrams(headlines):
    all_bgrams = []
    training = [h.tokens_lower for h in headlines[:splitpoint] if h.label == 'bait']
    for h in training:
        all_bgrams.extend(list(nltk.bigrams(h)))
    fdist = nltk.FreqDist(all_bgrams)
    most_common = fdist.most_common(25)
    common_bigrams = [x[0] for x in most_common]
    return common_bigrams


def bait_features(headline):
    featureset = {}
    featureset['procount'] = procount(headline)
    featureset['punct'] = punct(headline)
    featureset['averagewordlength'] = averagewordlength(headline)
    featureset['mostcommontag'] = mostcommontag(headline)
    featureset['wh'] = wh(headline)
    featureset['startswithnum'] = startswithnum(headline)
    featureset['superlative'] = superlative(headline)
    featureset['imperative'] = imperative(headline)
    featureset['bigrams'] = bigrams(headline)
    featureset['function_words'] = function_words(headline)
    featureset['flag_words'] = flag_words(headline)
    return featureset

# Checks for the use of first and second-person pronouns in article headline; returns true if any found.
def procount(headline):
    pronouns = ["we", "you", "i", "everyone", "us", "your", "our"]
    for w in headline.tokens_lower:
        if w in pronouns:
            return True
    return False


# Checks end-of-sentence punctuation count in headline; returns true if count is greater than 0.
def punct(headline):
    punct = [".", "!", "?", ","]
    found = False
    for w in headline.tokens:
        if w in punct:
            found = True
    return found


# Checks whether the headline starts with a digit; returns true if so.
def startswithnum(headline):
    tags = [w[1] for w in headline.pos_tagged]
    if tags[0] == 'CD':
        return True
    return False


# Calculates average word length within headline; returns true if the average is greater than 4.
def averagewordlength(headline):
    charactercount = 0
    wordcount = 0
    commonshortwords = ["the", "a", "for", "an", "of", "and", "so", "but", "with", ",", ".", ":", ";"]
    for w in headline.tokens_lower:
        if w not in commonshortwords:
            charactercount += len(w)
            wordcount += 1
    avg = charactercount / wordcount
    return avg < 4


# Checks for the most common POS tag in the headline; returns true if the most common tag is NN.
def mostcommontag(headline):
    counts = Counter(headline.pos_tags)
    return counts.most_common()[0][0] == "NN"


# Checks for the use of superlative adjectives in the headline; returns true if any found.
def superlative(headline):
    for tag in headline.pos_tags:
        if tag == 'JJS' or tag == 'RBS':
            return True
    return False


# Checks for the use of wh-words in the headline; returns true if any found.
def wh(headline):
    for tag in headline.pos_tags:
        if tag == 'WP':
            return True
    return False


# Checks whether the first word in a headline is tagged as a bare-form verb, indicating an imperative;
# returns true if it is.
def imperative(headline):
    if headline.pos_tags[0] == 'VB':
        return True
    return False


# Checks all bigrams in the headline, and compares them against a list of most common clickbait bigrams. Returns
# true if any match.
def bigrams(headline):
    bigrams = nltk.bigrams(headline.tokens_lower)

    for x in bigrams:
        if x in common_bigrams:
            return True
    return False

# Calculates proportion of word in headline that are stopwords or function words
def function_words(headline):
    fun_words = [w for w in headline.tokens_lower if w in stopwords.words('english')]
    #return len(fun_words)/headline.num_tokens
    return True if ((len(fun_words)/headline.num_tokens) == .5) else False

def flag_words(headline):
    flags = ['this', 'will', 'believe', 'surprise']
    found = False
    for word in headline.tokens_lower:
        if word in flags:
            found = True
    return found

# def get_common_words(headlines):
#     #get list of all tokens in corpus
#     corpus_tokens = []
#     for h in headlines:
#         corpus_tokens.extend(h.tokens_lower)
#     #return most common words
#     fdist = nltk.FreqDist(corpus_tokens)
#     half = int(len(fdist)/2)
#     common_words = fdist.most_common(half)
#     return common_words
#
# def rare_words(headline):
#     rare = 0
#     for word in headline.tokens_lower:
#         if word not in common_words:
#             rare += 1
#     return rare

def train_classifier(training_set):
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    return classifier


def evaluate_classifier(classifier, test_set):
    print(nltk.classify.accuracy(classifier, test_set))


if __name__ == '__main__':

    headlines = create_headlines()
    training_set, test_set = create_feature_sets(headlines)
    classifier = train_classifier(training_set)
    evaluate_classifier(classifier, test_set)
    classifier.show_most_informative_features()